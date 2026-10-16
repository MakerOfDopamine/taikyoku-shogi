"""Export a model_policy.py checkpoint to ONNX for the C self-play driver.

One graph per precision, net_fp32.onnx and net_fp16.onnx, with the value and policy heads
in a single pass of the trunk:

    board   int32 (B, 1296)  embedding indices 0..602 -- unpack.embedding_index of a board
                             canonicalised to the side to move
    moves   int32 (M, 4)     rows of (board row, from square, to square, special), each in
                             its own board's canonical frame
    value   (B,)             for the side to move, in [-1, 1]
    logits  (M,)             one per move row

nnplay evaluates a whole batch of search leaves in one call: every leaf's board, and all of
their legal moves with the row of the leaf they belong to. Both axes are dynamic.

Each graph goes through ONNX Runtime's offline transformer fusion, and the pre-fusion graph
is kept beside it as net_<p>_raw.onnx. The graph is the eval-mode network. Parity against
torch is checked here, through onnxruntime, and again by `nnplay selfcheck`.
"""
import argparse
import copy
import os
import sys
import json
import hashlib
import shutil

import numpy as np
import torch

from export_rewrite import rewrite, attention_shape
from unpack import INIT_BOARD, canonical_board, embedding_index

NSQ = 36 * 36
NUM_EMBEDDING = 603          # unpack.embedding_index range: 0 empty, then index*2 + colour


class Wrapped(torch.nn.Module):
    """int32 in, so the C side ships contiguous rows without a conversion."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, board, moves):
        value, logits = self.net(board.to(torch.int64), moves.to(torch.int64))
        return value.reshape(-1), logits


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    if not any(k.startswith("unpatch.") for k in sd):
        sys.exit(f"{path}: no policy head -- a material-head model.py checkpoint, which is no "
                 f"longer supported")
    from model_policy import TaikyokuShogiBot
    net = TaikyokuShogiBot().to(device)
    net.load_state_dict(sd)
    net.eval()
    return net, ckpt.get("epoch") if isinstance(ckpt, dict) else None


def fuse(path, nhead, hidden, verbose):
    """Fold the attention pattern into com.microsoft.Attention, offline.

    The C runtime's own load-time transformers do not do this, so it happens here. The
    onnxruntime python package must be the same version as the C library in
    third_party/onnxruntime, since a contrib op is only guaranteed loadable by the runtime
    that defines it; export_onnx checks and says so if they differ.
    """
    from onnxruntime.transformers import optimizer as ort_opt
    from onnxruntime.transformers.fusion_options import FusionOptions

    opt = ort_opt.optimize_model(path, model_type="bert", num_heads=nhead,
                                 hidden_size=hidden, opt_level=0,
                                 optimization_options=FusionOptions("bert"))
    stats = opt.get_fused_operator_statistics()
    fused = stats.get("Attention", 0) + stats.get("MultiHeadAttention", 0)
    opt.save_model_to_file(path)
    if verbose:
        print(f"  fused {fused} attention blocks, {stats.get('SkipLayerNormalization', 0)} skip-layernorms")
    return fused


def wrapped(net, dtype, device, bert_shaped):
    inner = copy.deepcopy(net)
    if bert_shaped:
        rewrite(inner)
    return Wrapped(inner).to(device=device, dtype=dtype).eval()


def export(net, out, dtype, device, opset, bert_shaped):
    m = wrapped(net, dtype, device, bert_shaped)
    board = torch.zeros((2, NSQ), dtype=torch.int32, device=device)
    moves = torch.tensor([[0, 0, 1, 0], [1, 2, 3, 9], [1, 4, 5, 3]], dtype=torch.int32, device=device)
    with torch.no_grad():
        torch.onnx.export(
            m, (board, moves), out,
            input_names=["board", "moves"], output_names=["value", "logits"],
            dynamic_axes={"board": {0: "batch"}, "moves": {0: "moves"},
                          "value": {0: "batch"}, "logits": {0: "moves"}},
            opset_version=opset, do_constant_folding=True,
        )
    return out


def start_position_moves():
    """The start position's legal moves in nnplay's generation order, as canonical
    (from, to, special) rows -- black is to move, so no rotation. Built through tkykids.c,
    the same generator nnplay runs, so selfcheck can match move for move."""
    import ctypes
    import subprocess
    import tempfile
    d = tempfile.mkdtemp()
    so = os.path.join(d, "libtkykids.so")
    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", so, "tkykids.c", "-lm"], check=True,
                   cwd=os.path.dirname(os.path.abspath(__file__)))
    lib = ctypes.CDLL(so)
    lib.tk_init()
    board = np.ascontiguousarray(INIT_BOARD, dtype=np.uint16)
    buf = np.zeros(8192, np.uint32)
    forced = ctypes.c_uint32()
    P = ctypes.POINTER
    lib.tk_moves.argtypes = [P(ctypes.c_uint16), ctypes.c_int, P(ctypes.c_uint32), ctypes.c_int,
                             P(ctypes.c_uint32)]
    n = lib.tk_moves(board.ctypes.data_as(P(ctypes.c_uint16)), 1, buf.ctypes.data_as(P(ctypes.c_uint32)),
                     len(buf), ctypes.byref(forced))
    mv = buf[:n].astype(np.int64)
    shutil.rmtree(d, ignore_errors=True)
    return np.stack([(mv >> 17) & 0x7FF, (mv >> 6) & 0x7FF, mv >> 28], 1).astype(np.int32)


def write_graphs(net, prefix, a, device, bert_shaped, nhead, hidden, run, ref, feeds, out):
    """net_<p>.onnx (fused) and net_<p>_raw.onnx for both precisions, each checked against
    the fp32 torch reference, in torch and through onnxruntime."""
    import onnxruntime as ort
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        path = f"{a.out_dir}/{prefix}_{name}.onnx"
        export(net, path, dtype, device, a.opset, bert_shaped)
        raw = None
        if bert_shaped and not a.no_fuse:
            # the pre-fusion graph is plain ONNX with no contrib ops: the thing to re-fuse
            # with if some future runtime will not load the fused one
            raw = path.replace(".onnx", "_raw.onnx")
            shutil.copyfile(path, raw)
        nfused = fuse(path, nhead, hidden, True) if bert_shaped and not a.no_fuse else 0
        v, lg = run(net, dtype, bert_shaped)
        sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        ov, olg = sess.run(None, feeds)
        dev = {"torch_value": float(np.abs(v - ref[0]).max()),
               "torch_logits": float(np.abs(lg - ref[1]).max()),
               "onnxruntime_value": float(np.abs(ov.astype(np.float32) - ref[0]).max()),
               "onnxruntime_logits": float(np.abs(olg.astype(np.float32) - ref[1]).max())}
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        out[name] = {"path": path, "raw": raw, "sha256_16": digest, "attention_fused": nfused,
                     "max_abs_dev": dev}
        print(f"wrote {path}  sha={digest}  vs fp32 torch: value torch {dev['torch_value']:.2e} "
              f"ort {dev['onnxruntime_value']:.2e}, logits torch {dev['torch_logits']:.2e} "
              f"ort {dev['onnxruntime_logits']:.2e}  (logit spread {ref[1].std():.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint.pt")
    ap.add_argument("--baseline", default="baseline.pt",
                    help="a second checkpoint to export as baseline_*.onnx, for `nnplay elo`. "
                         "Skipped without complaint when it is the default and absent; an error "
                         "when named explicitly and absent.")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check-batch", type=int, default=16)
    ap.add_argument("--no-fuse", action="store_true",
                    help="skip the offline attention fusion (slower, much more GPU memory)")
    ap.add_argument("--packed-attention", action="store_true",
                    help="export nn.MultiheadAttention as-is instead of the BERT-shaped rewrite "
                         "that ORT can fuse; slower and far heavier on GPU memory")
    a = ap.parse_args()

    if not a.no_fuse:
        import onnxruntime
        vf = "third_party/onnxruntime/VERSION_NUMBER"
        want = open(vf).read().strip() if os.path.exists(vf) else onnxruntime.__version__
        if onnxruntime.__version__ != want:
            print(f"WARNING: onnxruntime python is {onnxruntime.__version__} but the C library in "
                  f"third_party/onnxruntime is {want}. The fused contrib ops may not load; "
                  f"pip install onnxruntime=={want}, or pass --no-fuse.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, epoch = load(a.checkpoint, device)
    print(f"loaded {a.checkpoint} (epoch {epoch}), {sum(p.numel() for p in net.parameters())} params")

    rng = np.random.default_rng(0)
    B, M = a.check_batch, 50 * a.check_batch
    boards = rng.integers(0, NUM_EMBEDDING, (B, NSQ), dtype=np.int32)
    moves = np.stack([rng.integers(0, B, M), rng.integers(0, NSQ, M), rng.integers(0, NSQ, M),
                      rng.integers(0, 10, M)], 1).astype(np.int32)
    tb, tm = torch.from_numpy(boards).to(device), torch.from_numpy(moves).to(device)
    feeds = {"board": boards, "moves": moves}

    def run(net_, dtype, bert_shaped=False):
        with torch.no_grad():
            v, lg = wrapped(net_, dtype, device, bert_shaped)(tb, tm)
        return v.float().cpu().numpy(), lg.float().cpu().numpy()

    # TransformerEncoderLayer's fused eval-mode fast path is one fat aten op with no ONNX
    # symbolic; turning it off recovers the decomposed graph. Same arithmetic, not fused,
    # so measure what the unfusing itself costs before using the decomposed run as the
    # reference every other number is quoted against.
    fused_v, _ = run(net, torch.float32)
    torch.backends.mha.set_fastpath_enabled(False)
    ref = run(net, torch.float32)
    print(f"fused vs decomposed attention (both fp32): max|dv| = {np.abs(fused_v - ref[0]).max():.3e}")

    nhead, hidden = attention_shape(net)
    print(f"encoder: {len(net.transformers.layers)} layers, {nhead} heads, d_model {hidden}")
    bert_shaped = not a.packed_attention
    if bert_shaped:
        bv, blg = run(net, torch.float32, bert_shaped=True)
        print(f"BERT-shaped attention rewrite (both fp32): max|dv| = {np.abs(bv - ref[0]).max():.3e}, "
              f"max|dlogit| = {np.abs(blg - ref[1]).max():.3e}")

    manifest = {"checkpoint": a.checkpoint, "epoch": epoch, "opset": a.opset,
                "num_embedding": NUM_EMBEDDING, "nsq": NSQ,
                "bert_shaped_attention": bert_shaped, "fused": not a.no_fuse, "graphs": {}}
    write_graphs(net, "net", a, device, bert_shaped, nhead, hidden, run, ref, feeds, manifest["graphs"])

    # The start position's value and policy, so a bundle can verify a runtime reproduces
    # *these* weights. They travel with the graph: baking them into setup.sh at bundle time
    # made the check fail the moment the model was replaced, which is the normal thing to do.
    start = torch.from_numpy(embedding_index(canonical_board(INIT_BOARD.reshape(36, 36), 1))
                             .reshape(1, -1).astype(np.int32)).to(device)
    mv = start_position_moves()
    rows = torch.from_numpy(np.concatenate([np.zeros((len(mv), 1), np.int32), mv], 1)).to(device)
    with torch.no_grad():
        sv, slg = wrapped(net, torch.float32, device, False)(start, rows)
    sv, slg = float(sv[0]), slg.float().cpu().numpy()
    manifest["selfcheck"] = {"start_position_value": sv, "tolerance": 3e-3,
                             "start_position_policy_moves": mv.ravel().tolist(),
                             "start_position_policy_logits": [float(x) for x in slg],
                             "policy_tolerance": 0.05,
                             "note": "fp32 torch; fp16 and TF32 both land well inside the tolerances"}
    print(f"start position (fp32 torch): value {sv:+.9f}, {len(mv)} moves, logits "
          f"{slg.min():+.3f}..{slg.max():+.3f}")

    # The Elo baseline: the same graphs from a second checkpoint, so `nnplay elo` can rate
    # this network against the previous one.
    explicit = "--baseline" in sys.argv
    if os.path.exists(a.baseline):
        bnet, bepoch = load(a.baseline, device)
        print(f"baseline {a.baseline} (epoch {bepoch})")
        bref = run(bnet, torch.float32)
        manifest["baseline"] = {"checkpoint": a.baseline, "epoch": bepoch, "graphs": {}}
        write_graphs(bnet, "baseline", a, device, bert_shaped, nhead, hidden, run, bref, feeds,
                     manifest["baseline"]["graphs"])
    elif explicit:
        sys.exit(f"--baseline {a.baseline} does not exist")
    else:
        print(f"no {a.baseline}: skipping the Elo baseline "
              f"(copy a previous checkpoint.pt there to rate against it)")

    with open(f"{a.out_dir}/net_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {a.out_dir}/net_manifest.json")


if __name__ == "__main__":
    main()
