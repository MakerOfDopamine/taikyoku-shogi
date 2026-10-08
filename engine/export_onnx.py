"""Export checkpoint.pt to ONNX for the C self-play driver.

A checkpoint of model_policy.TaikyokuShogiBot (one with a policy head) also gets a policy
graph per precision, net_<p>_policy.onnx: inputs `board` (1, 1296) int32 as below and
`moves` (M, 3) int32 rows of (from square, to square, special), in the board's canonical
frame; output `logits` (M,). nnplay ranks every legal move with it and has the value head
score only the top few. The value graph of such a model returns `value` alone.

Writes two graphs from the same weights: net_fp32.onnx and net_fp16.onnx, each put through
ONNX Runtime's offline transformer fusion so the 1296-token attention runs as a fused
kernel instead of a materialised 1296x1296 score matrix. Both take a
single input `board`, an int32 tensor of shape (B, 1296) holding embedding indices in
0..602 -- unpack.embedding_index of a board already canonicalised to the side to move --
and return `value` and `material`, each (B, 1). The batch axis is dynamic; nnplay.c
sub-batches to whatever the GPU takes.

The graph is the eval-mode network, so BatchNorm is folded to its running statistics and
dropout is off. Parity against torch is checked here and again by `nnplay eval`.
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

NSQ = 36 * 36
NUM_EMBEDDING = 603          # unpack.embedding_index range: 0 empty, then index*2 + colour


class Wrapped(torch.nn.Module):
    """(B, 1296) int32 in, so the C side ships one contiguous row per board."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, board):
        # the policy model's forward gives (value, None) without moves; legacy (value, material)
        return tuple(t for t in self.net(board.to(torch.int64)) if t is not None)


class PolicyWrapped(torch.nn.Module):
    """board (1, 1296) and moves (M, 3) int32 in, one logit per move out."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, board, moves):
        m = moves.to(torch.int64)
        rows = torch.zeros_like(m[:, :1])            # every move belongs to board 0
        return self.net(board.to(torch.int64), torch.cat([rows, m], 1))[1]


def has_policy(net):
    return hasattr(net, "unpatch")


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    if any(k.startswith("unpatch.") for k in sd):
        from model_policy import TaikyokuShogiBot
    else:
        from model import TaikyokuShogiBot
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


def export(net, out, dtype, device, opset, bert_shaped):
    inner = copy.deepcopy(net)
    if bert_shaped:
        rewrite(inner)
    m = Wrapped(inner).to(device=device, dtype=dtype).eval()
    dummy = torch.zeros((2, NSQ), dtype=torch.int32, device=device)
    outs = ["value"] if has_policy(net) else ["value", "material"]
    with torch.no_grad():
        torch.onnx.export(
            m, (dummy,), out,
            input_names=["board"], output_names=outs,
            dynamic_axes={"board": {0: "batch"}, **{o: {0: "batch"} for o in outs}},
            opset_version=opset, do_constant_folding=True,
        )
    return out


def export_policy(net, out, dtype, device, opset, bert_shaped):
    inner = copy.deepcopy(net)
    if bert_shaped:
        rewrite(inner)
    m = PolicyWrapped(inner).to(device=device, dtype=dtype).eval()
    board = torch.zeros((1, NSQ), dtype=torch.int32, device=device)
    moves = torch.zeros((5, 3), dtype=torch.int32, device=device)
    with torch.no_grad():
        torch.onnx.export(
            m, (board, moves), out,
            input_names=["board", "moves"], output_names=["logits"],
            dynamic_axes={"moves": {0: "moves"}, "logits": {0: "moves"}},
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
    from unpack import INIT_BOARD
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoint.pt")
    ap.add_argument("--baseline", default="baseline.pt",
                    help="a second checkpoint to export as baseline_*.onnx, for `nnplay elo`. "
                         "Skipped without complaint when it is the default and absent; an error "
                         "when named explicitly and absent.")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check-batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=4)
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
    probe = torch.from_numpy(rng.integers(0, NUM_EMBEDDING, (a.check_batch, NSQ), dtype=np.int32)).to(device)

    def run(net_, dtype, bert_shaped=False):
        # In chunks: without the fused kernel a 1296-token layer holds an 8 x 1296 x 1296
        # score matrix per board, so the check batch does not fit whole on a small card.
        inner = copy.deepcopy(net_)
        if bert_shaped:
            rewrite(inner)
        m = Wrapped(inner).to(device=device, dtype=dtype).eval()
        out = [[], []]
        with torch.no_grad():
            for at in range(0, probe.shape[0], a.chunk):
                for k, t in enumerate(m(probe[at:at + a.chunk])):
                    out[k].append(t.float().cpu().numpy().ravel())
        v = np.concatenate(out[0])
        mat = np.concatenate(out[1]) if out[1] else np.zeros_like(v)   # no material head
        return v, mat

    pol_moves = torch.from_numpy(np.stack([rng.integers(0, NSQ, 600), rng.integers(0, NSQ, 600),
                                           rng.integers(0, 10, 600)], 1).astype(np.int32)).to(device)

    def run_policy(net_, dtype, bert_shaped=False):
        inner = copy.deepcopy(net_)
        if bert_shaped:
            rewrite(inner)
        m = PolicyWrapped(inner).to(device=device, dtype=dtype).eval()
        with torch.no_grad():
            return m(probe[:1], pol_moves).float().cpu().numpy()

    # TransformerEncoderLayer's fused eval-mode fast path is one fat aten op with no ONNX
    # symbolic; turning it off recovers the decomposed graph. Same arithmetic, not fused,
    # so measure what the unfusing itself costs before using the decomposed run as the
    # reference every other number is quoted against.
    fused_v, fused_m = run(net, torch.float32)
    torch.backends.mha.set_fastpath_enabled(False)
    ref_v, ref_m = run(net, torch.float32)
    print(f"fused vs decomposed attention (both fp32): max|dv| = {np.abs(fused_v - ref_v).max():.3e}")

    nhead, hidden = attention_shape(net)
    print(f"encoder: {len(net.transformers.layers)} layers, {nhead} heads, d_model {hidden}")
    bert_shaped = not a.packed_attention
    if bert_shaped:
        bv, bm = run(net, torch.float32, bert_shaped=True)
        print(f"BERT-shaped attention rewrite (both fp32): max|dv| = {np.abs(bv - ref_v).max():.3e}")

    policy = has_policy(net)
    ref_p = run_policy(net, torch.float32) if policy else None
    print("policy head: " + ("yes -- exporting a policy graph per precision" if policy
                            else "none (a legacy model): nnplay play/elo/variance will refuse it"))
    manifest = {"checkpoint": a.checkpoint, "epoch": epoch, "opset": a.opset,
                "num_embedding": NUM_EMBEDDING, "nsq": NSQ, "policy": policy,
                "bert_shaped_attention": bert_shaped, "fused": not a.no_fuse, "graphs": {}}
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        path = f"{a.out_dir}/net_{name}.onnx"
        export(net, path, dtype, device, a.opset, bert_shaped)
        raw = None
        if bert_shaped and not a.no_fuse:
            # keep the pre-fusion graph: it is plain ONNX with no contrib ops, so it is the
            # thing to re-fuse with if some future runtime will not load the fused one
            raw = path.replace(".onnx", "_raw.onnx")
            shutil.copyfile(path, raw)
        nfused = fuse(path, nhead, hidden, True) if bert_shaped and not a.no_fuse else 0
        v, mat = run(net, dtype, bert_shaped)
        dv = float(np.abs(v - ref_v).max())
        dm = float(np.abs(mat - ref_m).max())
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        manifest["graphs"][name] = {"path": path, "raw": raw, "sha256_16": digest,
                                    "attention_fused": nfused,
                                    "max_abs_dev_value": dv, "max_abs_dev_material": dm}
        print(f"wrote {path}  sha={digest}  max|dv| vs fp32 torch = {dv:.3e}  max|dm| = {dm:.3e}")
        if policy:
            ppath = f"{a.out_dir}/net_{name}_policy.onnx"
            export_policy(net, ppath, dtype, device, a.opset, bert_shaped)
            praw = None
            if bert_shaped and not a.no_fuse:
                praw = ppath.replace(".onnx", "_raw.onnx")
                shutil.copyfile(ppath, praw)
            pf = fuse(ppath, nhead, hidden, True) if bert_shaped and not a.no_fuse else 0
            dp = float(np.abs(run_policy(net, dtype, bert_shaped) - ref_p).max())
            # and through ONNX Runtime itself: the move gather is new in this graph
            import onnxruntime as ort
            sess = ort.InferenceSession(ppath, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            got = sess.run(None, {"board": probe[:1].cpu().numpy(), "moves": pol_moves.cpu().numpy()})[0]
            dpo = float(np.abs(got.astype(np.float32) - ref_p).max())
            manifest["graphs"][name + "_policy"] = {"path": ppath, "raw": praw, "attention_fused": pf,
                                                    "max_abs_dev_torch": dp, "max_abs_dev_onnxruntime": dpo}
            print(f"wrote {ppath}  fused {pf}  max|dlogit| vs fp32 torch: torch {dp:.3e}, "
                  f"onnxruntime {dpo:.3e}  (logit spread {ref_p.std():.3f})")

    # The value of the start position, so a bundle can verify a runtime reproduces *these*
    # weights. It has to travel with the graph: baking it into setup.sh at bundle time made
    # the check fail the moment the model was replaced, which is the normal thing to do.
    from unpack import INIT_BOARD, canonical_board, embedding_index
    start = torch.from_numpy(
        embedding_index(canonical_board(INIT_BOARD.reshape(36, 36), 1)).reshape(1, -1)
        .astype(np.int32)).to(device)
    with torch.no_grad():
        sv = float(Wrapped(copy.deepcopy(net)).to(device=device, dtype=torch.float32).eval()(start)[0])
    manifest["selfcheck"] = {"start_position_value": sv, "tolerance": 3e-3,
                             "note": "fp32 torch; fp16 and TF32 both land well inside the tolerance"}
    print(f"start-position value (fp32 torch) = {sv:+.9f}")
    if policy:
        mv = start_position_moves()
        with torch.no_grad():
            lg = PolicyWrapped(copy.deepcopy(net)).to(device).eval()(
                start, torch.from_numpy(mv).to(device)).float().cpu().numpy()
        manifest["selfcheck"]["start_position_policy_moves"] = mv.ravel().tolist()
        manifest["selfcheck"]["start_position_policy_logits"] = [float(x) for x in lg]
        manifest["selfcheck"]["policy_tolerance"] = 0.05
        print(f"start-position policy (fp32 torch): {len(mv)} moves, logits {lg.min():+.3f}..{lg.max():+.3f}")

    # The Elo baseline: the same graphs from a second checkpoint, so `nnplay elo` can rate
    # this network against the previous one instead of against random.
    explicit = "--baseline" in sys.argv
    if os.path.exists(a.baseline):
        bnet, bepoch = load(a.baseline, device)
        print(f"baseline {a.baseline} (epoch {bepoch})")
        manifest["baseline"] = {"checkpoint": a.baseline, "epoch": bepoch, "graphs": {}}
        for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
            path = f"{a.out_dir}/baseline_{name}.onnx"
            export(bnet, path, dtype, device, a.opset, bert_shaped)
            nf = fuse(path, nhead, hidden, True) if bert_shaped and not a.no_fuse else 0
            manifest["baseline"]["graphs"][name] = {"path": path, "attention_fused": nf}
            print(f"wrote {path}  fused {nf}")
            if has_policy(bnet):
                ppath = f"{a.out_dir}/baseline_{name}_policy.onnx"
                export_policy(bnet, ppath, dtype, device, a.opset, bert_shaped)
                nf = fuse(ppath, nhead, hidden, True) if bert_shaped and not a.no_fuse else 0
                manifest["baseline"]["graphs"][name + "_policy"] = {"path": ppath, "attention_fused": nf}
                print(f"wrote {ppath}  fused {nf}")
        if not has_policy(bnet):
            print(f"  {a.baseline} has no policy head: nnplay elo will refuse it as a baseline")
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
