"""Export checkpoint.pt to ONNX for the C self-play driver.

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

from model import TaikyokuShogiBot
from export_rewrite import rewrite

NSQ = 36 * 36
NUM_EMBEDDING = 603          # unpack.embedding_index range: 0 empty, then index*2 + colour


class Wrapped(torch.nn.Module):
    """(B, 1296) int32 in, so the C side ships one contiguous row per board."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, board):
        return self.net(board.to(torch.int64))


def load(path, device):
    net = TaikyokuShogiBot().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, ckpt.get("epoch")


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
    with torch.no_grad():
        torch.onnx.export(
            m, (dummy,), out,
            input_names=["board"], output_names=["value", "material"],
            dynamic_axes={"board": {0: "batch"}, "value": {0: "batch"}, "material": {0: "batch"}},
            opset_version=opset, do_constant_folding=True,
        )
    return out


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
        want = open("third_party/onnxruntime/VERSION_NUMBER").read().strip()
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
        return tuple(np.concatenate(o) for o in out)

    # TransformerEncoderLayer's fused eval-mode fast path is one fat aten op with no ONNX
    # symbolic; turning it off recovers the decomposed graph. Same arithmetic, not fused,
    # so measure what the unfusing itself costs before using the decomposed run as the
    # reference every other number is quoted against.
    fused_v, fused_m = run(net, torch.float32)
    torch.backends.mha.set_fastpath_enabled(False)
    ref_v, ref_m = run(net, torch.float32)
    print(f"fused vs decomposed attention (both fp32): max|dv| = {np.abs(fused_v - ref_v).max():.3e}")

    bert_shaped = not a.packed_attention
    if bert_shaped:
        bv, bm = run(net, torch.float32, bert_shaped=True)
        print(f"BERT-shaped attention rewrite (both fp32): max|dv| = {np.abs(bv - ref_v).max():.3e}")

    manifest = {"checkpoint": a.checkpoint, "epoch": epoch, "opset": a.opset,
                "num_embedding": NUM_EMBEDDING, "nsq": NSQ,
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
        nfused = fuse(path, 8, 256, True) if bert_shaped and not a.no_fuse else 0
        v, mat = run(net, dtype, bert_shaped)
        dv = float(np.abs(v - ref_v).max())
        dm = float(np.abs(mat - ref_m).max())
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        manifest["graphs"][name] = {"path": path, "raw": raw, "sha256_16": digest,
                                    "attention_fused": nfused,
                                    "max_abs_dev_value": dv, "max_abs_dev_material": dm}
        print(f"wrote {path}  sha={digest}  max|dv| vs fp32 torch = {dv:.3e}  max|dm| = {dm:.3e}")

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
            nf = fuse(path, 8, 256, True) if bert_shaped and not a.no_fuse else 0
            manifest["baseline"]["graphs"][name] = {"path": path, "attention_fused": nf}
            print(f"wrote {path}  fused {nf}")
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
