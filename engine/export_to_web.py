"""Export a checkpoint's value head, and its policy head, to ONNX graphs for the browser.

    python3 export_to_web.py 1.pt                  # writes 1_web.onnx and 1_web_policy.onnx
    python3 export_to_web.py 1.pt -o value.onnx    # writes value.onnx and value_policy.onnx

Input `board`: int32 (B, 1296), one row per position in the frame of the side to move --
white to move rotated 180 degrees with every colour flipped (unpack.canonical_board) --
holding embedding indices 0..602 (unpack.embedding_index): 0 empty, else
piece_index * 2 + colour, colour 1 for the side to move. Square order is x * 36 + y.
Output `value`: float32 (B,), in [-1, 1], for the side to move.

The policy graph, <name>_policy.onnx, takes the same `board` and `moves`: int32 (M, 4)
rows of (board row in the batch, from square, to square, special), in that board's frame.
For white to move a square s becomes 1295 - s and a special k in 1..8 becomes
(k - 1 + 4) % 8 + 1; 0 and 9 stay. Output `logits`: float32 (M,), one per move, higher
for a better move for the side to move.

Unlike export_onnx.py the graphs are fp32 with no attention rewrite or ONNX Runtime
fusion: plain ONNX with no contrib ops, which is what onnxruntime-web's wasm and WebGPU
backends load.
"""
import argparse
import copy
import os

import numpy as np
import torch

from export_onnx import load
from unpack import INIT_BOARD, canonical_board, embedding_index

NSQ = 36 * 36
NUM_EMBEDDING = 603


class ValueOnly(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, board):
        return self.net(board.to(torch.int64))[0].reshape(-1)


class PolicyOnly(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, board, moves):
        return self.net(board.to(torch.int64), moves.to(torch.int64))[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint", help="a .pt checkpoint")
    ap.add_argument("-o", "--out", help="output graph (default <checkpoint>_web.onnx)")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.checkpoint)[0] + "_web.onnx"
    pout = (out[:-5] if out.endswith(".onnx") else out) + "_policy.onnx"

    net, epoch = load(a.checkpoint, "cpu")
    print(f"loaded {a.checkpoint} (epoch {epoch}), {sum(p.numel() for p in net.parameters())} params")
    # the encoder's fused eval-mode fast path is one aten op with no ONNX symbolic
    torch.backends.mha.set_fastpath_enabled(False)
    m = ValueOnly(copy.deepcopy(net)).eval()
    with torch.no_grad():
        torch.onnx.export(m, (torch.zeros((2, NSQ), dtype=torch.int32),), out,
                          input_names=["board"], output_names=["value"],
                          dynamic_axes={"board": {0: "batch"}, "value": {0: "batch"}},
                          opset_version=a.opset, do_constant_folding=True)
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    ops = sorted({n.domain for n in __import__("onnx").load(out).graph.node} - {""})
    if ops:
        print(f"WARNING: the graph uses non-standard op domains {ops}; onnxruntime-web may not load it")
    start = embedding_index(canonical_board(INIT_BOARD.reshape(36, 36), 1)).reshape(1, -1).astype(np.int32)
    rng = np.random.default_rng(0)
    for b in (1, 7):
        boards = np.concatenate([start, rng.integers(0, NUM_EMBEDDING, (b - 1, NSQ), dtype=np.int32)])
        with torch.no_grad():
            ref = m(torch.from_numpy(boards)).numpy()
        got = sess.run(None, {"board": boards})[0]
        assert got.shape == (b,), got.shape
        print(f"batch {b}: max |onnxruntime - torch| = {np.abs(got - ref).max():.2e}")
    print(f"start-position value = {float(got[0]):+.6f}  (black to move)")

    pm = PolicyOnly(copy.deepcopy(net)).eval()
    with torch.no_grad():
        torch.onnx.export(pm, (torch.zeros((2, NSQ), dtype=torch.int32),
                               torch.tensor([[0, 0, 1, 0], [1, 2, 3, 9], [1, 4, 5, 3]], dtype=torch.int32)),
                          pout, input_names=["board", "moves"], output_names=["logits"],
                          dynamic_axes={"board": {0: "batch"}, "moves": {0: "moves"},
                                        "logits": {0: "moves"}},
                          opset_version=a.opset, do_constant_folding=True)
    print(f"wrote {pout}  ({os.path.getsize(pout) / 1e6:.1f} MB)")
    psess = ort.InferenceSession(pout, providers=["CPUExecutionProvider"])
    ops = sorted({n.domain for n in __import__("onnx").load(pout).graph.node} - {""})
    if ops:
        print(f"WARNING: the graph uses non-standard op domains {ops}; onnxruntime-web may not load it")
    for b, m_ in ((1, 300), (3, 50)):
        boards = np.concatenate([start, rng.integers(0, NUM_EMBEDDING, (b - 1, NSQ), dtype=np.int32)])
        moves = np.stack([rng.integers(0, b, m_), rng.integers(0, NSQ, m_), rng.integers(0, NSQ, m_),
                          rng.integers(0, 10, m_)], 1).astype(np.int32)
        with torch.no_grad():
            ref = pm(torch.from_numpy(boards), torch.from_numpy(moves)).numpy()
        got = psess.run(None, {"board": boards, "moves": moves})[0]
        assert got.shape == (m_,), got.shape
        print(f"batch {b}, {m_} moves: max |onnxruntime - torch| = {np.abs(got - ref).max():.2e}  "
              f"(logit spread {ref.std():.2f})")


if __name__ == "__main__":
    main()
