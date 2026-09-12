"""Validation for the nnplay self-play pipeline.

Four independent checks, all against code that is not nnplay.c:

  encode    nnplay's encode_stm vs unpack.canonical_board + unpack.embedding_index, on
            real positions of both colours from the existing corpus
  replay    every stored board is the previous board with the stored move applied, using
            unpack.apply_move -- which is game.js's Board.move, not nnplay's do_move
  values    every stored value re-evaluated through the network reproduces the stored one,
            which is the claim that a record's value belongs to that record's board
  torch     nnplay eval vs torch on the same boards, so the ONNX export, the attention
            fusion and the C plumbing are covered end to end

Run it as `python3 test_nnplay.py SHARD.tkn`; without an argument it plays a short game
first.
"""
import subprocess
import sys
import tempfile

import numpy as np

from unpack import (unpack, apply_move, canonical_board, embedding_index, EMPTY_IDX,
                    INIT_BOARD, rot180_move)
from unpack_nn import unpack_nn, games, NSQ

NNPLAY = "./nnplay"
fails = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def nnplay(args, stdin=None):
    r = subprocess.run([NNPLAY] + args, input=stdin, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{' '.join(args)} failed:\n{r.stderr[-2000:]}")
    return r.stdout


def test_encode(n=40):
    """The C and python canonicalisations must agree square for square, both colours."""
    bad = seen = 0
    sides = set()
    for board, info in unpack("shard_0000.tkp", canonical=False, limit=n):
        sides.add(int(info.side_to_move))
        flat = board.reshape(-1)
        lines = ["clear"] + [f"{sq} {c>>2} {(c>>1)&1} {c&1}"
                             for sq, c in enumerate(int(v) for v in flat) if (c >> 2) != EMPTY_IDX]
        lines.append(f"turn {info.side_to_move}")
        got = np.array([int(v) for v in nnplay(["encode"], "\n".join(lines) + "\n").split()])
        want = embedding_index(canonical_board(board, info.side_to_move)).reshape(-1)
        seen += 1
        bad += not np.array_equal(got, want)
    check("encode_stm == canonical_board + embedding_index", bad == 0 and len(sides) == 2,
          f"{seen} positions, both sides seen, {bad} mismatches")


def test_replay(path):
    """Board n must be board n-1 with move n applied, by unpack.apply_move."""
    bad = seen = 0
    cur = None
    last_game = None
    for board, info in unpack_nn(path, canonical=False):
        if info.game_index != last_game:
            cur, last_game = INIT_BOARD.copy(), info.game_index
        mv = (info.move["special"] << 28) | (info.move["origin"] << 17) | (info.move["target"] << 6)
        apply_move(cur, mv)
        seen += 1
        bad += not np.array_equal(cur, board.reshape(-1))
    check("stored boards replay from the stored moves", bad == 0, f"{seen} plies, {bad} mismatches")


def test_values(path, precision, extra, limit=64):
    """Feed the stored boards back through the network and expect the stored values."""
    rows, want = [], []
    for board, info in unpack_nn(path, limit=limit):
        rows.append(embedding_index(board).reshape(-1))
        want.append(info.value)
    out = nnplay(["eval", "--precision", precision, "--quiet"] + extra,
                 " ".join(map(str, np.stack(rows).ravel())))
    got = np.array([float(x) for x in out.split()])
    want = np.array(want)
    d = np.abs(got - want).max() if got.shape == want.shape else float("inf")
    check(f"stored value == re-evaluated value ({precision})", d < 3e-3,
          f"{len(want)} plies, max deviation {d:.2e}")


def test_torch(precision, extra, n=16):
    """nnplay eval against torch, on the same boards."""
    import torch
    from model import TaikyokuShogiBot
    torch.backends.mha.set_fastpath_enabled(False)
    net = TaikyokuShogiBot().eval()
    net.load_state_dict(torch.load("checkpoint.pt", map_location="cpu", weights_only=False)["model"])
    rows = np.stack([embedding_index(b).reshape(-1) for b, _ in unpack("shard_0000.tkp", limit=n)])
    with torch.no_grad():
        want = np.concatenate([net(torch.from_numpy(rows[i:i + 4]).long())[0].numpy().ravel()
                               for i in range(0, len(rows), 4)])
    out = nnplay(["eval", "--precision", precision, "--quiet"] + extra,
                 " ".join(map(str, rows.astype(np.int32).ravel())))
    got = np.array([float(x) for x in out.split()])
    d = np.abs(got - want).max() if got.shape == want.shape else float("inf")
    tol = 1e-5 if extra else 3e-3
    check(f"nnplay eval == torch ({precision}{' '.join([''] + extra)})", d < tol,
          f"{n} boards, max deviation {d:.2e}")


def test_elo_estimator():
    """The rating posterior against closed-form logistic Elo, and on cases it cannot do.

    `nnplay elotest` runs the estimator alone on given counts and opens no ONNX session,
    so this costs nothing and needs no GPU.
    """
    import math

    def fit(wb, lb, db, ww, lw, dw):
        out = nnplay(["elotest", *map(str, (wb, lb, db, ww, lw, dw))])
        f = out.split()
        return {"elo": float(f[1]), "lo": float(f[2].strip("[,")), "hi": float(f[3].strip("]")),
                "draw": float(f[5]), "adv": float(f[7]), "sat": int(f[9])}

    bad = []
    # symmetric colours, no draws: the model reduces to the logistic curve
    for w, l in ((25, 25), (75, 25), (45, 5), (300, 200)):
        got = fit(w, l, 0, w, l, 0)
        want = -400.0 * math.log10(1.0 / ((2 * w) / (2 * w + 2 * l)) - 1.0)
        # a few Elo high is expected: draw_elo is constrained non-negative, so with no draws
        # observed the mass at draw_elo > 0 inflates the rating slightly. BayesElo does this too.
        if not (want - 2 <= got["elo"] <= want + 20):
            bad.append(f"{w}W {l}L: posterior {got['elo']:+.0f} vs closed form {want:+.0f}")
        if not (got["lo"] < got["elo"] < got["hi"]):
            bad.append(f"{w}W {l}L: {got['elo']:+.0f} outside [{got['lo']:+.0f}, {got['hi']:+.0f}]")
    check("elo posterior == closed-form logistic Elo", not bad, "; ".join(bad) or "4 records")

    # the interval must narrow as sqrt(n)
    a, b = fit(25, 25, 0, 25, 25, 0), fit(250, 250, 0, 250, 250, 0)
    ratio = (a["hi"] - a["lo"]) / (b["hi"] - b["lo"])
    check("elo interval narrows as sqrt(n)", 2.7 < ratio < 3.7,
          f"10x the games narrows it {ratio:.2f}x (sqrt(10) = 3.16)")

    # a pure first-move effect must land on the advantage, not on the rating
    f = fit(45, 5, 0, 5, 45, 0)
    check("first-move effect is not read as strength", abs(f["elo"]) <= 5 and f["adv"] > 150,
          f"elo {f['elo']:+.0f}, advantage {f['adv']:+.0f}")

    # an unbeaten record cannot be bounded above, and must say so
    f = fit(50, 0, 0, 50, 0, 0)
    check("an unbeaten record is flagged saturated", f["sat"] == 1 and f["lo"] > 500,
          f"elo {f['elo']:+.0f} [{f['lo']:+.0f}, {f['hi']:+.0f}], saturated {f['sat']}")

    # draws should be absorbed by the draw rating, not by the strength
    f = fit(30, 30, 40, 30, 30, 40)
    check("draws land on the draw rating", abs(f["elo"]) <= 5 and f["draw"] > 50,
          f"elo {f['elo']:+.0f}, draw rating {f['draw']:.0f}")


def test_symmetry(path):
    """A rotated move code must land where the rotated board says it does.

    Cheap consistency check on the canonicalisation of the move alongside the board: the
    absolute and canonical readers must agree about which square the mover left.
    """
    bad = seen = 0
    for (ba, ia), (bc, ic) in zip(unpack_nn(path, canonical=False), unpack_nn(path)):
        o, m = rot180_move(ia.move["origin"], ia.move["special"] * 1296 + ia.move["target"])
        want = (ia.move["origin"], ia.move["special"], ia.move["target"]) if ia.side == 0 else \
               (o, m // 1296, m % 1296)
        seen += 1
        bad += (ic.move["origin"], ic.move["special"], ic.move["target"]) != want
        bad += not np.array_equal(bc, canonical_board(ba, 1 - ia.side))
    check("canonical move and board agree", bad == 0, f"{seen} plies, {bad} mismatches")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    tmp = None
    if path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".tkn", delete=False)
        path = tmp.name
        print(f"playing a short game into {path}")
        nnplay(["play", "--games", "2", "--max-plies", "6", "--quiet", "--out", path])

    n_games = len(list(games(path)))
    n_plies = sum(1 for _ in unpack_nn(path))
    print(f"{path}: {n_games} games, {n_plies} plies")

    test_elo_estimator()
    test_encode()
    test_replay(path)
    test_symmetry(path)
    test_values(path, "fp16", [])
    test_torch("fp16", [])
    test_torch("fp32", ["--no-tf32"])

    print(f"\n{len(fails)} failed" if fails else "\nall checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
