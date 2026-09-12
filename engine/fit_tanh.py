"""Fits the tanh squash used to turn a material balance into a training label.

Run `tky sample` first: it plays a batch of games and writes samples.bin, one row per
sampled position. This script picks the scale S in

    label = tanh(material_in_pawns / S)

Criterion: S maximises the entropy of the label distribution over [-1, 1]. That is the
two-sided version of "most balances land where tanh still has a gradient" -- too small an
S piles mass onto +/-1 where the derivative vanishes, too large an S piles it onto 0 where
the label stops distinguishing positions. The maximum sits between the two, and at the
optimum the label deciles come out nearly evenly spaced, so every part of the output range
carries roughly the same number of examples.

Usage:  python3 fit_tanh.py [samples.bin]
"""
from __future__ import annotations

import json
import sys

import numpy as np

DTYPE = np.dtype([("ply", "<u2"), ("npieces", "<u2"), ("nmoves", "<u2"),
                  ("full", "<u2"), ("mat", "<i4")])
VALUE_FP = 1000
PLY_EDGES = [0, 25, 50, 100, 200, 400, 800, 1600, 1 << 20]


def label_entropy(m: np.ndarray, S: float, bins: int = 128) -> float:
    h, _ = np.histogram(np.tanh(m / S), bins=bins, range=(-1.0, 1.0))
    p = h[h > 0] / h.sum()
    return float(-(p * np.log(p)).sum() / np.log(bins))


def maximise(f, lo: float, hi: float, iters: int = 80) -> float:
    """Golden-section search on a unimodal objective."""
    r = (5 ** 0.5 - 1) / 2
    c, d = hi - r * (hi - lo), lo + r * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - r * (hi - lo); fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + r * (hi - lo); fd = f(d)
    return (lo + hi) / 2


def fit(path: str = "samples.bin") -> dict:
    d = np.fromfile(path, dtype=DTYPE)
    if not len(d):
        raise SystemExit(f"{path} is empty -- run `./tky sample --games N --threads T` first")
    m = d["mat"].astype(np.float64) / VALUE_FP
    ply = d["ply"].astype(np.int64)

    S = maximise(lambda s: label_entropy(m, s), 20.0, 2000.0)
    lab = np.tanh(m / S)
    a = np.abs(m)

    out = {
        "n_samples": int(len(d)),
        "criterion": "S maximises the entropy of tanh(material/S) over 128 bins on [-1,1]",
        "global_S_pawns": round(float(S), 2),
        "material_pawns": {
            "mean": round(float(m.mean()), 3), "sd": round(float(m.std()), 3),
            "abs_p50": round(float(np.percentile(a, 50)), 2),
            "abs_p90": round(float(np.percentile(a, 90)), 2),
            "abs_p99": round(float(np.percentile(a, 99)), 2),
            "min": round(float(m.min()), 2), "max": round(float(m.max()), 2),
        },
        "label": {
            "entropy_normalised": round(label_entropy(m, S), 4),
            "sd": round(float(lab.std()), 4),
            "frac_saturated_abs_gt_0.95": round(float(np.mean(np.abs(lab) > 0.95)), 4),
            "frac_in_high_gradient_abs_x_lt_1.5": round(float(np.mean(a / S < 1.5)), 4),
            "deciles": [round(float(v), 3) for v in np.percentile(lab, np.arange(10, 100, 10))],
        },
        "cross_check_S_from_p95_rule": round(float(np.percentile(a, 95) / np.arctanh(0.90)), 1),
        "per_ply_S_pawns": [],
    }
    for lo, hi in zip(PLY_EDGES, PLY_EDGES[1:]):
        sel = (ply >= lo) & (ply < hi)
        if sel.sum() < 2000:
            continue
        mb = m[sel]
        Sb = maximise(lambda s: label_entropy(mb, s, bins=64), 10.0, 2000.0)
        out["per_ply_S_pawns"].append({
            "ply_lo": lo, "ply_hi": None if hi > 1 << 19 else hi,
            "n": int(sel.sum()), "S_pawns": round(float(Sb), 1),
            "mean_pieces": round(float(d["npieces"][sel].mean()), 1),
            "sd_material": round(float(mb.std()), 1),
        })
    return out


if __name__ == "__main__":
    res = fit(sys.argv[1] if len(sys.argv) > 1 else "samples.bin")
    json.dump(res, open("tanh_fit.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    print(f"\n-> pass --tanh-scale {res['global_S_pawns']} to `tky gen`", file=sys.stderr)
