"""Summarises a generated corpus and writes manifest.json next to the shards.

Reads only the game headers (one linear pass, no replay), so it is fast even on a corpus
of tens of millions of games.
"""
from __future__ import annotations

import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from tky_reader import Corpus, PositionCorpus, RESULT_NAMES, TERM_NAMES

def build_positions(root: str) -> dict:
    import numpy as np
    pc = PositionCorpus(root)
    h = pc.header
    lab = pc.labels()
    hist, _ = np.histogram(lab, bins=h.sel_bins, range=(-1.0, 1.0))
    nz = hist[hist > 0]
    return {
        "kind": "positions",
        "shards": len(pc.parts),
        "positions": len(pc),
        "positions_sampled": pc.positions_sampled,
        "kept_fraction": round(len(pc) / max(1, pc.positions_sampled), 5),
        "bytes": sum(p.stat().st_size for p in sorted(Path(root).glob("*.tkp"))),
        "bytes_per_position": 1296 * 2 + 32,
        "label_balance": {
            "bins": int(h.sel_bins),
            "min_max_ratio": round(float(nz.max() / nz.min()), 3) if len(nz) else None,
            "sd_of_bin_share": round(float((hist / hist.sum() * len(hist)).std()), 4),
        },
        "ply": {
            "min": int(pc.records["ply"].min()), "max": int(pc.records["ply"].max()),
            "median": int(np.median(pc.records["ply"])),
        },
        "generation": {
            "format_version": h.version, "save_percentage": h.save_percentage,
            "policy": "uniform-move" if h.policy == 0 else "uniform-piece",
            "max_plies": h.max_plies, "forced_king_capture": bool(h.forced_king_capture),
            "seed_base": h.seed_base,
        },
        "labels": {
            "tanh_scale_pawns": h.tanh_scale_pawns,
            "material_unit_pawns": h.mat_unit_pawns,
            "formula": "label = tanh(mat * material_unit_pawns / tanh_scale_pawns)",
            "perspective": "black minus white; King excluded",
        },
    }


def build(root: str) -> dict:
    if not list(Path(root).glob("*.tky")) and list(Path(root).glob("*.tkp")):
        return build_positions(root)
    c = Corpus(root)
    h = c.header
    res, term, lens = Counter(), Counter(), []
    total_bytes = 0
    for s in c.shards:
        total_bytes += s.path.stat().st_size
        for off in s.index():
            n_plies, r, t = struct.unpack_from("<IBB", s.buf, off)
            res[RESULT_NAMES[r]] += 1
            term[TERM_NAMES[t]] += 1
            lens.append(n_plies)
    lens = np.asarray(lens)
    n = len(lens)
    return {
        "kind": "games",
        "shards": len(c.shards),
        "games": n,
        "positions": int(lens.sum() + n),          # every game also stores its final position
        "bytes": total_bytes,
        "bytes_per_position": round(total_bytes / max(1, int(lens.sum() + n)), 3),
        "game_length_plies": {
            "mean": round(float(lens.mean()), 1), "min": int(lens.min()), "max": int(lens.max()),
            "p10": int(np.percentile(lens, 10)), "p50": int(np.percentile(lens, 50)),
            "p90": int(np.percentile(lens, 90)),
        },
        "results": {k: round(v / n, 5) for k, v in sorted(res.items())},
        "terminations": {k: round(v / n, 5) for k, v in sorted(term.items())},
        "generation": {
            "format_version": h.version,
            "policy": "uniform-move" if h.policy == 0 else "uniform-piece",
            "max_plies": h.max_plies, "repetition_limit": h.rep_limit,
            "no_progress_limit": h.no_progress_limit, "stalemate_loses": bool(h.stalemate_loses),
            "forced_king_capture": bool(h.forced_king_capture),
            "save_percentage": h.save_percentage,
            "seed_base": h.seed_base,
        },
        "labels": {
            "tanh_scale_pawns": h.tanh_scale_pawns,
            "material_unit_pawns": h.mat_unit_pawns,
            "formula": "label = tanh(material_field * material_unit_pawns / tanh_scale_pawns)",
            "perspective": "black minus white; King excluded (both sides hold one until the game ends)",
        },
    }


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "out"
    m = build(root)
    p = Path(root) / "manifest.json"
    p.write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m, indent=2))
    print(f"\nwrote {p}", file=sys.stderr)
