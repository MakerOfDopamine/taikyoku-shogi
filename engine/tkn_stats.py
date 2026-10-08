"""Statistics over a .tkn self-play corpus: calibration, diversity, lengths, sign accuracy,
material.

    python3 tkn_stats.py selfplay.tkn.tgz            # streams the archive, nothing unpacked
    python3 tkn_stats.py corpus/                     # a directory of shards, or one .tkn
    python3 tkn_stats.py selfplay.tkn --csv stats_   # also write stats_*.csv

Everything is read in the frame of the side to move in each recorded position, which is the
frame the network was given. ``value`` and ``result_stm`` therefore agree in sign when the
network was right, and perfect calibration is the identity line: a value of v should be the
mean outcome of the positions that scored v.

One pass, vectorised per game, so a million plies costs a couple of minutes and a few
hundred MB rather than a million Python objects.
"""
from __future__ import annotations

import argparse
import glob
import os
import struct
import sys
from collections import Counter

import numpy as np

NSQ = 1296
REC = NSQ * 2 + 16
FILE_HDR = 128
GAME_HDR = 16
MAGIC = b"TKYNNSP\x00"
TERM_NAMES = ("royal_capture", "stalemate", "repetition", "no_progress", "ply_cap",
              "resignation")

DT = np.dtype([("board", "<u2", (NSQ,)), ("move", "<u4"), ("value", "<f4"),
               ("n_legal", "<u2"), ("material", "<i2"), ("side", "u1"),
               ("flags", "u1"), ("n_scored", "<u2")])
assert DT.itemsize == REC, DT.itemsize


def bar(frac, width=26, ch="#"):
    frac = 0.0 if frac != frac else max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return ch * n + "." * (width - n)


def open_streams(path):
    """Yield (name, binary stream) for a .tkn, a directory of them, or a .tgz/.tar.gz."""
    if path.endswith((".tgz", ".tar.gz", ".tar")):
        import tarfile
        tf = tarfile.open(path)
        for m in tf.getmembers():
            if m.isfile():
                yield m.name, tf.extractfile(m)
        return
    paths = sorted(glob.glob(os.path.join(path, "*.tkn"))) if os.path.isdir(path) else [path]
    if not paths:
        raise SystemExit(f"no .tkn files under {path}")
    for p in paths:
        yield p, open(p, "rb")


def read_header(f, name):
    raw = f.read(FILE_HDR)
    if len(raw) != FILE_HDR or raw[:8] != MAGIC:
        raise SystemExit(f"{name}: not a .tkn shard (magic {raw[:8]!r})")
    u32 = lambda o: struct.unpack_from("<I", raw, o)[0]
    u64 = lambda o: struct.unpack_from("<Q", raw, o)[0]
    if u32(20) != REC:
        raise SystemExit(f"{name}: ply record is {u32(20)} bytes, expected {REC}")
    return {"games": u64(40), "plies": u64(48), "mat_shift": u32(28),
            "temperature": struct.unpack_from("<d", raw, 56)[0],
            "max_plies": u32(64), "regicide": u32(80) & 1,
            "king_safety": (u32(80) >> 1) & 1 if u32(8) >= 3 else 0,
            "cap_by_material": (u32(80) >> 2) & 1 if u32(8) >= 4 else 0,
            # version 4 scored any lead, version 5 on draws under 50 pawns
            "cap_draw_margin": (50 if u32(8) >= 5 else 0) if (u32(80) >> 2) & 1 and u32(8) >= 4 else None,
            "resign_lead": (u32(80) >> 8) & 0xFFFF if u32(8) >= 4 else 0,
            "resign_plies": u32(80) >> 24 if u32(8) >= 4 else 0,
            "policy_top": (u32(100) >> 16) / 1000 if u32(8) >= 6 else None,
            "policy_explore": ((u32(100) >> 8) & 0xFF) / 1000 if u32(8) >= 6 else None,
            "model": raw[104:128].split(b"\x00")[0].decode()}


def iter_games(f, limit=None):
    n = 0
    while True:
        gh = f.read(GAME_HDR)
        if len(gh) < GAME_HDR:
            return
        n_plies, result, term = struct.unpack_from("<IbB", gh, 0)
        raw = f.read(n_plies * REC)
        if len(raw) < n_plies * REC:
            return
        yield n_plies, result, term, np.frombuffer(raw, dtype=DT)
        n += 1
        if limit is not None and n >= limit:
            return


def canonical_boards(boards, stm):
    """Rotate white-to-move rows 180 degrees and flip their colour bits, in place on a copy.

    Duplicates are counted in this frame because it is the one the network sees: a position
    and its colour-mirrored rotation are the same input, so they are the same sample.
    """
    out = boards.copy()
    flip = stm == 0
    if flip.any():
        r = boards[flip][:, ::-1]
        occupied = (r >> 2) != 0
        out[flip] = np.where(occupied, r ^ 2, r)
    return out


def collect(path, limit=None, hash_boards=True):
    val, res, mat, rem, nleg, nsc, dec = [], [], [], [], [], [], []
    lengths, terms, results = [], [], []
    openings = {n: [] for n in (2, 4, 8)}
    counts = Counter()
    hdr = None
    for name, f in open_streams(path):
        h = read_header(f, name)
        hdr = hdr or h
        shift = h["mat_shift"]
        for n_plies, result, term, recs in iter_games(f, limit):
            lengths.append(n_plies)
            terms.append(term)
            results.append(result)
            if n_plies == 0:
                continue
            side = recs["side"].astype(np.int8)
            stm = 1 - side                                  # side to move in this position
            sign = np.where(stm == 1, 1, -1).astype(np.int8)
            val.append(recs["value"].astype(np.float32))
            res.append((result * sign).astype(np.int8))      # outcome for the side to move
            mat.append(recs["material"].astype(np.float32) / (1 << shift) * sign)
            rem.append((n_plies - 1 - np.arange(n_plies)).astype(np.int32))
            nleg.append(recs["n_legal"].astype(np.int32))
            nsc.append(recs["n_scored"].astype(np.int32))
            dec.append(np.full(n_plies, result != 0, dtype=bool))
            mv = recs["move"]
            for n in openings:
                if n_plies >= n:
                    openings[n].append(mv[:n].tobytes())
            if hash_boards:
                for row in canonical_boards(recs["board"], stm):
                    counts[hash(row.tobytes())] += 1
        if hasattr(f, "close"):
            f.close()
    if not val:
        raise SystemExit("no plies found")
    return {
        "hdr": hdr,
        "value": np.concatenate(val), "result": np.concatenate(res),
        "material": np.concatenate(mat), "remaining": np.concatenate(rem),
        "n_legal": np.concatenate(nleg), "n_scored": np.concatenate(nsc),
        "decided": np.concatenate(dec),
        "lengths": np.array(lengths), "terms": np.array(terms),
        "results": np.array(results), "openings": openings, "counts": counts,
    }


def write_csv(prefix, name, header, rows):
    if not prefix:
        return
    with open(f"{prefix}{name}.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def calibration(d, nbins, prefix):
    v, z = d["value"], d["result"]
    print("\n(a) CALIBRATION -- does a value of v actually mean an outcome of v?\n")
    if not np.any(v != 0):
        print("    the value field is all zeros: this corpus carries no per-ply value.")
        return False
    edges = np.linspace(-1, 1, nbins + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, nbins - 1)
    print("    bin           n      mean v   mean outcome   win%  draw%  loss%   calibration")
    rows = []
    for b in range(nbins):
        m = idx == b
        n = int(m.sum())
        if not n:
            continue
        mv, mz = float(v[m].mean()), float(z[m].mean())
        w = float((z[m] == 1).mean()); dr = float((z[m] == 0).mean()); l = float((z[m] == -1).mean())
        # perfect calibration puts mean outcome on the identity line; show the gap
        print(f"    [{edges[b]:+.2f},{edges[b+1]:+.2f}) {n:8d}  {mv:+7.4f}   {mz:+8.4f}   "
              f"{100*w:5.1f} {100*dr:6.1f} {100*l:6.1f}   {bar((mz+1)/2)}")
        rows.append((f"{edges[b]:.3f}", f"{edges[b+1]:.3f}", n, mv, mz, w, dr, l))
    write_csv(prefix, "calibration", ["lo", "hi", "n", "mean_value", "mean_outcome",
                                      "win", "draw", "loss"], rows)
    zf = z.astype(np.float64)
    r = float(np.corrcoef(v, zf)[0, 1])
    mse = float(((v - zf) ** 2).mean())
    base = float((zf ** 2).mean())
    print(f"\n    correlation(value, outcome) = {r:+.4f}   (R2 vs predicting 0: "
          f"{1 - mse / base:+.4f})")
    print(f"    mean |value - outcome|      = {np.abs(v - zf).mean():.4f}")
    print(f"    Brier-style MSE             = {mse:.4f}   (predicting 0 everywhere gives {base:.4f})")

    # least squares outcome ~ a + b*v. b < 1 is overconfidence, and 1/b is how much the
    # value would have to be shrunk to be calibrated.
    b = float(np.cov(v, zf)[0, 1] / v.var())
    a = float(zf.mean() - b * v.mean())
    mse_cal = float(((a + b * v - zf) ** 2).mean())
    print(f"    least squares: outcome ~ {a:+.4f} {b:+.4f} * value")
    print(f"      slope {b:.3f} < 1 means overconfident: scaling every value by {b:.3f} would")
    print(f"      calibrate it and cut MSE to {mse_cal:.4f} ({1 - mse_cal / base:+.4f} R2).")
    print("    Perfect calibration is the identity: mean outcome should equal mean v in every")
    print("    row. Mean outcome flatter than mean v means the net is overconfident.")

    # does the value head know anything material does not?
    m = d["material"].astype(np.float64)
    r_vm = float(np.corrcoef(v, m)[0, 1])
    r_mz = float(np.corrcoef(m, zf)[0, 1])
    denom = np.sqrt((1 - r_vm ** 2) * (1 - r_mz ** 2))
    partial = (r - r_vm * r_mz) / denom if denom > 0 else float("nan")
    print(f"\n    value vs material:  corr(value, material) = {r_vm:+.4f}")
    print(f"      corr(value, outcome) {r:+.4f} vs corr(material, outcome) {r_mz:+.4f}")
    print(f"      partial corr(value, outcome | material) = {partial:+.4f}")
    print("      That last number is what the value head adds beyond counting material.")
    return True


def diversity(d, prefix):
    print("\n(b) DIVERSITY -- distinct opening sequences and repeated positions\n")
    ngames = len(d["lengths"])
    rows = []
    for n in sorted(d["openings"]):
        seqs = d["openings"][n]
        if not seqs:
            continue
        c = Counter(seqs)
        top = c.most_common(1)[0][1]
        print(f"    first {n} moves: {len(c):6d} distinct of {len(seqs):6d} games "
              f"({len(c)/len(seqs):6.2%})   most repeated line: {top}x")
        rows.append((n, len(c), len(seqs), top))
    write_csv(prefix, "openings", ["n_moves", "distinct", "games", "largest_group"], rows)

    counts = d["counts"]
    if counts:
        total = sum(counts.values())
        distinct = len(counts)
        dup = total - distinct
        rep = Counter(counts.values())
        print(f"\n    positions: {total:8d} total, {distinct:8d} distinct, "
              f"{dup:8d} duplicated ({dup/total:.3%})")
        print(f"    most repeated position appears {max(counts.values())}x; "
              f"{rep[1]} positions occur exactly once ({rep[1]/distinct:.2%} of distinct)")
        print("\n    Duplicates are counted in the canonical frame, the one the network sees,")
        print("    so a position and its colour-mirrored rotation count as the same sample.")
        print("    HIGH duplication would mean the games keep re-treading the same lines: the")
        print("    corpus carries less information than its size suggests, the effective")
        print("    dataset is smaller than the ply count, and training over-weights whatever")
        print("    those repeated positions happen to teach. LOW duplication means broad")
        print("    exploration -- but note the flip side: with almost every position seen once,")
        print("    each one's value target is a single noisy game outcome that never gets")
        print("    averaged with another visit, so the labels stay high-variance.")


def lengths(d, prefix):
    L = d["lengths"]
    print("\n(c) GAME LENGTHS\n")
    edges = [0, 10, 25, 50, 100, 200, 400, 600, 800, 1000, 10 ** 9]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (L >= lo) & (L < hi)
        n = int(m.sum())
        label = f"{lo}-{hi-1}" if hi < 10 ** 9 else f"{lo}+"
        print(f"    {label:>10} plies  {n:6d}  {n/len(L):6.2%}  {bar(n/len(L))}")
        rows.append((label, n, n / len(L)))
    write_csv(prefix, "lengths", ["bucket", "games", "fraction"], rows)
    q = np.percentile(L, [50, 90, 99])
    print(f"\n    {len(L)} games: mean {L.mean():.1f}  median {q[0]:.0f}  "
          f"p90 {q[1]:.0f}  p99 {q[2]:.0f}  max {L.max()}")
    tc = Counter(int(t) for t in d["terms"])
    print("    terminations: " + ", ".join(
        f"{TERM_NAMES[t]} {c} ({c/len(L):.1%})" for t, c in tc.most_common()))
    rc = Counter(int(r) for r in d["results"])
    print(f"    results: black {rc.get(1,0)} white {rc.get(-1,0)} draw {rc.get(0,0)}")


def sign_accuracy(d, prefix):
    v, z, rem = d["value"], d["result"], d["remaining"]
    print("\n(d) SIGN ACCURACY vs distance from the end of the game\n")
    keep = (z != 0) & (v != 0)
    if not keep.any():
        print("    no decided positions with a non-zero value")
        return
    v, z, rem = v[keep], z[keep], rem[keep]
    correct = np.sign(v) == np.sign(z)
    edges = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 10 ** 9]
    rows = []
    print("    plies from end        n    accuracy   mean |v|   ")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rem >= lo) & (rem < hi)
        n = int(m.sum())
        if not n:
            continue
        acc = float(correct[m].mean())
        label = f"{lo}-{hi-1}" if hi < 10 ** 9 else f"{lo}+"
        print(f"    {label:>14} {n:9d}    {acc:6.2%}     {np.abs(v[m]).mean():.4f}   "
              f"{bar((acc-0.5)*2)}")
        rows.append((label, n, acc, float(np.abs(v[m]).mean())))
    write_csv(prefix, "sign_accuracy", ["bucket", "n", "accuracy", "mean_abs_value"], rows)
    print(f"\n    overall {correct.mean():.2%} over {correct.size} decided positions. "
          f"0.5 is chance;\n    the bar shows how far above chance each bucket is.")


def material(d, prefix, nbuckets=10):
    m, z = d["material"], d["result"]
    print("\n(e) MATERIAL vs OUTCOME (material is the side to move minus the opponent)\n")
    qs = np.quantile(m, np.linspace(0, 1, nbuckets + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    print("    material (pawns)          n     win%   draw%   loss%   mean outcome")
    rows = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (m >= lo) & (m < hi)
        n = int(sel.sum())
        if not n:
            continue
        w = float((z[sel] == 1).mean()); dr = float((z[sel] == 0).mean())
        l = float((z[sel] == -1).mean()); mo = float(z[sel].mean())
        lab = f"{'-inf' if lo == -np.inf else f'{lo:+.0f}'} .. {'+inf' if hi == np.inf else f'{hi:+.0f}'}"
        print(f"    {lab:>20} {n:8d}   {100*w:5.1f}  {100*dr:6.1f}  {100*l:6.1f}   "
              f"{mo:+6.3f}  {bar((mo+1)/2)}")
        rows.append((lab, n, w, dr, l, mo))
    write_csv(prefix, "material", ["bucket", "n", "win", "draw", "loss", "mean_outcome"], rows)
    dec = z[m > 0]
    print(f"\n    with a material surplus ({(m > 0).sum()} positions): "
          f"{(dec == 1).mean():.2%} win, {(dec == 0).mean():.2%} draw, {(dec == -1).mean():.2%} loss")
    dec = z[m < 0]
    print(f"    with a material deficit ({(m < 0).sum()} positions): "
          f"{(dec == 1).mean():.2%} win, {(dec == 0).mean():.2%} draw, {(dec == -1).mean():.2%} loss")
    ok = np.isfinite(m) & np.isfinite(z)
    print(f"    correlation(material, outcome) = {np.corrcoef(m[ok], z[ok])[0,1]:+.4f}")


def adjudication(path, limit=None, leads=(100, 200, 300, 400, 600, 800), holds=(1, 10, 30)):
    """(f) For each resignation rule -- a lead of M pawns held K plies -- how many games it
    would end, how early, how often the leader went on to win, and how often the lead was
    later reversed (the other side went M ahead). Reversal is what a resignation threshold
    must avoid; "leader won" is low whenever games are decided by King blunders rather than
    by material. Needs games played to their natural end: a corpus written with
    resignation on stops at the rule in force, so nothing past it can be measured."""
    print("\n(f) ADJUDICATION -- what a material resignation rule would have done\n")
    G = []
    for name, f in open_streams(path):
        h = read_header(f, name)
        for n_plies, result, term, recs in iter_games(f, limit):
            if n_plies:
                G.append((result, recs["material"].astype(np.float64) / (1 << h["mat_shift"])))
        if hasattr(f, "close"):
            f.close()
    if h.get("resign_lead"):
        print(f"    this corpus already resigned at {h['resign_lead']} pawns held "
              f"{h['resign_plies']} plies: games stop there, so rows at or above that lead "
              f"are censored. Calibrate on a run with --resign-lead 0.\n")
    d = np.concatenate([np.abs(np.diff(np.concatenate([[0.0], m]))) for _, m in G])
    print("    one ply moves material by (pawns): "
          + "  ".join(f"p{q} {np.percentile(d, q):.0f}" for q in (50, 90, 99, 99.9))
          + f"  max {d.max():.0f}")
    print(f"\n    {'lead':>6} {'held':>5} | {'games ended':>11} {'median ply':>10} | "
          f"{'leader won':>10} {'lead reversed':>13}")
    for M in leads:
        for K in holds:
            ended = agree = decided = rev = 0
            plies = []
            for r, m in G:
                s_ = np.where(m >= M, 1, np.where(m <= -M, -1, 0))
                run, at = 0, -1
                for i, x in enumerate(s_):
                    run = 0 if x == 0 else (run + x if run * x > 0 else x)
                    if abs(run) >= K:
                        at = i
                        break
                if at < 0:
                    continue
                ended += 1
                plies.append(at + 1)
                lead = np.sign(m[at])
                if r != 0:
                    decided += 1
                    agree += lead == r
                rev += bool((lead * m[at:] <= -M).any())
            if ended:
                print(f"    {M:>6} {K:>5} | {ended / len(G):>11.1%} {np.median(plies):>10.0f} | "
                      f"{agree / max(decided, 1):>10.1%} {rev / ended:>13.1%}")
    print("\n    Pick the smallest lead whose reversal rate you can live with: a reversed lead is"
          "\n    a game scored for the side that went on to be behind.")


def main():
    ap = argparse.ArgumentParser(description="statistics over a .tkn self-play corpus")
    ap.add_argument("path", help="a .tkn file, a directory of them, or a .tgz archive")
    ap.add_argument("--bins", type=int, default=20, help="calibration bins over [-1,1]")
    ap.add_argument("--material-buckets", type=int, default=10)
    ap.add_argument("--limit-games", type=int, default=None)
    ap.add_argument("--no-hash", action="store_true", help="skip duplicate detection (faster)")
    ap.add_argument("--csv", default="", help="prefix for CSV output")
    ap.add_argument("--adjudication", action="store_true",
                    help="also (f): how material resignation thresholds would have fared")
    a = ap.parse_args()

    d = collect(a.path, a.limit_games, not a.no_hash)
    h = d["hdr"]
    print(f"{a.path}: {len(d['lengths'])} games, {d['value'].size} plies")
    print(f"  model {h['model']}  temperature {h['temperature']:g}  max_plies {h['max_plies']}  "
          f"regicide {'on' if h['regicide'] else 'off'}  "
          f"king safety {'on' if h['king_safety'] else 'off'}")
    if h["resign_lead"] or h["cap_by_material"]:
        print(f"  adjudication: {h['resign_lead']}-pawn lead held {h['resign_plies']} plies"
              f"{', ply cap scored on material, a draw under ' + str(h['cap_draw_margin']) + ' pawns' if h['cap_by_material'] else ''}")
    if h.get("policy_top") is not None:
        print(f"  children scored: the policy's top {h['policy_top']:.1%} of the legal moves "
              f"plus {h['policy_explore']:.1%} at random")
    frac = d["n_scored"] / np.maximum(d["n_legal"], 1)
    print(f"  children scored per position: {d['n_scored'].mean():.1f} of "
          f"{d['n_legal'].mean():.1f} legal ({frac.mean():.1%})")

    if calibration(d, a.bins, a.csv):
        sign_accuracy(d, a.csv)
    diversity(d, a.csv)
    lengths(d, a.csv)
    material(d, a.csv, a.material_buckets)
    if a.adjudication:
        adjudication(a.path, a.limit_games)
    if a.csv:
        print(f"\nCSV written with prefix {a.csv!r}")


if __name__ == "__main__":
    main()
