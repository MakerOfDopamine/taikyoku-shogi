"""Statistics over a .tkn self-play corpus: calibration, diversity, lengths, sign accuracy,
material.

    python3 tkn_stats.py selfplay.tkn.tgz            # streams the archive, nothing unpacked
    python3 tkn_stats.py corpus/                     # a directory of shards, or one .tkn
    python3 tkn_stats.py selfplay.tkn --csv stats_   # also write stats_*.csv
    python3 tkn_stats.py selfplay.tkn.tgz --brief    # only the headline numbers

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


BRIEF = False


def say(text="", key=False):
    """Print a report line. With --brief, only the key lines, without the section indent."""
    if not BRIEF:
        print(text)
    elif key:
        print("\n".join(l[4:] if l.startswith("    ") else l for l in text.strip("\n").split("\n")))


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
            if m.isfile() and m.name.endswith(".tkn"):     # not a .kids beside it
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
    f32 = lambda o: struct.unpack_from("<f", raw, o)[0]
    if u32(12) > FILE_HDR:                         # version 7 on: 256 bytes, the search
        raw += f.read(u32(12) - FILE_HDR)
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
            "drop_capped": (u32(100) >> 1) & 1, "endgames": (u32(100) >> 2) & 1,
            "model": raw[104:128].split(b"\x00")[0].decode(),
            # version 7 on; sims 0 is the one-ply search
            "sims": u32(128) if u32(8) >= 7 else 0, "leaves": u32(132) if u32(8) >= 7 else 0,
            "cpuct": f32(136) if u32(8) >= 7 else 0.0,
            "noise_frac": f32(140) if u32(8) >= 7 else 0.0,
            "noise_alpha": f32(144) if u32(8) >= 7 else 0.0,
            "games_par": u32(148) if u32(8) >= 7 else 1, "opp_sims": u32(152) if u32(8) >= 7 else 0,
            "contempt": f32(160) if u32(8) >= 7 else 0.0,
            "contempt_flat": (u32(156) >> 3) & 1 if u32(8) >= 7 else 0,
            "soft_scale": f32(164) if u32(12) >= 256 else 0.0,
            "soft_pawns": f32(168) if u32(12) >= 256 else 0.0}


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
    g_caps, g_quiet, g_lastcap, g_swing, g_final = [], [], [], [], []
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
                g_caps.append(0); g_quiet.append(0); g_lastcap.append(-1)
                g_swing.append(np.nan); g_final.append(0.0)
                continue
            cap_at = np.nonzero(recs["flags"] & 1)[0]
            g_caps.append(len(cap_at))
            g_lastcap.append(int(cap_at[-1]) if len(cap_at) else -1)
            g_quiet.append(int(np.diff(np.concatenate([[-1], cap_at, [n_plies]])).max()) - 1)
            m_abs = recs["material"].astype(np.float64) / (1 << shift)   # black minus white
            g_final.append(float(m_abs[-1]))
            # the biggest single-ply jump in the winner's favour, from a level start
            g_swing.append(float(np.diff(np.concatenate([[0.0], result * m_abs])).max())
                           if result else np.nan)
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
        "game_captures": np.array(g_caps), "game_quiet": np.array(g_quiet),
        "game_last_capture": np.array(g_lastcap), "game_swing": np.array(g_swing),
        "game_final_material": np.array(g_final),
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
    say("\n(a) CALIBRATION -- does a value of v actually mean an outcome of v?\n")
    if not np.any(v != 0):
        say("    the value field is all zeros: this corpus carries no per-ply value.")
        return False
    if not np.any(z != 0):
        say("    every game is a draw: nothing to calibrate against.")
        return False
    edges = np.linspace(-1, 1, nbins + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, nbins - 1)
    say("    bin           n      mean v   mean outcome   win%  draw%  loss%   calibration")
    rows = []
    for b in range(nbins):
        m = idx == b
        n = int(m.sum())
        if not n:
            continue
        mv, mz = float(v[m].mean()), float(z[m].mean())
        w = float((z[m] == 1).mean()); dr = float((z[m] == 0).mean()); l = float((z[m] == -1).mean())
        # perfect calibration puts mean outcome on the identity line; show the gap
        say(f"    [{edges[b]:+.2f},{edges[b+1]:+.2f}) {n:8d}  {mv:+7.4f}   {mz:+8.4f}   "
              f"{100*w:5.1f} {100*dr:6.1f} {100*l:6.1f}   {bar((mz+1)/2)}")
        rows.append((f"{edges[b]:.3f}", f"{edges[b+1]:.3f}", n, mv, mz, w, dr, l))
    write_csv(prefix, "calibration", ["lo", "hi", "n", "mean_value", "mean_outcome",
                                      "win", "draw", "loss"], rows)
    zf = z.astype(np.float64)
    r = float(np.corrcoef(v, zf)[0, 1])
    mse = float(((v - zf) ** 2).mean())
    base = float((zf ** 2).mean())
    say(f"\n    correlation(value, outcome) = {r:+.4f}   (R2 vs predicting 0: "
          f"{1 - mse / base:+.4f})")
    say(f"    mean |value - outcome|      = {np.abs(v - zf).mean():.4f}")
    say(f"    Brier-style MSE             = {mse:.4f}   (predicting 0 everywhere gives {base:.4f})")

    # least squares outcome ~ a + b*v: b < 1 the values are too extreme (overconfident),
    # b > 1 too timid (underconfident); a is a constant shift.
    b = float(np.cov(v, zf)[0, 1] / v.var())
    a = float(zf.mean() - b * v.mean())
    mse_cal = float(((a + b * v - zf) ** 2).mean())
    say(f"    least squares: outcome ~ {a:+.4f} {b:+.4f} * value", key=True)
    if b < 0.95:
        verdict = (f"slope {b:.3f} < 1: overconfident -- outcomes move less than the values do, "
                   f"so the values are too extreme")
    elif b > 1.05:
        verdict = (f"slope {b:.3f} > 1: underconfident -- outcomes move more than the values do, "
                   f"so the values are too timid")
    else:
        verdict = f"slope {b:.3f}: about right in scale"
    say(f"      {verdict}.")
    say(f"      Mapping every value v to {a:+.3f} {b:+.3f} * v would calibrate it and cut MSE to "
          f"{mse_cal:.4f} ({1 - mse_cal / base:+.4f} R2).")
    if abs(a) > 0.05:
        say(f"      The offset {a:+.3f} is a constant shift. At a low sampling temperature it is\n"
              f"      expected: each recorded value is that of the child the mover picked as best,\n"
              f"      and the best of many noisy estimates is biased in the mover's favour.")
    say("    Perfect calibration is the identity: mean outcome should equal mean v in every\n"
          "    row. Mean outcome flatter than mean v is overconfidence, steeper is underconfidence.")

    # does the value head know anything material does not?
    m = d["material"].astype(np.float64)
    r_vm = float(np.corrcoef(v, m)[0, 1])
    r_mz = float(np.corrcoef(m, zf)[0, 1])
    denom = np.sqrt((1 - r_vm ** 2) * (1 - r_mz ** 2))
    partial = (r - r_vm * r_mz) / denom if denom > 0 else float("nan")
    say(f"\n    value vs material:  corr(value, material) = {r_vm:+.4f}", key=True)
    say(f"      corr(value, outcome) {r:+.4f} vs corr(material, outcome) {r_mz:+.4f}", key=True)
    say(f"      partial corr(value, outcome | material) = {partial:+.4f}", key=True)
    say("      That last number is what the value head adds beyond counting material.")
    return True


def diversity(d, prefix):
    say("\n(b) DIVERSITY -- distinct opening sequences and repeated positions\n")
    ngames = len(d["lengths"])
    rows = []
    for n in sorted(d["openings"]):
        seqs = d["openings"][n]
        if not seqs:
            continue
        c = Counter(seqs)
        top = c.most_common(1)[0][1]
        say(f"    first {n} moves: {len(c):6d} distinct of {len(seqs):6d} games "
              f"({len(c)/len(seqs):6.2%})   most repeated line: {top}x", key=True)
        rows.append((n, len(c), len(seqs), top))
    write_csv(prefix, "openings", ["n_moves", "distinct", "games", "largest_group"], rows)

    counts = d["counts"]
    if counts:
        total = sum(counts.values())
        distinct = len(counts)
        dup = total - distinct
        rep = Counter(counts.values())
        say(f"\n    positions: {total:8d} total, {distinct:8d} distinct, "
              f"{dup:8d} duplicated ({dup/total:.3%})")
        say(f"    most repeated position appears {max(counts.values())}x; "
              f"{rep[1]} positions occur exactly once ({rep[1]/distinct:.2%} of distinct)")
        say("\n    Duplicates are counted in the canonical frame, the one the network sees,")
        say("    so a position and its colour-mirrored rotation count as the same sample.")
        say("    HIGH duplication would mean the games keep re-treading the same lines: the")
        say("    corpus carries less information than its size suggests, the effective")
        say("    dataset is smaller than the ply count, and training over-weights whatever")
        say("    those repeated positions happen to teach. LOW duplication means broad")
        say("    exploration -- but note the flip side: with almost every position seen once,")
        say("    each one's value target is a single noisy game outcome that never gets")
        say("    averaged with another visit, so the labels stay high-variance.")


def lengths(d, prefix):
    L = d["lengths"]
    say("\n(c) GAME LENGTHS\n")
    edges = [0, 10, 25, 50, 100, 200, 400, 600, 800, 1000, 10 ** 9]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (L >= lo) & (L < hi)
        n = int(m.sum())
        label = f"{lo}-{hi-1}" if hi < 10 ** 9 else f"{lo}+"
        say(f"    {label:>10} plies  {n:6d}  {n/len(L):6.2%}  {bar(n/len(L))}")
        rows.append((label, n, n / len(L)))
    write_csv(prefix, "lengths", ["bucket", "games", "fraction"], rows)
    q = np.percentile(L, [50, 90, 99])
    say(f"\n    {len(L)} games: mean {L.mean():.1f}  median {q[0]:.0f}  "
          f"p90 {q[1]:.0f}  p99 {q[2]:.0f}  max {L.max()}", key=True)
    tc = Counter(int(t) for t in d["terms"])
    say("    terminations: " + ", ".join(
        f"{TERM_NAMES[t]} {c} ({c/len(L):.1%})" for t, c in tc.most_common()), key=True)
    rc = Counter(int(r) for r in d["results"])
    say(f"    results: black {rc.get(1,0)} white {rc.get(-1,0)} draw {rc.get(0,0)}", key=True)


def sign_accuracy(d, prefix):
    v, z, rem = d["value"], d["result"], d["remaining"]
    say("\n(d) SIGN ACCURACY vs distance from the end of the game\n")
    keep = (z != 0) & (v != 0)
    if not keep.any():
        say("    no decided positions with a non-zero value")
        return
    v, z, rem = v[keep], z[keep], rem[keep]
    correct = np.sign(v) == np.sign(z)
    edges = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 10 ** 9]
    rows = []
    say("    plies from end        n    accuracy   mean |v|   ")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rem >= lo) & (rem < hi)
        n = int(m.sum())
        if not n:
            continue
        acc = float(correct[m].mean())
        label = f"{lo}-{hi-1}" if hi < 10 ** 9 else f"{lo}+"
        say(f"    {label:>14} {n:9d}    {acc:6.2%}     {np.abs(v[m]).mean():.4f}   "
              f"{bar((acc-0.5)*2)}")
        rows.append((label, n, acc, float(np.abs(v[m]).mean())))
    write_csv(prefix, "sign_accuracy", ["bucket", "n", "accuracy", "mean_abs_value"], rows)
    say(f"\n    overall {correct.mean():.2%} over {correct.size} decided positions. "
        f"0.5 is chance;", key=True)
    say("    the bar shows how far above chance each bucket is.")


def material(d, prefix, nbuckets=10):
    m, z = d["material"], d["result"]
    say("\n(e) MATERIAL vs OUTCOME (material is the side to move minus the opponent)\n")
    qs = np.quantile(m, np.linspace(0, 1, nbuckets + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    say("    material (pawns)          n     win%   draw%   loss%   mean outcome")
    rows = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (m >= lo) & (m < hi)
        n = int(sel.sum())
        if not n:
            continue
        w = float((z[sel] == 1).mean()); dr = float((z[sel] == 0).mean())
        l = float((z[sel] == -1).mean()); mo = float(z[sel].mean())
        lab = f"{'-inf' if lo == -np.inf else f'{lo:+.0f}'} .. {'+inf' if hi == np.inf else f'{hi:+.0f}'}"
        say(f"    {lab:>20} {n:8d}   {100*w:5.1f}  {100*dr:6.1f}  {100*l:6.1f}   "
              f"{mo:+6.3f}  {bar((mo+1)/2)}")
        rows.append((lab, n, w, dr, l, mo))
    write_csv(prefix, "material", ["bucket", "n", "win", "draw", "loss", "mean_outcome"], rows)
    dec = z[m > 0]
    say(f"\n    with a material surplus ({(m > 0).sum()} positions): "
          f"{(dec == 1).mean():.2%} win, {(dec == 0).mean():.2%} draw, {(dec == -1).mean():.2%} loss", key=True)
    dec = z[m < 0]
    say(f"    with a material deficit ({(m < 0).sum()} positions): "
          f"{(dec == 1).mean():.2%} win, {(dec == 0).mean():.2%} draw, {(dec == -1).mean():.2%} loss", key=True)
    ok = np.isfinite(m) & np.isfinite(z)
    say(f"    correlation(material, outcome) = {np.corrcoef(m[ok], z[ok])[0,1]:+.4f}", key=True)


def endings(d, prefix):
    """(f) How games end, in the numbers worth comparing from one batch to the next."""
    L, t, res = d["lengths"], d["terms"], d["results"]
    h = d["hdr"]
    names = np.array([TERM_NAMES[k] for k in t])
    say("\n(f) HOW GAMES END -- compare these from one batch to the next\n")
    say(f"    {'ended by':>14} {'games':>15} {'positions':>18} {'median length':>14}", key=True)
    rows = []
    for name in TERM_NAMES:
        m = names == name
        if not m.any():
            continue
        g, p = int(m.sum()), int(L[m].sum())
        say(f"    {name:>14} {g:7d} ({g / len(L):5.1%}) {p:9d} ({p / L.sum():5.1%}) "
              f"{np.median(L[m]):14.0f}", key=True)
        rows.append((name, g, g / len(L), p, p / L.sum(), float(np.median(L[m]))))
    say("    Training sees positions, not games: a game type that is rare but long can still\n"
          "    supply most of the data, and every position in a game shares its one label.")

    out = {}
    rz = names == "resignation"
    if rz.any():
        q = np.percentile(L[rz], [10, 50, 90])
        lead = np.abs(d["game_final_material"][rz])
        sw = d["game_swing"][rz]
        share = sw / np.maximum(lead, 1e-9)
        say(f"\n    resignations ({int(rz.sum())}): decided at ply {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} "
              f"(p10 / median / p90); final lead median {np.median(lead):.0f} pawns", key=True)
        say(f"      biggest single-ply swing to the winner: median {np.nanmedian(sw):.0f} pawns, "
              f"{np.nanmedian(share):.0%} of the lead it resigned at", key=True)
        say("      Near 100%: one big event decided the game. Well under: the lead was built\n"
              "      over many moves. A rising resignation ply with a falling swing share means\n"
              "      both sides are giving material away less often.")
        out.update(resign_ply_median=float(q[1]), resign_ply_p10=float(q[0]),
                   resign_ply_p90=float(q[2]), resign_lead_median=float(np.median(lead)),
                   resign_swing_median=float(np.nanmedian(sw)),
                   resign_swing_share=float(np.nanmedian(share)))
    cz = names == "ply_cap"
    if cz.any():
        n = L[cz].astype(np.float64)
        rate = d["game_captures"][cz] / n * 100
        draws = int((res[cz] == 0).sum())
        say(f"\n    ply cap ({int(cz.sum())} games, {cz.mean():.1%}; {L[cz].sum() / L.sum():.1%} of positions): "
              f"{draws} drawn, {int(cz.sum()) - draws} scored on material", key=True)
        say(f"      captures per 100 plies: median {np.median(rate):.1f}; longest capture-free "
              f"stretch: median {np.median(d['game_quiet'][cz]):.0f} plies; last capture at ply "
              f"median {np.median(d['game_last_capture'][cz]):.0f} of {h['max_plies']}", key=True)
        say("      Captures continuing to the end: the game was still being played and simply\n"
              "      stayed level. Long capture-free stretches, or a last capture far from the\n"
              "      cap: stagnation, where length says nothing good about strength.")
        out.update(cap_games=float(cz.mean()), cap_positions=float(L[cz].sum() / L.sum()),
                   cap_draws=draws, cap_captures_per_100=float(np.median(rate)),
                   cap_quiet_median=float(np.median(d["game_quiet"][cz])),
                   cap_last_capture_median=float(np.median(d["game_last_capture"][cz])))
    for name in ("repetition", "stalemate", "no_progress"):
        k = int((names == name).sum())
        if k:
            say(f"\n    {name}: {k} games", key=True)
        out[name] = k
    write_csv(prefix, "endings", ["ended_by", "games", "game_share", "positions", "position_share",
                                  "median_length"], rows)
    write_csv(prefix, "endings_summary", list(out), [list(out.values())])


def adjudication(path, limit=None, leads=(100, 200, 300, 400, 600, 800), holds=(1, 10, 30)):
    """(g) For each resignation rule -- a lead of M pawns held K plies -- how many games it
    would end, how early, how often the leader went on to win, and how often the lead was
    later reversed (the other side went M ahead). Reversal is what a resignation threshold
    must avoid; "leader won" is low whenever games are decided by King blunders rather than
    by material. Needs games played to their natural end: a corpus written with
    resignation on stops at the rule in force, so nothing past it can be measured."""
    print("\n(g) ADJUDICATION -- what a material resignation rule would have done\n")
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
                    help="also (g): how material resignation thresholds would have fared")
    ap.add_argument("--brief", action="store_true",
                    help="only the headline numbers, for comparing one batch with the next")
    a = ap.parse_args()
    global BRIEF
    BRIEF = a.brief

    d = collect(a.path, a.limit_games, not (a.no_hash or a.brief))
    h = d["hdr"]
    say(f"{a.path}: {len(d['lengths'])} games, {d['value'].size} plies", key=True)
    say(f"  model {h['model']}  temperature {h['temperature']:g}  max_plies {h['max_plies']}  "
          f"regicide {'on' if h['regicide'] else 'off'}  "
          f"king safety {'on' if h['king_safety'] else 'off'}", key=True)
    if h["resign_lead"] or h["cap_by_material"]:
        say(f"  adjudication: {h['resign_lead']}-pawn lead held {h['resign_plies']} plies"
              f"{', ply cap scored on material, a draw under ' + str(h['cap_draw_margin']) + ' pawns' if h['cap_by_material'] else ''}", key=True)
    if h["sims"]:
        say(f"  search: {h['sims']} simulations a move, {h['leaves']} leaves per call, c_puct "
            f"{h['cpuct']:g}, root noise {h['noise_frac']:g} at alpha {h['noise_alpha']:g}/moves, "
            f"{h['games_par']} games in lockstep, contempt {h['contempt']:g}"
            f"{' (flat)' if h['contempt_flat'] else ''}", key=True)
    elif h.get("policy_top") is not None:
        say(f"  children scored: the policy's top {h['policy_top']:.1%} of the legal moves "
              f"plus {h['policy_explore']:.1%} at random", key=True)
    if h["soft_scale"]:
        say(f"  soft labels: a game cut off by the ply cap or by repetition is labelled "
            f"{h['soft_scale']:g} * tanh(material / {h['soft_pawns']:g} pawns)", key=True)
    if h["endgames"]:
        say("  endgames: random sparse starts with a material lead, only royal captures kept",
            key=True)
    if h["drop_capped"]:
        say("  games that reached the ply cap were dropped: none are in this corpus", key=True)
    frac = d["n_scored"] / np.maximum(d["n_legal"], 1)
    say(f"  network evaluations per position: {d['n_scored'].mean():.1f}, against "
          f"{d['n_legal'].mean():.1f} legal moves ({frac.mean():.1%})", key=True)

    if calibration(d, a.bins, a.csv):
        sign_accuracy(d, a.csv)
    diversity(d, a.csv)
    lengths(d, a.csv)
    material(d, a.csv, a.material_buckets)
    endings(d, a.csv)
    if a.adjudication:
        adjudication(a.path, a.limit_games)
    if a.csv:
        print(f"\nCSV written with prefix {a.csv!r}")


if __name__ == "__main__":
    main()
