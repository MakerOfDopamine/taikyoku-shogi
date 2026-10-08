"""Compare temperature schedules, in one of two modes.

    python3 sweep_temperature.py --schedules 1.0:0:1.0,1.0:0:0.5,1.5:20:0.3 --games 400
    python3 sweep_temperature.py --mode variance --schedules ... --games 300 --dry-run
    python3 sweep_temperature.py --mode variance --schedules ... --games 300 --jobs 2

A schedule is written ``OPENING_TEMP:OPENING_PLIES:MAIN_TEMP``: play the first
OPENING_PLIES hot, then switch. ``1.0:0:1.0`` is a flat temperature of 1.0.
``--temperatures`` is shorthand for a list of flat schedules.

--mode play (default) generates a corpus per schedule and reports game length, ply-cap
share, duplicates, and how well the *current net* predicts each corpus. That last part is
a proxy: a net trained on one schedule's games predicts that schedule best whether or not
its labels are any better.

--mode variance answers the question the proxy stands in for. It runs `nnplay variance`
per schedule: positions are sampled along each game and every one is replayed --branches
times, so the outcome variance splits into

    Var(z) = Var(E[z|s])  +  E[Var(z|s)]
             signal          noise: decided by what happens after s

and ACHIEVABLE R2 = signal / (signal + noise) is the most any value head could learn from
that schedule's labels, whatever net happens to be generating them. Higher is better. Every
number comes with a 95% interval from a bootstrap over games, because positions within a
game share their future and are not independent.

Run variance mode after play mode with the same --out-dir and schedules: it reads each
schedule's play shard to set the branch stride (--branch-every auto) and to estimate cost.
Snapshot cost grows with the square of game length, so check --dry-run first.

Both modes run the same number of *games* per schedule, because positions inside one game
share a single outcome label and the game count is the real sample size.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import struct
import subprocess
import sys
import time

import numpy as np

from tkn_stats import GAME_HDR, REC, collect, read_header


def tag_of(t0, n0, T):
    return f"T{T:g}" if n0 == 0 else f"T{t0:g}x{n0}_{T:g}"


def label_of(t0, n0, T):
    return f"flat {T:g}" if n0 == 0 else f"{t0:g} x{n0} -> {T:g}"


def schedule_args(t0, n0, T):
    return ["--temperature", str(T), "--opening-plies", str(n0),
            "--opening-temperature", str(t0)]


# ------------------------------------------------------------------------------ play mode

def metrics(path, seconds):
    d = collect(path)
    v = d["value"].astype(np.float64)
    z = d["result"].astype(np.float64)
    m = d["material"].astype(np.float64)
    L = d["lengths"]
    capped = int((d["terms"] == 4).sum())
    capped_plies = int(L[d["terms"] == 4].sum())

    def fit(vv, zz):
        if vv.size < 10 or vv.std() == 0 or zz.std() == 0:
            return float("nan"), float("nan")
        mse, base = float(((vv - zz) ** 2).mean()), float((zz ** 2).mean())
        return float(np.corrcoef(vv, zz)[0, 1]), (1 - mse / base if base else float("nan"))

    # the ply cap writes result 0 on games that were not drawn; those labels are simply
    # wrong, so report the correlations again with those games dropped
    ok = d["decided"]
    r_dec, r2_dec = fit(v[ok], z[ok])
    r_vz = float(np.corrcoef(v, z)[0, 1])
    r_vm = float(np.corrcoef(v, m)[0, 1])
    r_mz = float(np.corrcoef(m, z)[0, 1])
    den = np.sqrt((1 - r_vm ** 2) * (1 - r_mz ** 2))
    partial = (r_vz - r_vm * r_mz) / den if den > 0 else float("nan")
    mse, base = float(((v - z) ** 2).mean()), float((z ** 2).mean())

    counts = d["counts"]
    total = sum(counts.values())
    dup = (total - len(counts)) / total if total else 0.0
    seqs = d["openings"][4]
    distinct4 = len(set(seqs)) / len(seqs) if seqs else float("nan")

    return {
        "games": len(L), "plies": int(L.sum()),
        "mean_len": float(L.mean()), "median_len": float(np.median(L)),
        "cap_games": capped / len(L), "cap_plies": capped_plies / max(1, int(L.sum())),
        "decisive": float((d["results"] != 0).mean()),
        "dup": dup, "distinct4": distinct4,
        "r_vz": r_vz, "r2": 1 - mse / base, "partial": partial, "r_mz": r_mz,
        "r_dec": r_dec, "r2_dec": r2_dec, "n_dec": int(ok.sum()),
        "pos_per_s": int(L.sum()) / seconds if seconds > 0 else float("nan"),
        "seconds": seconds,
    }


def run_play(a, scheds):
    rows = []
    for t0, n0, T in scheds:
        tag = tag_of(t0, n0, T)
        shard = os.path.join(a.out_dir, f"{tag}.tkn")
        if not (a.reuse and os.path.exists(shard)):
            cmd = [a.nnplay, "play", "--games", str(a.games), *schedule_args(t0, n0, T),
                   "--max-plies", str(a.max_plies),
                   "--policy-top", str(a.policy_top),
                   "--quiet", "--out", shard] + a.extra.split()
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: {' '.join(cmd)}", flush=True)
            start = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.time() - start
            if r.returncode:
                print(r.stderr[-2000:])
                sys.exit(f"nnplay failed at {tag}")
        else:
            dt = float("nan")
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: reusing {shard}", flush=True)
        rows.append(((t0, n0, T), metrics(shard, dt)))
        m = rows[-1][1]
        print(f"    {m['games']} games, {m['plies']} plies in {m['seconds']:.0f}s "
              f"({m['pos_per_s']:.0f} pos/s)", flush=True)

    print("\n" + "=" * 108)
    print(f"{'schedule':>16} {'games':>6} {'plies':>8} {'median':>7} {'cap%g':>7} {'cap%p':>7} "
          f"{'decis':>6} {'dup%':>6} {'r(v,z)':>7} {'R2':>7} {'r|mat':>7} {'r(m,z)':>7}"
          f" | {'decided only':>13} {'r(v,z)':>7} {'R2':>7}")
    for (t0, n0, T), m in rows:
        print(f"{label_of(t0, n0, T):>16} {m['games']:>6} {m['plies']:>8} {m['median_len']:>7.0f} "
              f"{m['cap_games']:>6.1%} {m['cap_plies']:>6.1%} {m['decisive']:>5.0%} "
              f"{m['dup']:>5.2%} "
              f"{m['r_vz']:>+7.3f} {m['r2']:>+7.3f} {m['partial']:>+7.3f} {m['r_mz']:>+7.3f}"
              f" | {m['n_dec']:>13d} {m['r_dec']:>+7.3f} {m['r2_dec']:>+7.3f}")
    print("""
cap%g / cap%p  share of games / of POSITIONS ending on the ply cap. Those carry result 0
               though nothing was drawn, so cap%p is the fraction of labels that are wrong.
dup%           repeated positions, canonical frame. Rises as play sharpens.
r(v,z), R2     how well the current net's value predicts the eventual outcome.
r|mat          the same correlation controlling for material -- what the value head knows
               beyond counting pieces.
r(m,z)         material against outcome. Does not involve the net at all.

Positions inside a game share one label, so treat the game count as the sample size: these
correlations are noisy below a few hundred games.

CAVEAT on r(v,z) and R2: they mix two things -- how much signal the labels carry, and how
well this particular net fits the schedule's positions. A net trained on one schedule's
games will score best on that schedule regardless. To measure the signal alone, run
--mode variance over the same schedules.""")
    with open(os.path.join(a.out_dir, "sweep.csv"), "w") as f:
        keys = list(rows[0][1])
        f.write("opening_temperature,opening_plies,temperature," + ",".join(keys) + "\n")
        for (t0, n0, T), m in rows:
            f.write(f"{t0},{n0},{T}," + ",".join(str(m[k]) for k in keys) + "\n")
    print(f"\nwrote {a.out_dir}/sweep.csv")


# -------------------------------------------------------------------------- variance mode

def shard_lengths(path):
    """Game lengths and ply cap from a play shard's game headers, skipping the boards."""
    with open(path, "rb") as f:
        h = read_header(f, path)
        L = []
        while True:
            gh = f.read(GAME_HDR)
            if len(gh) < GAME_HDR:
                break
            n = struct.unpack_from("<I", gh, 0)[0]
            L.append(n)
            f.seek(n * REC, 1)
    return np.array(L, dtype=np.float64), h["max_plies"]


def play_rates(out_dir):
    """pos/s per schedule from a previous play sweep, for turning plies into hours."""
    rates = {}
    path = os.path.join(out_dir, "sweep.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            try:
                key = (float(r["opening_temperature"]), int(r["opening_plies"]),
                       float(r["temperature"]))
                rate = float(r["pos_per_s"])
            except (KeyError, ValueError):
                continue
            if rate == rate and rate > 0:
                rates[key] = rate
    return rates


def plan(a, sched, rates):
    tag = tag_of(*sched)
    shard = os.path.join(a.out_dir, f"{tag}.tkn")
    L, cap = shard_lengths(shard) if os.path.exists(shard) else (None, None)
    if a.branch_every == "auto":
        if L is None or not len(L):
            sys.exit(f"{tag}: --branch-every auto takes the stride from the play shard {shard}, "
                     f"which does not exist. Run --mode play over the same schedules first, "
                     f"or give --branch-every N.")
        stride = max(1, int(round(L.mean() / a.points_per_game)))
    else:
        stride = int(a.branch_every)
    p = {"tag": tag, "stride": stride, "plies": None, "hours": None, "cap": cap,
         "mean_len": float(L.mean()) if L is not None and len(L) else None}
    if L is not None and len(L):
        # every ply is a branch point with probability 1/stride, and a playout from ply q
        # runs about as long as the rest of a game that reached q
        per_game = L + a.branches * L * (L + 1) / (2 * stride)
        p["plies"] = float(per_game.mean()) * a.games
        rate = rates.get(tuple(sched))
        if rate:
            p["hours"] = p["plies"] / rate / 3600
    return p


def print_plan(a, scheds, plans):
    print(f"{'schedule':>16} {'mean len':>9} {'stride':>7} {'points/game':>12} "
          f"{'est. plies':>12} {'est. hours':>11}")
    total_h, unknown = 0.0, False
    for s, p in zip(scheds, plans):
        ppg = p["mean_len"] / p["stride"] if p["mean_len"] else float("nan")
        ml = f"{p['mean_len']:.0f}" if p["mean_len"] else "?"
        pl = f"{p['plies']:.3g}" if p["plies"] else "?"
        hr = f"{p['hours']:.1f}" if p["hours"] is not None else "?"
        if p["hours"] is None:
            unknown = True
        else:
            total_h += p["hours"]
        print(f"{label_of(*s):>16} {ml:>9} {p['stride']:>7} {ppg:>12.1f} {pl:>12} {hr:>11}")
        if p["cap"] is not None and p["cap"] != a.max_plies:
            print(f"{'':>16} (play shard used --max-plies {p['cap']}, this run uses "
                  f"{a.max_plies}: the estimate is off)")
    print(f"total {total_h:.1f} h at one job per schedule"
          + (" (plus schedules with no estimate)" if unknown else ""))
    print("Hours use the pos/s the play sweep measured. --jobs J splits each schedule's games "
          "over J\nprocesses, which only helps if one process leaves the GPU idle -- compare "
          "the rate it reports.", flush=True)


def games_done(logs):
    n = 0
    for lp in logs:
        try:
            with open(lp) as f:
                n += sum(1 for line in f if re.match(r"\s+game \d+:", line))
        except OSError:
            pass
    return n


def run_variance_jobs(a, sched, p):
    tag = p["tag"]
    merged = os.path.join(a.out_dir, f"variance_{tag}.csv")
    if a.reuse and os.path.exists(merged):
        print(f"[{time.strftime('%H:%M:%S')}] {tag}: reusing {merged}", flush=True)
        return merged

    split = [a.games // a.jobs + (1 if j < a.games % a.jobs else 0) for j in range(a.jobs)]
    procs, parts, logs = [], [], []
    first = a.first_game
    for j, g in enumerate(split):
        if g == 0:
            continue
        part = os.path.join(a.out_dir, f"variance_{tag}.{j}.csv")
        log = os.path.join(a.out_dir, f"variance_{tag}.{j}.log")
        cmd = [a.nnplay, "variance", "--games", str(g), "--first-game", str(first),
               *schedule_args(*sched),
               "--branch-every", str(p["stride"]), "--branches", str(a.branches),
               "--max-points", str(a.max_points), "--max-plies", str(a.max_plies),
               "--policy-top", str(a.policy_top), "--csv", part] + a.extra.split()
        if not procs:
            print(f"[{time.strftime('%H:%M:%S')}] {tag}: {' '.join(cmd)}"
                  + (f"  (x{a.jobs} jobs)" if a.jobs > 1 else ""), flush=True)
        procs.append(subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT))
        parts.append(part)
        logs.append(log)
        first += g

    start = time.time()
    last = -1
    while any(pr.poll() is None for pr in procs):
        time.sleep(a.poll)
        done = games_done(logs)
        if done != last:
            el = time.time() - start
            eta = el / done * (a.games - done) if done else float("nan")
            eta_s = f"ETA {eta / 60:.0f} min" if eta == eta else "ETA ?"
            print(f"    [{time.strftime('%H:%M:%S')}] {tag}: {done}/{a.games} games, "
                  f"{el / 60:.0f} min, {eta_s}", flush=True)
            last = done
    for pr, log in zip(procs, logs):
        if pr.returncode:
            print(open(log).read()[-3000:])
            sys.exit(f"nnplay variance failed at {tag}; log in {log}")

    with open(merged, "w") as out:
        for i, part in enumerate(parts):
            with open(part) as f:
                header = f.readline()
                if i == 0:
                    out.write(header)
                out.writelines(f)
    print(f"    {tag}: done in {(time.time() - start) / 60:.1f} min -> {merged}", flush=True)
    return merged


def load_points(path):
    d = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if d.size == 0 or not d.dtype.names:
        return None
    return d


def game_totals(d):
    """Per-game sums of everything the statistics need, so a bootstrap is one matmul."""
    _, inv = np.unique(d["game"].astype(np.int64), return_inverse=True)
    z, s2, v = d["mean_outcome"], d["var_outcome"], d["net_value"]
    cols = np.stack([np.ones_like(z), s2, z, z * z, v, v * v, v * z], 1)
    A = np.zeros((inv.max() + 1, cols.shape[1]))
    np.add.at(A, inv, cols)
    return A


def stats(T, K):
    with np.errstate(divide="ignore", invalid="ignore"):
        n, s2, sz, szz, sv, svv, svz = np.moveaxis(np.asarray(T, dtype=np.float64), -1, 0)
        noise = s2 / n
        mz = sz / n
        var_zbar = (szz - n * mz ** 2) / (n - 1)
        # each point's mean still carries noise/K of sampling error
        signal = var_zbar - noise / K
        r2 = signal / (signal + noise)
        mv = sv / n
        sdv = np.sqrt(np.maximum(svv / n - mv ** 2, 0))
        sdz = np.sqrt(np.maximum(szz / n - mz ** 2, 0))
        r = (svz / n - mv * mz) / (sdv * sdz)
        # corr with the noisy K-playout mean, corrected to corr with the true E[z|s]
        r_true = r / np.sqrt(signal / var_zbar)
    return {"noise": noise, "signal": signal, "r2": r2, "r": r, "r_true": r_true}


def ci(x):
    x = x[np.isfinite(x)]
    return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if x.size else (np.nan, np.nan)


def report_variance(a, scheds, plans, paths):
    rng = np.random.default_rng(a.bootstrap_seed)
    res = []
    for s, p, path in zip(scheds, plans, paths):
        d = load_points(path)
        if d is None or d.size < 3:
            print(f"{label_of(*s)}: too few branch points in {path}, skipped")
            continue
        K = int(np.median(d["branches"]))
        A = game_totals(d)
        G = A.shape[0]
        C = rng.multinomial(G, np.full(G, 1.0 / G), size=a.bootstrap)
        res.append({"sched": s, "stride": p["stride"], "K": K, "games": G,
                    "points": int(d.size), "d": d,
                    "point": stats(A.sum(0), K), "boot": stats(C @ A, K)})
    if not res:
        sys.exit("nothing to report")

    ref = res[min(a.reference, len(res) - 1)]
    print("\n" + "=" * 118)
    print(f"{'schedule':>16} {'games':>6} {'points':>7} {'K':>3} {'noise':>7} {'signal':>7} "
          f"{'ACHIEVABLE R2 [95%]':>24} {'vs ' + label_of(*ref['sched']):>28} {'net r(v,E)':>11}")
    for r in res:
        pt, bt = r["point"], r["boot"]
        lo, hi = ci(bt["r2"])
        if r is ref:
            diff = "(reference)"
        else:
            dd = bt["r2"] - ref["boot"]["r2"]
            dlo, dhi = ci(dd)
            ok = np.isfinite(dd)
            pgt = float((dd[ok] > 0).mean()) if ok.any() else float("nan")
            diff = f"{pt['r2'] - ref['point']['r2']:+.3f} [{dlo:+.3f},{dhi:+.3f}] P>0 {pgt:.0%}"
        print(f"{label_of(*r['sched']):>16} {r['games']:>6} {r['points']:>7} {r['K']:>3} "
              f"{pt['noise']:>7.4f} {pt['signal']:>7.4f} "
              f"{pt['r2']:>9.3f} [{lo:.3f},{hi:.3f}] {diff:>28} {pt['r_true']:>+11.3f}")

    edges = [0, 25, 50, 100, 200, 400, 1 << 30]
    names = [f"{lo}-{hi - 1}" if hi < 1 << 29 else f"{lo}+" for lo, hi in zip(edges, edges[1:])]
    print(f"\nachievable R2 by ply (points in brackets; blank under {a.min_bucket})")
    print(f"{'schedule':>16} " + " ".join(f"{n:>13}" for n in names))
    for r in res:
        d, cells = r["d"], []
        for lo, hi in zip(edges, edges[1:]):
            m = (d["ply"] >= lo) & (d["ply"] < hi)
            if m.sum() < a.min_bucket:
                cells.append(f"{'':>13}")
                continue
            st = stats(game_totals(d[m]).sum(0), r["K"])
            cells.append(f"{st['r2']:>6.3f} ({int(m.sum()):>4})")
        print(f"{label_of(*r['sched']):>16} " + " ".join(cells))

    print(f"""
noise          E[Var(z|s)]: outcome variance left after the position is fixed, i.e. decided
               by the moves played after it.
signal         Var(E[z|s]): outcome variance the position itself determines.
ACHIEVABLE R2  signal / (signal + noise), the ceiling on how much of the outcome any value
               head could predict from this schedule's labels. HIGHER IS BETTER. It does not
               depend on how good the generating net is at predicting, only on how
               deterministically the schedule turns positions into results.
vs reference   difference in achievable R2, bootstrap interval, and the bootstrap share of
               draws where this schedule beats the reference. An interval that includes 0
               has not been resolved -- run more games rather than reading the sign.
net r(v,E)     correlation of the current net's value with the true E[z|s], corrected for
               the noise in a {ref['K']}-playout mean. How much of the learnable signal this net
               already has on each schedule's positions; low on a schedule unlike the one it
               was trained on. Noisy, can exceed 1.

Intervals resample whole games ({a.bootstrap} draws), since points in a game share a future.
The branch stride is set per schedule so each game gets about the same number of points;
every ply is sampled with the same probability, so the positions are weighted as the
corpus weights them. Read alongside dup% and cap% from the play sweep: a schedule that buys
R2 by replaying the same lines is not better.""")

    out = os.path.join(a.out_dir, "variance.csv")
    with open(out, "w") as f:
        f.write("opening_temperature,opening_plies,temperature,stride,branches,games,points,"
                "noise,signal,r2,r2_lo,r2_hi,d_r2,d_r2_lo,d_r2_hi,p_better,net_r_true\n")
        for r in res:
            pt, bt = r["point"], r["boot"]
            lo, hi = ci(bt["r2"])
            dd = bt["r2"] - ref["boot"]["r2"]
            dlo, dhi = ci(dd) if r is not ref else (0.0, 0.0)
            ok = np.isfinite(dd)
            pgt = float((dd[ok] > 0).mean()) if r is not ref and ok.any() else float("nan")
            t0, n0, T = r["sched"]
            f.write(f"{t0},{n0},{T},{r['stride']},{r['K']},{r['games']},{r['points']},"
                    f"{pt['noise']},{pt['signal']},{pt['r2']},{lo},{hi},"
                    f"{pt['r2'] - ref['point']['r2']},{dlo},{dhi},{pgt},{pt['r_true']}\n")
    print(f"\nwrote {out}")


def run_variance(a, scheds):
    rates = play_rates(a.out_dir)
    plans = [plan(a, s, rates) for s in scheds]
    print_plan(a, scheds, plans)
    if a.dry_run:
        return
    print()
    paths = [run_variance_jobs(a, s, p) for s, p in zip(scheds, plans)]
    report_variance(a, scheds, plans, paths)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("play", "variance"), default="play")
    ap.add_argument("--temperatures", default="", help="flat schedules, comma separated")
    ap.add_argument("--schedules", default="",
                    help="OPENING_TEMP:OPENING_PLIES:MAIN_TEMP, comma separated")
    ap.add_argument("--games", type=int, default=None,
                    help="games per schedule (default 40 in play mode, 200 in variance mode)")
    ap.add_argument("--max-plies", type=int, default=1000)
    ap.add_argument("--policy-top", type=float, default=0.05,
                    help="the value head scores the policy's top this much of the moves")
    ap.add_argument("--out-dir", default="sweep")
    ap.add_argument("--nnplay", default="./nnplay")
    ap.add_argument("--extra", default="", help="extra nnplay arguments, space separated")
    ap.add_argument("--reuse", action="store_true",
                    help="skip any schedule whose output already exists, e.g. to resume")
    v = ap.add_argument_group("variance mode")
    v.add_argument("--branches", type=int, default=8, help="playouts per branch point")
    v.add_argument("--branch-every", default="auto",
                   help="branch stride in plies, or 'auto': the play shard's mean game length "
                        "divided by --points-per-game")
    v.add_argument("--points-per-game", type=float, default=8)
    v.add_argument("--max-points", type=int, default=0,
                   help="cap per game (0 = none; a cap drops late positions, biasing R2)")
    v.add_argument("--jobs", type=int, default=1, help="nnplay processes per schedule")
    v.add_argument("--first-game", type=int, default=0)
    v.add_argument("--bootstrap", type=int, default=2000)
    v.add_argument("--bootstrap-seed", type=int, default=1)
    v.add_argument("--reference", type=int, default=0,
                   help="index of the schedule the others are compared against")
    v.add_argument("--min-bucket", type=int, default=30)
    v.add_argument("--poll", type=float, default=60, help="seconds between progress lines")
    v.add_argument("--dry-run", action="store_true", help="print the cost estimate and stop")
    a = ap.parse_args()
    if a.games is None:
        a.games = 40 if a.mode == "play" else 200
    if a.jobs < 1 or a.branches < 2:
        sys.exit("--jobs must be >= 1 and --branches >= 2")

    scheds = []
    for t in a.temperatures.split(","):
        if t.strip():
            scheds.append((1.0, 0, float(t)))
    for spec in a.schedules.split(","):
        if spec.strip():
            t0, n, t1 = spec.split(":")
            scheds.append((float(t0), int(n), float(t1)))
    if not scheds:
        scheds = [(1.0, 0, t) for t in (0.3, 0.5, 0.7, 1.0)]

    os.makedirs(a.out_dir, exist_ok=True)
    (run_play if a.mode == "play" else run_variance)(a, scheds)


if __name__ == "__main__":
    main()
