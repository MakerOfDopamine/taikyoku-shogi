def _variance_sweep():
    # ============================== settings ==============================
    # (opening temperature, opening plies, main temperature); opening plies 0 = flat
    SCHEDULES = [(1.0, 0, 1.0), (1.0, 0, 0.5), (1.0, 0, 0.3), (1.5, 20, 0.3)]
    MODE = "estimate"         # "estimate": cost and progress only, runs nothing
                              # "run":      play chunks until SESSION_HOURS, then report
                              # "report":   report on whatever has finished so far
    SESSION_HOURS = 9.0       # no new chunk starts after this; running ones finish
    GAMES = 300               # target per schedule; stop earlier once the report resolves
    CHUNK = 10                # games per saved chunk: a dead session loses at most one
    BRANCHES = 4              # playouts per snapshot
    POINTS_PER_GAME = 8       # snapshot stride = mean game length / this
    MAX_PLIES = 1000
    POLICY_TOP = 0.05           # the value head scores the policy's top this much of the moves
    EXTRA = ["--sub-batch", "32"]
    JOBS = 1                  # nnplay processes at once; >1 only helps an idle GPU
    LENGTH_GAMES = 40         # play games to measure lengths when sweep/<tag>.tkn is missing
    RATE_POS_PER_S = 70.0     # cost estimate until real chunk timings exist
    REFERENCE = 0             # index into SCHEDULES the others are compared against
    WORKDIR = "."             # where ./nnplay lives
    OUT_DIR = "sweep"         # relative to WORKDIR; the play sweep's shards live here
    BOOTSTRAP = 2000
    POLL_S = 60
    # ======================================================================
    #
    # Var(z) = Var(E[z|s]) + E[Var(z|s)]: the part of the outcome the position decides
    # (signal) and the part decided by moves played after it (noise). ACHIEVABLE R2 =
    # signal / (signal + noise) is the most any value head could learn from a schedule's
    # labels, independent of how good the generating net is. Higher is better.
    #
    # Chunks are taken round-robin across schedules, so every schedule has about the same
    # number of games whenever the session stops. Re-run the cell to continue. Finished
    # chunks live in sweep/variance_<tag>_K<branches>_m<max plies>_p<policy top>/; changing
    # BRANCHES, MAX_PLIES or POLICY_TOP starts a separate set rather than mixing.

    import glob, os, re, struct, subprocess, time
    import numpy as np

    REC, FILE_HDR, GAME_HDR = 1296 * 2 + 16, 128, 16
    if MODE not in ("estimate", "run", "report"):
        raise RuntimeError(f"MODE must be estimate, run or report, not {MODE!r}")
    work = os.path.abspath(WORKDIR)
    nnplay = os.path.join(work, "nnplay")
    out = os.path.join(work, OUT_DIR)
    os.makedirs(out, exist_ok=True)
    if MODE != "report" and not os.access(nnplay, os.X_OK):
        raise RuntimeError(f"no executable nnplay at {nnplay}")

    def tag_of(t0, n0, T):
        return f"T{T:g}" if n0 == 0 else f"T{t0:g}x{n0}_{T:g}"

    def label_of(t0, n0, T):
        return f"flat {T:g}" if n0 == 0 else f"{t0:g} x{n0} -> {T:g}"

    def sched_args(t0, n0, T):
        return ["--temperature", str(T), "--opening-plies", str(n0),
                "--opening-temperature", str(t0)]

    def stamp():
        return time.strftime("%H:%M:%S")

    def shard_lengths(path):
        with open(path, "rb") as f:
            h = f.read(FILE_HDR)
            if len(h) != FILE_HDR or h[:8] != b"TKYNNSP\x00":
                raise RuntimeError(f"{path}: not a .tkn shard")
            if struct.unpack_from("<I", h, 20)[0] != REC:
                raise RuntimeError(f"{path}: unexpected ply record size")
            cap = struct.unpack_from("<I", h, 64)[0]
            L = []
            while True:
                gh = f.read(GAME_HDR)
                if len(gh) < GAME_HDR:
                    break
                n = struct.unpack_from("<I", gh, 0)[0]
                L.append(n)
                f.seek(n * REC, 1)
        return np.array(L, dtype=np.float64), cap

    def done_chunks(d):
        """{first game: n games} for every finished chunk in d."""
        got = {}
        for p in glob.glob(os.path.join(d, "g*_n*.csv")):
            m = re.match(r"g(\d+)_n(\d+)\.csv$", os.path.basename(p))
            if m:
                got[int(m.group(1))] = int(m.group(2))
        return got

    def timings(d):
        """Seconds per game from every chunk ever finished for this schedule."""
        secs = games = 0.0
        try:
            for line in open(os.path.join(d, "timing.csv")):
                a = line.strip().split(",")
                if len(a) == 3 and a[0].isdigit():
                    games += int(a[1])
                    secs += float(a[2])
        except OSError:
            pass
        return secs / games if games else None

    # ---------------------------------------------------------------- plan
    plans = []
    for s in SCHEDULES:
        tag = tag_of(*s)
        d = os.path.join(out, f"variance_{tag}_K{BRANCHES}_m{MAX_PLIES}_p{POLICY_TOP:g}")
        p = {"sched": s, "tag": tag, "dir": d, "stride": None, "mean_len": None, "cap": None}
        shard = os.path.join(out, f"{tag}.tkn")
        if not os.path.exists(shard) and MODE == "run":
            cmd = [nnplay, "play", "--games", str(LENGTH_GAMES), *sched_args(*s),
                   "--max-plies", str(MAX_PLIES), "--policy-top", str(POLICY_TOP),
                   "--quiet", "--out", shard] + EXTRA
            print(f"[{stamp()}] {tag}: measuring game length: {' '.join(cmd)}", flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=work)
            if r.returncode:
                raise RuntimeError(f"nnplay play failed for {tag}:\n{r.stderr[-2000:]}")
        if os.path.exists(shard):
            L, cap = shard_lengths(shard)
            if len(L):
                p["stride"] = max(1, int(round(L.mean() / POINTS_PER_GAME)))
                p["mean_len"], p["cap"] = float(L.mean()), cap
                # every ply is a snapshot with probability 1/stride, and a playout from ply
                # q runs about as long as the rest of a game that reached q
                plies = float((L + BRANCHES * L * (L + 1) / (2 * p["stride"])).mean())
                p["est_s_per_game"] = plies / RATE_POS_PER_S
        p["done"] = done_chunks(d)
        p["s_per_game"] = timings(d)
        plans.append(p)

    def games_of(p):
        return sum(p["done"].values())

    def s_per_game(p):
        return p["s_per_game"] or p.get("est_s_per_game")

    def show_plan():
        print(f"\n{'schedule':>16} {'mean len':>9} {'stride':>7} {'games':>9} "
              f"{'h/100 games':>12} {'h to target':>12}")
        total = 0.0
        for p in plans:
            spg = s_per_game(p)
            left = max(0, GAMES - games_of(p))
            src = "" if p["s_per_game"] else " (est.)"
            if spg:
                total += left * spg / 3600
            ml = f"{p['mean_len']:.0f}" if p["mean_len"] else "?"
            st = str(p["stride"]) if p["stride"] else "?"
            h100 = f"{100 * spg / 3600:.1f}" if spg else "?"
            htg = f"{left * spg / 3600:.1f}{src}" if spg else "?"
            print(f"{label_of(*p['sched']):>16} {ml:>9} {st:>7} "
                  f"{games_of(p):>4}/{GAMES:<4} {h100:>12} {htg:>12}")
            if p["cap"] is not None and p["cap"] != MAX_PLIES:
                print(f"{'':>16} (length shard played with max plies {p['cap']}, not "
                      f"{MAX_PLIES}: the estimate is off)")
            if p["stride"] is None:
                print(f"{'':>16} (no sweep/{p['tag']}.tkn yet: a run first plays "
                      f"{LENGTH_GAMES} games to measure length)")
        print(f"~{total / JOBS:.1f} h to reach {GAMES} games each"
              f" ({total / JOBS / SESSION_HOURS:.1f} sessions of {SESSION_HOURS:g} h). "
              f"'(est.)' rows use {RATE_POS_PER_S:g} pos/s until a chunk has finished.",
              flush=True)

    show_plan()
    if MODE == "estimate":
        print("\nMODE is 'estimate': nothing was run. Set MODE = \"run\" to start.")
        return

    # ----------------------------------------------------------------- run
    if MODE == "run":
        queue = []
        for i in range(0, GAMES, CHUNK):          # round-robin: chunk i of every schedule
            for p in plans:
                if i not in p["done"]:
                    queue.append((p, i, min(CHUNK, GAMES - i)))
        deadline = time.time() + SESSION_HOURS * 3600
        session_start = time.time()
        running, last_beat = [], -1

        def totals_line():
            return " | ".join(f"{p['tag']} {games_of(p)}/{GAMES}" for p in plans)

        try:
            while queue or running:
                while queue and len(running) < JOBS and time.time() < deadline:
                    p, first, n = queue.pop(0)
                    os.makedirs(p["dir"], exist_ok=True)
                    part = os.path.join(p["dir"], f"g{first:06d}_n{n}.partial")
                    log = os.path.join(p["dir"], f"g{first:06d}_n{n}.log")
                    cmd = [nnplay, "variance", "--games", str(n), "--first-game", str(first),
                           *sched_args(*p["sched"]), "--branch-every", str(p["stride"]),
                           "--branches", str(BRANCHES), "--max-plies", str(MAX_PLIES),
                           "--policy-top", str(POLICY_TOP), "--csv", part] + EXTRA
                    print(f"[{stamp()}] {p['tag']}: games {first}-{first + n - 1}", flush=True)
                    with open(log, "w") as lf:
                        pr = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=work)
                    running.append((pr, p, first, n, part, log, time.time()))
                if not running:
                    print(f"[{stamp()}] session budget of {SESSION_HOURS:g} h reached",
                          flush=True)
                    break
                time.sleep(POLL_S)
                for job in [j for j in running if j[0].poll() is not None]:
                    running.remove(job)
                    pr, p, first, n, part, log, t = job
                    text = open(log).read()
                    if pr.returncode and "too few branch points" not in text:
                        raise RuntimeError(f"nnplay variance failed at {p['tag']} games "
                                           f"{first}-{first + n - 1} (log {log}):\n"
                                           f"{text[-3000:]}")
                    dt = time.time() - t
                    os.replace(part, os.path.join(p["dir"], f"g{first:06d}_n{n}.csv"))
                    with open(os.path.join(p["dir"], "timing.csv"), "a") as tf:
                        tf.write(f"{first},{n},{dt:.1f}\n")
                    p["done"][first] = n
                    p["s_per_game"] = timings(p["dir"])
                    print(f"[{stamp()}] {p['tag']}: games {first}-{first + n - 1} done in "
                          f"{dt / 60:.1f} min ({dt / n / 60:.1f} min/game)  ||  {totals_line()}"
                          f"  ||  session {(time.time() - session_start) / 3600:.1f}/"
                          f"{SESSION_HOURS:g} h", flush=True)
                # heartbeat from the per-game lines nnplay logs, so a long chunk is not silent
                beat = 0
                for pr, p, first, n, part, log, t in running:
                    try:
                        beat += sum(1 for line in open(log) if re.match(r"\s+game \d+:", line))
                    except OSError:
                        pass
                if running and beat != last_beat:
                    now = ", ".join(f"{p['tag']} g{first}+" for _, p, first, *_ in running)
                    print(f"    [{stamp()}] {beat} games into the running chunk(s): {now}",
                          flush=True)
                    last_beat = beat
        finally:
            # an interrupted cell must not leave nnplay running on the GPU
            for pr, *_ in running:
                if pr.poll() is None:
                    pr.terminate()
                    try:
                        pr.wait(10)
                    except subprocess.TimeoutExpired:
                        pr.kill()
        show_plan()

    # -------------------------------------------------------------- report
    def load(p):
        rows = []
        for first in sorted(p["done"]):
            path = os.path.join(p["dir"], f"g{first:06d}_n{p['done'][first]}.csv")
            d = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
            if d.size and d.dtype.names:
                rows.append(d)
        return np.concatenate(rows) if rows else None

    def totals(d):
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
            signal = var_zbar - noise / K        # each mean still carries noise/K
            mv = sv / n
            sdv = np.sqrt(np.maximum(svv / n - mv ** 2, 0))
            sdz = np.sqrt(np.maximum(szz / n - mz ** 2, 0))
            r = (svz / n - mv * mz) / (sdv * sdz)
            return {"noise": noise, "signal": signal, "r2": signal / (signal + noise),
                    "r_true": r / np.sqrt(signal / var_zbar)}

    def ci(x):
        x = x[np.isfinite(x)]
        return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if x.size else (np.nan, np.nan)

    rng = np.random.default_rng(1)
    res = []
    for p in plans:
        d = load(p)
        if d is None or d.size < 3:
            print(f"{label_of(*p['sched'])}: no finished chunks yet")
            continue
        K = int(np.median(d["branches"]))
        A = totals(d)
        G = A.shape[0]
        C = rng.multinomial(G, np.full(G, 1.0 / G), size=BOOTSTRAP)
        res.append({"p": p, "K": K, "games": G, "points": int(d.size), "d": d,
                    "pt": stats(A.sum(0), K), "bt": stats(C @ A, K)})
    if not res:
        print("nothing finished yet to report on")
        return
    ref = next((r for r in res if r["p"] is plans[min(REFERENCE, len(plans) - 1)]), res[0])

    print("\n" + "=" * 128)
    print(f"{'schedule':>16} {'games':>6} {'points':>7} {'K':>3} {'noise':>7} {'signal':>7} "
          f"{'ACHIEVABLE R2 [95%]':>24} {'vs ' + label_of(*ref['p']['sched']):>32} "
          f"{'resolves at':>12} {'net r(v,E)':>11}")
    rows = []
    for r in res:
        pt, bt = r["pt"], r["bt"]
        lo, hi = ci(bt["r2"])
        dd = bt["r2"] - ref["bt"]["r2"]
        dlo, dhi = ci(dd)
        ok = np.isfinite(dd)
        pgt = float((dd[ok] > 0).mean()) if ok.any() and r is not ref else float("nan")
        delta = pt["r2"] - ref["pt"]["r2"]
        if r is ref:
            diff, need = "(reference)", ""
        else:
            diff = f"{delta:+.3f} [{dlo:+.3f},{dhi:+.3f}] P>0 {pgt:.0%}"
            # the interval narrows as 1/sqrt(games); games at which it would clear zero,
            # if the difference stays where it is now
            g = min(r["games"], ref["games"])
            half = (dhi - dlo) / 2
            if dlo > 0 or dhi < 0:
                need = "resolved"
            elif abs(delta) > 1e-9 and np.isfinite(half):
                need = f"~{int(np.ceil(g * (half / abs(delta)) ** 2))} games"
            else:
                need = "?"
        print(f"{label_of(*r['p']['sched']):>16} {r['games']:>6} {r['points']:>7} {r['K']:>3} "
              f"{pt['noise']:>7.4f} {pt['signal']:>7.4f} {pt['r2']:>9.3f} [{lo:.3f},{hi:.3f}] "
              f"{diff:>32} {need:>12} {pt['r_true']:>+11.3f}")
        rows.append((r, lo, hi, dlo, dhi, pgt))

    edges = [0, 25, 50, 100, 200, 400, 1 << 30]
    names = [f"{a}-{b - 1}" if b < 1 << 29 else f"{a}+" for a, b in zip(edges, edges[1:])]
    print("\nachievable R2 by ply (snapshots in brackets; blank under 30)")
    print(f"{'schedule':>16} " + " ".join(f"{n:>13}" for n in names))
    for r in res:
        d, cells = r["d"], []
        for a, b in zip(edges, edges[1:]):
            m = (d["ply"] >= a) & (d["ply"] < b)
            if m.sum() < 30:
                cells.append(" " * 13)
            else:
                cells.append(f"{stats(totals(d[m]).sum(0), r['K'])['r2']:>6.3f} ({int(m.sum()):>4})")
        print(f"{label_of(*r['p']['sched']):>16} " + " ".join(cells))

    print(f"""
noise          E[Var(z|s)]: outcome variance decided by the moves played after s.
signal         Var(E[z|s]): outcome variance the position s itself decides.
ACHIEVABLE R2  signal / (signal + noise): the ceiling on what any value head could learn
               from this schedule's labels. HIGHER IS BETTER. Independent of how well the
               generating net predicts.
vs reference   difference in achievable R2, its 95% interval, and the share of bootstrap
               draws where this schedule wins. An interval spanning 0 is unresolved.
resolves at    games per schedule at which that interval would clear 0 if the difference
               held where it is. A rough guide to whether another session is worth it: a
               difference too small to matter can need thousands.
net r(v,E)     the current net's correlation with the true E[z|s], corrected for the noise
               in a K-playout mean: how much of the learnable signal it already has on each
               schedule's positions. Noisy; can exceed 1.

Intervals resample whole games ({BOOTSTRAP} draws), since snapshots in a game share a future.
Read alongside dup% and cap% from the play sweep: R2 bought by replaying the same lines, or
by games ending on the ply cap, is not an improvement.""")

    csv_path = os.path.join(out, "variance.csv")
    with open(csv_path, "w") as f:
        f.write("opening_temperature,opening_plies,temperature,stride,branches,games,points,"
                "noise,signal,r2,r2_lo,r2_hi,d_r2,d_r2_lo,d_r2_hi,p_better,net_r_true\n")
        for r, lo, hi, dlo, dhi, pgt in rows:
            t0, n0, T = r["p"]["sched"]
            pt = r["pt"]
            f.write(f"{t0},{n0},{T},{r['p']['stride']},{r['K']},{r['games']},{r['points']},"
                    f"{pt['noise']},{pt['signal']},{pt['r2']},{lo},{hi},"
                    f"{pt['r2'] - ref['pt']['r2']},{dlo},{dhi},{pgt},{pt['r_true']}\n")
    print(f"\nwrote {csv_path}")


_variance_sweep()
