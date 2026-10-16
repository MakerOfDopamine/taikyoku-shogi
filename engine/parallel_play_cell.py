def _parallel_play(merge_tkn):
    # ============================== settings ==============================
    PROCS = 2                  # nnplay processes sharing the GPU
    POSITIONS = 1_000_000      # positions to keep in all; every process stops once the
                               # total gets there
    ARGS = ["--endgames", "--temperature", "0.05", "--max-plies", "1000",
            "--opening-plies", "0"]
    CHILDREN = True            # also record .kids for policy training
    SUB_BATCH = 256            # fixed, so no process runs the startup probe and grabs the
                               # memory for an 8192-board batch; 256 is one full round of
                               # 32 games x 8 leaves
    WORKDIR = "."              # where ./nnplay lives
    OUT = "endgames.tkn.tgz"   # the merged archive, relative to WORKDIR
    KEEP_PARTS = False         # keep each process's own .tkn/.kids/.log after merging
    POLL_S = 60
    # ======================================================================
    #
    # One nnplay process already plays --parallel-games games in lockstep (32 by default)
    # and pools their search leaves into each GPU call. Its tree walks, move generation and
    # encoding run on one CPU thread, though, so once the GPU is fast enough that thread is
    # the limit; a second process uses a second core. Check the status lines' "network %
    # of the wall clock": well under ~80% means the CPU is the limit and more processes
    # help. Each draws its own random seed, so their games never repeat each other.
    #
    # Each process is given the whole target, and all of them are stopped together once
    # their combined total reaches it. A fixed share each would leave the unlucky last one
    # running alone at the end. Stopping costs each process only the game it was playing,
    # which the merge drops as truncated.
    #
    # The parts go to parallel_<time>/ under WORKDIR and are merged into OUT when every
    # process has finished. If the session dies first, finished games are still on disk:
    # merge_tkn(sorted(glob.glob("parallel_<time>/*.tkn") + glob.glob("parallel_<time>/*.kids")),
    # "endgames.tkn.tgz") recovers them.

    import os, re, subprocess, time

    work = os.path.abspath(WORKDIR)
    nnplay = os.path.join(work, "nnplay")
    if not os.access(nnplay, os.X_OK):
        raise RuntimeError(f"no executable nnplay at {nnplay}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parts = os.path.join(work, f"parallel_{stamp}")
    os.makedirs(parts)

    procs, logs = [], []
    for i in range(PROCS):
        tkn = os.path.join(parts, f"part{i:02d}.tkn")
        cmd = [nnplay, "play", "--positions", str(POSITIONS), "--out", tkn,
               "--sub-batch", str(SUB_BATCH), "--progress-every", "0"] + list(ARGS)
        if CHILDREN:
            cmd += ["--children", os.path.join(parts, f"part{i:02d}.kids")]
        log = open(os.path.join(parts, f"part{i:02d}.log"), "w")
        logs.append(log)
        procs.append(subprocess.Popen(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT))
    print(f"{PROCS} processes, {POSITIONS} positions in all, parts in {parts}")
    print("  " + " ".join(cmd).replace(nnplay, "./nnplay"), flush=True)

    line = re.compile(rb"Game (\d+), Positions (\d+)/\d+")

    def progress(i):
        """(games played, positions kept) from the last per-game line of process i's log"""
        try:
            with open(logs[i].name, "rb") as f:
                f.seek(max(0, os.path.getsize(logs[i].name) - 4096))
                m = line.findall(f.read())
        except OSError:
            return 0, 0
        return (int(m[-1][0]), int(m[-1][1])) if m else (0, 0)

    t0 = time.time()
    stopped = False
    try:
        while any(p.poll() is None for p in procs):
            time.sleep(POLL_S)
            got = [progress(i) for i in range(PROCS)]
            kept, played = sum(k for _, k in got), sum(g for g, _ in got)
            el = time.time() - t0
            eta = el / kept * max(0, POSITIONS - kept) if kept else 0
            running = sum(p.poll() is None for p in procs)
            print(f"{time.strftime('%H:%M:%S')}  {kept}/{POSITIONS} positions kept "
                  f"({kept / POSITIONS:.1%}) from {played} games, {kept / el:.0f}/s, "
                  f"{running} running, ETA {eta / 3600:.1f} h", flush=True)
            if kept >= POSITIONS:
                stopped = True
                for p in procs:
                    p.terminate()
                for p in procs:
                    p.wait()
    except BaseException:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()
        print(f"stopped; finished games are in {parts}, merge them with merge_tkn")
        raise
    finally:
        for log in logs:
            log.close()

    failed = [i for i, p in enumerate(procs)
              if p.returncode != 0 and not (stopped and p.returncode == -15)]
    for i in failed:
        with open(logs[i].name, errors="replace") as f:
            tail = f.read().splitlines()[-8:]
        print(f"\nprocess {i} exited with {procs[i].returncode}:\n    " + "\n    ".join(tail))
    print(f"\nall {PROCS} processes done in {(time.time() - t0) / 3600:.2f} h"
          + (f", {len(failed)} failed" if failed else "")
          + ("; the target was met, so the rest were stopped and lose only the game each was "
             "playing" if stopped else "") + "; merging\n", flush=True)

    inputs = sorted(os.path.join(parts, n) for n in os.listdir(parts)
                    if n.endswith((".tkn", ".kids")))
    out = os.path.join(work, OUT)
    tot = merge_tkn(inputs, out, remove_inputs=not KEEP_PARTS)
    if not KEEP_PARTS and not failed:
        for n in os.listdir(parts):
            os.remove(os.path.join(parts, n))
        os.rmdir(parts)
    return {"out": out, "failed": failed, **tot}


parallel_play = _parallel_play(merge_tkn)
