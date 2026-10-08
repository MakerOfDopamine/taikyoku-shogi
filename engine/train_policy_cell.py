def _train_policy():
    # ================================= settings =================================
    DATA_DIR = "."                     # every *.tkn.tgz here is used, and every bare
                                       # .tkn (with a .kids beside it, if there is one)
    CACHE_DIR = "tkn_cache"            # archives are unpacked here once, then memory-mapped
    MODEL_DIR = "."                    # where model_policy.py lives
    ENGINE_DIR = "."                   # tky.c, tables.h, tkykids.c: built into the move
                                       # generator for archives that have no .kids file
    CHECKPOINT_IN = "checkpoint_policy.pt"     # e.g. the distillation cell's output
    CHECKPOINT_OUT = "checkpoint_policy2.pt"
    LOAD_OPTIMIZER = True
    BATCH_SIZE = 64
    STEP_SIZE = 256 // 64              # batches accumulated per optimizer step
    LR, WEIGHT_DECAY = 3e-4, 0.01
    EPOCHS = 1
    MAX_STEPS = None                   # optimizer steps; None = the whole epoch(s)
    GAMMA, LAMBDA = 0.999, 0.99        # the value head's TD(lambda) target, as engine_marimo
    POLICY_WEIGHT = 1.0                # total loss = value MSE + POLICY_WEIGHT * policy CE
    POLICY_TEMPERATURE = 0.3           # target = softmax(-child value / T), what nnplay samples
    ONFLY_CHILDREN = 16                # children scored per position when there is no .kids
    EVAL_GAME_FRACTION = 0.02          # games held out for the recall report
    EVAL_POSITIONS = 2000
    EVAL_EVERY = 500                   # optimizer steps between reports
    FULL_EVAL_POSITIONS = 200          # positions ranked over ALL their legal moves...
    FULL_EVAL_EVERY = 2500             # ...this often (and at the start and end); ~1000
                                       # child evaluations per position, so keep it small
    SAVE_EVERY = 1000                  # optimizer steps between checkpoint saves
    PRINT_EVERY = 50
    SEED = 0
    # ============================================================================
    #
    # Trains the value head and the policy head together. The policy's job is to be a cheap
    # stand-in for the value head inside the search: one forward pass over the parent gives
    # a logit for every legal move, and the question is how well those logits pick out the
    # children the value head itself would rank best -- if well, the search can evaluate
    # only the policy's top few instead of hundreds of children.
    #
    # Children come from a .kids side file (nnplay play --children) where an archive has
    # one: every child the search scored, with the value it got, so the policy learns from
    # values that cost nothing extra. Archives without one fall back to generating the
    # legal moves here and scoring ONFLY_CHILDREN of them with the model's own value head.
    #
    # The report ranks each held-out position's children by the policy and asks what the
    # search would get if it evaluated only the policy's top q of them: whether the value
    # head's best child is in there, how much of the target softmax is, and how much value
    # is given up against the best child. Each row sits above what a random pick of the
    # same size would get. A second report does the same over every legal move of a few
    # positions, which is the question that matters for cutting the search down.
    #
    # Games appearing twice -- an archive beside its own unpacked .tkn, a merged file
    # beside its sources -- are counted once, matched by seed, length and move sequence.

    import ctypes
    import glob
    import hashlib
    import os
    import struct
    import subprocess
    import sys
    import tarfile
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F

    REC = 1296 * 2 + 16
    DT = np.dtype([("board", "<u2", (1296,)), ("move", "<u4"), ("value", "<f4"),
                   ("n_legal", "<u2"), ("material", "<i2"), ("side", "u1"), ("flags", "u1"),
                   ("n_scored", "<u2")])
    KID = np.dtype([("move", "<u4"), ("value", "<f2")])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    def say(*a):
        print(*a, flush=True)

    # ---- unpack every archive once --------------------------------------------------
    archives = sorted(glob.glob(os.path.join(DATA_DIR, "*.tkn.tgz")))
    if not archives and not glob.glob(os.path.join(DATA_DIR, "*.tkn")):
        raise RuntimeError(f"no *.tkn.tgz or *.tkn in {os.path.abspath(DATA_DIR)}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    dirs = []
    for path in archives:
        st = os.stat(path)
        dest = os.path.join(CACHE_DIR, os.path.basename(path)[: -len(".tkn.tgz")])
        stamp = os.path.join(dest, ".unpacked")
        mark = f"{st.st_size} {int(st.st_mtime)}"
        dirs.append(dest)
        if os.path.exists(stamp) and open(stamp).read() == mark:
            continue
        os.makedirs(dest, exist_ok=True)
        t0 = time.time()
        with tarfile.open(path, "r|gz") as tf:
            for m in tf:
                if m.isfile() and m.name.endswith((".tkn", ".kids")):
                    with tf.extractfile(m) as src, open(os.path.join(dest, os.path.basename(m.name)), "wb") as dst:
                        while True:
                            b = src.read(1 << 24)
                            if not b:
                                break
                            dst.write(b)
        open(stamp, "w").write(mark)
        say(f"unpacked {path} in {time.time() - t0:.0f}s")

    # ---- index: positions, value labels, children -----------------------------------
    def fnv_games(moves_2d, lengths):
        """FNV-1a 64 over each game's move words, vectorised across games."""
        h = np.full(len(lengths), 0xcbf29ce484222325, dtype=np.uint64)
        prime = np.uint64(0x100000001b3)
        with np.errstate(over="ignore"):
            for j in range(moves_2d.shape[1]):
                act = lengths > j
                m = moves_2d[act, j].astype(np.uint64)
                for b in range(4):
                    x = h[act] ^ ((m >> np.uint64(8 * b)) & np.uint64(0xFF))
                    h[act] = x * prime
        return h

    def fnv_file(mm, games):
        """fnv_games over a file's games, a thousand at a time so memory stays small."""
        out = np.empty(len(games), np.uint64)
        for c in range(0, len(games), 1000):
            chunk = games[c:c + 1000]
            L = np.array([g[1] for g in chunk])
            M = np.zeros((len(chunk), max(1, int(L.max()))), np.uint32)
            for gi, (off, n, _r, _s) in enumerate(chunk):
                M[gi, :n] = np.ndarray((n,), DT, buffer=mm, offset=off)["move"]
            out[c:c + len(chunk)] = fnv_games(M, L)
        return out

    # the unpacked archives, plus whatever bare .tkn and .kids files sit in DATA_DIR itself
    sources = dirs + [DATA_DIR]
    tkn_maps, kids_maps = [], []
    kids_by_key = {}                       # (seed, n_plies, move hash) -> children, any file
    for d in sources:
        for kp in sorted(glob.glob(os.path.join(d, "*.kids"))):
            km = np.memmap(kp, dtype=np.uint8, mode="r")
            if bytes(km[:8]) != b"TKYKIDS\0":
                say(f"  {kp}: not a .kids file, ignored")
                continue
            ki = len(kids_maps)
            kids_maps.append(km)
            p = 32
            while p + 24 <= len(km):
                n, _r, seed, hsh = struct.unpack_from("<IIQQ", km, p)
                p += 24
                offs = np.empty(n, np.int64)
                cnts = np.empty(n, np.int32)
                for i in range(n):
                    c = struct.unpack_from("<H", km, p)[0]
                    offs[i], cnts[i] = p + 2, c
                    p += 2 + 6 * c
                if p > len(km):
                    break                         # a truncated last game
                kids_by_key[(seed, n, hsh)] = (ki, offs, cnts)

    pos_file, pos_off, pos_stm, pos_label, pos_game = [], [], [], [], []
    pos_kfile, pos_koff, pos_kn = [], [], []
    n_games = dup_games = n_bare = 0
    seen = set()
    t0 = time.time()
    for d in sources:
        for tp in sorted(glob.glob(os.path.join(d, "*.tkn"))):
            mm = np.memmap(tp, dtype=np.uint8, mode="r")
            if bytes(mm[:8]) != b"TKYNNSP\0":
                say(f"  {tp}: not a .tkn file, ignored")
                continue
            fi = len(tkn_maps)
            tkn_maps.append(mm)
            n_bare += d is DATA_DIR
            games = []
            p = 128
            while p + 16 <= len(mm):
                n, res = struct.unpack_from("<Ib", mm, p)
                seed = struct.unpack_from("<Q", mm, p + 8)[0]
                if p + 16 + n * REC > len(mm):
                    break                         # a killed run's half-written game
                games.append((p + 16, n, res, seed))
                p += 16 + n * REC
            hashes = fnv_file(mm, games) if kids_by_key and games else None
            for gi, (off, n, res, seed) in enumerate(games):
                if n < 2:
                    continue
                recs = np.ndarray((n,), DT, buffer=mm, offset=off)
                key = (seed, n, hashlib.blake2b(recs["move"].tobytes(), digest_size=16).digest())
                if key in seen:
                    dup_games += 1                 # the same game from another file
                    continue
                seen.add(key)
                v = recs["value"].astype(np.float64)
                stm = 1 - recs["side"].astype(np.int64)        # side to move after each ply
                z_last = res if stm[-1] == 1 else -res
                # engine_marimo's TD(lambda) recursion, unchanged
                labels = np.empty(n - 1)
                accum = z_last - v[-1]
                for k in range(n - 2, -1, -1):
                    accum *= GAMMA * LAMBDA * -1
                    accum += -v[k + 1] - v[k]
                    labels[k] = v[k] + accum
                kid = kids_by_key.get((seed, n, int(hashes[gi]))) if hashes is not None else None
                pos_file.append(np.full(n - 1, fi, np.int32))
                pos_off.append(off + np.arange(n - 1, dtype=np.int64) * REC)
                pos_stm.append(stm[:-1].astype(np.int8))
                pos_label.append(labels.astype(np.float32))
                pos_game.append(np.full(n - 1, n_games, np.int64))
                if kid is not None:                # record k's children are ply k+1's
                    pos_kfile.append(np.full(n - 1, kid[0], np.int32))
                    pos_koff.append(kid[1][1:])
                    pos_kn.append(kid[2][1:])
                else:
                    pos_kfile.append(np.full(n - 1, -1, np.int32))
                    pos_koff.append(np.zeros(n - 1, np.int64))
                    pos_kn.append(np.zeros(n - 1, np.int32))
                n_games += 1
    if not pos_file:
        raise RuntimeError("no positions found")
    pos_file, pos_off = np.concatenate(pos_file), np.concatenate(pos_off)
    pos_stm, pos_label = np.concatenate(pos_stm), np.concatenate(pos_label)
    pos_game = np.concatenate(pos_game)
    pos_kfile, pos_koff, pos_kn = np.concatenate(pos_kfile), np.concatenate(pos_koff), np.concatenate(pos_kn)
    N = len(pos_label)
    rec_kids = pos_kfile >= 0
    say(f"{n_games} games, {N} positions from {len(tkn_maps)} .tkn file(s) ({len(archives)} "
        f"archive(s), {n_bare} bare) in {time.time() - t0:.0f}s"
        + (f"; {dup_games} duplicate games skipped" if dup_games else ""))
    say(f"{rec_kids.mean():.1%} of positions have recorded children "
        f"(mean {pos_kn[rec_kids].mean() if rec_kids.any() else 0:.0f} each); the rest get "
        f"{ONFLY_CHILDREN} sampled children scored by this model")
    say(f"value labels: mean {pos_label.mean():+.4f} sd {pos_label.std():.4f} "
        f"max |label| {np.abs(pos_label).max():.3f}")

    # ---- the move generator, only if some positions need it -------------------------
    lib = None
    if (~rec_kids).any() or FULL_EVAL_POSITIONS > 0:
        so = os.path.abspath(os.path.join(CACHE_DIR, "libtkykids.so"))
        src = os.path.join(ENGINE_DIR, "tkykids.c")
        if not os.path.exists(so) or (os.path.exists(src) and os.path.getmtime(src) > os.path.getmtime(so)):
            r = subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", so, "tkykids.c", "-lm"],
                               cwd=ENGINE_DIR, capture_output=True, text=True)
            if r.returncode:
                say(f"could not build the move generator, so the {(~rec_kids).mean():.1%} of "
                    f"positions without recorded children train the value head only, and "
                    f"there is no all-legal-moves report:\n{r.stderr[-1500:]}")
        if os.path.exists(so):
            lib = ctypes.CDLL(so)
            P = ctypes.POINTER
            lib.tk_moves.argtypes = [P(ctypes.c_uint16), ctypes.c_int, P(ctypes.c_uint32),
                                     ctypes.c_int, P(ctypes.c_uint32)]
            lib.tk_children.argtypes = [P(ctypes.c_uint16), ctypes.c_int, P(ctypes.c_uint32),
                                        ctypes.c_int, P(ctypes.c_uint16)]
            if lib.tk_init() != 0:
                raise RuntimeError("tk_init failed")
    movebuf = np.zeros(8192, np.uint32)
    forced_out = ctypes.c_uint32()

    # ---- the model ------------------------------------------------------------------
    sys.path.insert(0, os.path.abspath(MODEL_DIR))
    from model_policy import TaikyokuShogiBot as _PolicyBot
    model = _PolicyBot().to(device)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
    if CHECKPOINT_IN and os.path.exists(CHECKPOINT_IN):
        ck = torch.load(CHECKPOINT_IN, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        if LOAD_OPTIMIZER and "optim" in ck:
            opt.load_state_dict(ck["optim"])
        say(f"loaded {CHECKPOINT_IN}")
    else:
        say(f"no {CHECKPOINT_IN}: starting from random weights")

    # ---- batches --------------------------------------------------------------------
    def canonical(board, stm):
        """The side to move as black: white to move is rotated 180 degrees, colours flipped."""
        if stm == 1:
            return board
        b = board[::-1]
        return np.where((b >> 2) != 0, b ^ 2, b)

    def embed(boards):
        return torch.from_numpy(((np.stack(boards).astype(np.int64) >> 1) - 1).reshape(-1, 36, 36))

    def canon_moves(moves, stm):
        o = (moves >> 17) & 0x7FF
        t = (moves >> 6) & 0x7FF
        s = moves >> 28
        if stm == 0:                                   # rotate: squares and lion directions
            o, t = 1295 - o, 1295 - t
            s = np.where((s >= 1) & (s <= 8), (s - 1 + 4) % 8 + 1, s)
        return o.astype(np.int64), t.astype(np.int64), s.astype(np.int64)

    def onfly(board, stm, gen):
        """(moves, child boards) for a few children, to be scored by the model's own value head."""
        n = lib.tk_moves(board.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), int(stm),
                         movebuf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(movebuf),
                         ctypes.byref(forced_out))
        if forced_out.value or n < 2:
            return None                                # forced King capture: no search there
        pick = movebuf[gen.choice(min(n, len(movebuf)), size=min(ONFLY_CHILDREN, n), replace=False)].copy()
        kids = np.zeros((len(pick), 1296), np.uint16)
        lib.tk_children(board.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), int(stm),
                        pick.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(pick),
                        kids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
        return pick, kids

    def build(idx, fixed=False):
        """Boards, value labels, and every child's (batch row, from, to, type) with a target.
        fixed: each position always draws the same on-the-fly children, so reports compare."""
        boards, rows, frm, to, typ, tval = [], [], [], [], [], []
        pending = []                                   # on-the-fly children to score
        for r, i in enumerate(idx):
            mm, off, stm = tkn_maps[pos_file[i]], int(pos_off[i]), int(pos_stm[i])
            board = np.frombuffer(mm, np.uint16, 1296, off)
            boards.append(canonical(board, stm))
            if pos_kfile[i] >= 0:
                n = int(pos_kn[i])
                if n < 2:
                    continue                           # a forced King capture
                kids = np.frombuffer(kids_maps[pos_kfile[i]], KID, n, int(pos_koff[i]))
                vals = kids["value"].astype(np.float32)
                moves = kids["move"]
            elif lib is not None:
                got = onfly(np.ascontiguousarray(board), stm,
                            np.random.default_rng([SEED, int(i)]) if fixed else rng)
                if got is None:
                    continue
                moves, kid_boards = got
                pending.append((len(tval), len(moves), kid_boards, 1 - stm))
                vals = np.zeros(len(moves), np.float32)  # filled in below
            else:
                continue
            o, t, s = canon_moves(moves.astype(np.int64), stm)
            rows.append(np.full(len(moves), r, np.int64))
            frm.append(o); to.append(t); typ.append(s)
            tval.append(vals)
        x = embed(boards).to(device)
        y = torch.from_numpy(pos_label[idx]).unsqueeze(-1).to(device)
        if not rows:
            return x, y, None, None, None
        tvals = np.concatenate(tval)
        if pending:                                    # the model's own value head
            seg = np.concatenate([[0], np.cumsum([len(v) for v in tval])])
            kb = [canonical(b, ks) for (_, _, bs, ks) in pending for b in bs]
            was = model.training
            model.eval()
            with torch.no_grad():
                out = []
                for j in range(0, len(kb), 512):
                    v, _ = model(embed(kb[j:j + 512]).to(device))
                    out.append(v.squeeze(-1).float().cpu().numpy())
            model.train(was)
            out = np.concatenate(out)
            k = 0
            for (seg_i, cnt, _bs, _ks) in pending:
                tvals[seg[seg_i]:seg[seg_i] + cnt] = out[k:k + cnt]
                k += cnt
        moves = torch.from_numpy(np.stack([np.concatenate(rows), np.concatenate(frm),
                                           np.concatenate(to), np.concatenate(typ)], 1)).to(device)
        return x, y, moves, torch.from_numpy(tvals).to(device), torch.from_numpy(np.concatenate(rows)).to(device)

    def segment_softmax_parts(logits, seg_rows, B):
        """Per-position log-softmax of `logits`, positions given by seg_rows (0..B-1)."""
        mx = torch.full((B,), -1e30, device=logits.device).scatter_reduce(0, seg_rows, logits, "amax")
        ex = torch.exp(logits - mx[seg_rows])
        den = torch.zeros(B, device=logits.device).index_add(0, seg_rows, ex)
        return logits - mx[seg_rows] - torch.log(den[seg_rows])

    def policy_loss(policy, tvals, seg_rows, B):
        target_logp = segment_softmax_parts(-tvals.float() / POLICY_TEMPERATURE, seg_rows, B)
        logp = segment_softmax_parts(policy.float(), seg_rows, B)
        ce = -(target_logp.exp() * logp)
        per = torch.zeros(B, device=policy.device).index_add(0, seg_rows, ce)
        has = torch.zeros(B, device=policy.device).index_add(0, seg_rows, torch.ones_like(ce)) > 0
        return per[has].mean(), int(has.sum())

    # ---- held-out games and the reports ---------------------------------------------
    eval_game = rng.random(n_games) < EVAL_GAME_FRACTION
    is_eval = eval_game[pos_game]
    train_idx = np.nonzero(~is_eval)[0]
    eval_pool = np.nonzero(is_eval & ((rec_kids & (pos_kn >= 2)) | (~rec_kids & (lib is not None))))[0]
    eval_idx = rng.choice(eval_pool, size=min(EVAL_POSITIONS, len(eval_pool)), replace=False) \
        if len(eval_pool) else np.zeros(0, np.int64)
    held = np.nonzero(is_eval)[0]
    full_idx = rng.choice(held, size=min(FULL_EVAL_POSITIONS, len(held)), replace=False) \
        if lib is not None and FULL_EVAL_POSITIONS > 0 and len(held) else np.zeros(0, np.int64)
    say(f"training on {len(train_idx)} positions; held out {int(eval_game.sum())} games for a "
        f"report on {len(eval_idx)} positions' scored children"
        + (f" and {len(full_idx)} positions' every legal move" if len(full_idx) else ""))
    QS = (0.05, 0.10, 0.25, 0.50)

    def new_acc():
        z = lambda: np.zeros(len(QS))
        return {"n": 0, "K": 0, "top1": 0, "kl": 0.0, "spread": 0.0, "rec": z(), "rand": z(),
                "mass": z(), "ideal": z(), "lost": z(), "rlost": z()}

    def tally(acc, p, v):
        """One position: policy logits p and the children's values v (lower is better for
        the side choosing among them)."""
        K = len(p)
        order = np.argsort(-p)
        vs = np.sort(v)
        best = vs[0]
        rank = int(np.nonzero(order == int(np.argmin(v)))[0][0])
        t = np.exp(-(v - best) / POLICY_TEMPERATURE)
        t /= t.sum()
        ts = np.sort(t)[::-1]
        acc["n"] += 1
        acc["K"] += K
        acc["top1"] += rank == 0
        acc["spread"] += float(np.median(v) - best)
        steps = np.arange(1, K)
        for qi, q in enumerate(QS):
            k = max(1, int(np.ceil(q * K)))
            acc["rand"][qi] += k / K               # a random k: hit chance, and mass held
            acc["rec"][qi] += rank < k
            acc["mass"][qi] += t[order[:k]].sum()
            acc["ideal"][qi] += ts[:k].sum()
            acc["lost"][qi] += float(v[order[:k]].min() - best)
            # a random k's best is the i-th best overall with chance C(K-i, k-1) / C(K, k)
            ratio = np.clip((K - steps - k + 1) / (K - steps), 0.0, None)
            chance = (k / K) * np.concatenate([[1.0], np.cumprod(ratio)])
            acc["rlost"][qi] += float((chance * vs).sum() - best)
        lp = p - p.max()
        lp = lp - np.log(np.exp(lp).sum())
        acc["kl"] += float((t * (np.log(t + 1e-30) - lp)).sum())

    def show(title, acc):
        n = acc["n"]
        f = lambda key, fmt: "  ".join(fmt.format(x / n) for x in acc[key])
        pct, val = "{:7.0%}", "{:7.4f}"
        say(f"\n  [{title}] {n} positions, {acc['K'] / n:.0f} children each on average   "
            f"top-1 {acc['top1'] / n:.1%}   KL {acc['kl'] / n:.3f}")
        say("    evaluating only the policy's top          " + "  ".join(f"{int(q * 100):>6d}%" for q in QS))
        say("    catches the value head's best child       " + f("rec", pct))
        say("      a random pick of that many would        " + f("rand", pct))
        say("    holds this much of the target softmax     " + f("mass", pct))
        say("      an ideal ordering would hold            " + f("ideal", pct))
        say("      a random pick would hold                " + f("rand", pct))
        say("    value given up against the best child     " + f("lost", val))
        say("      a random pick would give up             " + f("rlost", val))
        say(f"    For scale, the median child is {acc['spread'] / n:.3f} worse than the best, on "
            f"the value head's [-1, 1].\n")
        return {"step": None, "positions": n, "top1": acc["top1"] / n, "kl": acc["kl"] / n,
                **{f"{key}@{q}": acc[key][i] / n for key in ("rec", "rand", "mass", "ideal", "lost", "rlost")
                   for i, q in enumerate(QS)}}

    def report(step):
        if not len(eval_idx):
            return {}
        model.eval()
        acc = new_acc()
        vmse = 0.0; nv = 0; n_rec = 0
        with torch.no_grad():
            for j in range(0, len(eval_idx), BATCH_SIZE):
                idx = eval_idx[j:j + BATCH_SIZE]
                x, y, moves, tvals, seg = build(idx, fixed=True)
                value, policy = model(x, moves) if moves is not None else (model(x)[0], None)
                vmse += float(((value - y) ** 2).sum()); nv += len(idx)
                n_rec += int(rec_kids[idx].sum())
                if policy is None:
                    continue
                pol, tv, sg = policy.float().cpu().numpy(), tvals.float().cpu().numpy(), seg.cpu().numpy()
                for r in np.unique(sg):
                    m = sg == r
                    tally(acc, pol[m], tv[m])
        model.train()
        if not acc["n"]:
            return {}
        say(f"\n  [report, step {step}] value MSE {vmse / nv:.4f} on {nv} held-out positions")
        out = show(f"scored children, step {step}: {n_rec} recorded by self-play, "
                   f"{len(eval_idx) - n_rec} sampled and scored by this model", acc)
        out.update(step=step, value_mse=vmse / nv)
        if len(full_idx) and step % FULL_EVAL_EVERY == 0:
            out["full"] = full_report(step)
        return out

    def full_report(step):
        """Every legal move of a few positions, each child scored by this model's value head."""
        model.eval()
        acc = new_acc()
        t0 = time.time()
        with torch.no_grad():
            for i in full_idx:
                board = np.ascontiguousarray(np.frombuffer(tkn_maps[pos_file[i]], np.uint16, 1296,
                                                           int(pos_off[i])))
                stm = int(pos_stm[i])
                n = lib.tk_moves(board.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), stm,
                                 movebuf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                                 len(movebuf), ctypes.byref(forced_out))
                if forced_out.value or n < 2:
                    continue
                moves = movebuf[:n].copy()
                kids = np.zeros((n, 1296), np.uint16)
                lib.tk_children(board.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), stm,
                                moves.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), n,
                                kids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
                kb = [canonical(b, 1 - stm) for b in kids]
                vals = []
                for j in range(0, n, 512):
                    v, _ = model(embed(kb[j:j + 512]).to(device))
                    vals.append(v.squeeze(-1).float().cpu().numpy())
                o, t, sp = canon_moves(moves.astype(np.int64), stm)
                mv = torch.from_numpy(np.stack([np.zeros(n, np.int64), o, t, sp], 1)).to(device)
                _, pol = model(embed([canonical(board, stm)]).to(device), mv)
                tally(acc, pol.float().cpu().numpy(), np.concatenate(vals))
        model.train()
        if not acc["n"]:
            return {}
        return show(f"every legal move, step {step}: scored by this model "
                    f"({time.time() - t0:.0f}s)", acc)

    def save(path):
        torch.save({"model": model.state_dict(), "optim": opt.state_dict()}, path)

    # ---- training -------------------------------------------------------------------
    history = [report(0)]
    model.train()
    step = acc = 0
    t0 = time.time()
    run_v = run_p = 0.0
    run_n = 0
    done = False
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(train_idx)
        for j in range(0, len(order), BATCH_SIZE):
            idx = order[j:j + BATCH_SIZE]
            x, y, moves, tvals, seg = build(idx)
            if moves is not None:
                value, policy = model(x, moves)
                lp, _np = policy_loss(policy, tvals, seg, len(idx))
            else:
                value, _ = model(x)
                lp = torch.zeros((), device=device)
            lv = F.mse_loss(value, y)
            ((lv + POLICY_WEIGHT * lp) / STEP_SIZE).backward()
            run_v += float(lv); run_p += float(lp); run_n += 1
            acc += 1
            if acc == STEP_SIZE:
                opt.step()
                opt.zero_grad()
                acc = 0
                step += 1
                if step % PRINT_EVERY == 0:
                    el = time.time() - t0
                    left = (len(order) - j) / BATCH_SIZE / STEP_SIZE
                    if MAX_STEPS:
                        left = min(left, MAX_STEPS - step)
                    say(f"epoch {epoch} step {step}  value {run_v / run_n * 1000:.2f}e-3  "
                        f"policy {run_p / run_n:.4f}  ({el / step:.2f}s/step, ETA "
                        f"{left * el / step / 60:.0f} min)")
                    run_v = run_p = 0.0
                    run_n = 0
                if step % EVAL_EVERY == 0:
                    history.append(report(step))
                if step % SAVE_EVERY == 0:
                    save(CHECKPOINT_OUT)
                if MAX_STEPS and step >= MAX_STEPS:
                    done = True
                    break
        if done:
            break
    if not history or not history[-1] or history[-1].get("step") != step:
        history.append(report(step))
    if len(full_idx) and step % FULL_EVAL_EVERY != 0:
        history.append({"step": step, "full": full_report(step)})
    save(CHECKPOINT_OUT)
    say(f"saved {CHECKPOINT_OUT} after {step} optimizer steps ({(time.time() - t0) / 60:.1f} min)")
    return {"model": model, "optimizer": opt, "history": history}


policy_training = _train_policy()
