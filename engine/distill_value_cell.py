def _distill_value():
    # ================================= settings =================================
    DATA_DIR = "."                     # every *.tkn.tgz here supplies positions
    CACHE_DIR = "tkn_cache"            # archives are unpacked here once, then memory-mapped
    MODEL_DIR = "."                    # where model.py (old) and model_policy.py (new) live
    TEACHER = "checkpoint.pt"          # the old model's checkpoint
    CHECKPOINT_OUT = "checkpoint_policy.pt"
    RESUME = False                     # continue from CHECKPOINT_OUT instead of warm-starting
    BATCH_SIZE = 64
    STEP_SIZE = 256 // 64              # batches accumulated per optimizer step
    LR, WEIGHT_DECAY = 3e-4, 0.01
    EPOCHS = 1
    MAX_STEPS = None                   # optimizer steps; None = the whole epoch(s)
    EVAL_GAME_FRACTION = 0.02
    EVAL_POSITIONS = 4000
    EVAL_EVERY = 500
    SAVE_EVERY = 1000
    PRINT_EVERY = 50
    SEED = 0
    # ============================================================================
    #
    # Distils the old network's value head into the new policy network's value head: the
    # new model is trained to output what the old one outputs, on every position in the
    # self-play archives.
    #
    # Warm start: every old weight whose name and shape match is copied -- the embedding,
    # both convolutions (the old conv_stem's layers 0,1 and 3,4 are the new conv_norm and
    # conv_down), the positional table, attention and layer norms, lin_shared, lin_value.
    # The transformer's feed-forward layers are 2048 wide in the old model and 512 in the
    # new, so they start fresh, with the second linear zeroed: in a pre-norm layer that
    # makes the feed-forward branch add nothing at first, so the student starts as the
    # teacher without its feed-forward blocks rather than as noise. The policy layers are
    # untouched; the training cell trains them.

    import glob
    import os
    import struct
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    def say(*a):
        print(*a, flush=True)

    # ---- unpack every archive once --------------------------------------------------
    archives = sorted(glob.glob(os.path.join(DATA_DIR, "*.tkn.tgz")))
    if not archives:
        raise RuntimeError(f"no *.tkn.tgz in {os.path.abspath(DATA_DIR)}")
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

    # ---- index every position -------------------------------------------------------
    maps, pos_file, pos_off, pos_stm, pos_game = [], [], [], [], []
    n_games = 0
    for d in dirs:
        for tp in sorted(glob.glob(os.path.join(d, "*.tkn"))):
            mm = np.memmap(tp, dtype=np.uint8, mode="r")
            if bytes(mm[:8]) != b"TKYNNSP\0":
                continue
            fi = len(maps)
            maps.append(mm)
            p = 128
            while p + 16 <= len(mm):
                n = struct.unpack_from("<I", mm, p)[0]
                if p + 16 + n * REC > len(mm):
                    break                         # a killed run's half-written game
                if n:
                    side = np.ndarray((n,), DT, buffer=mm, offset=p + 16)["side"]
                    pos_file.append(np.full(n, fi, np.int32))
                    pos_off.append(p + 16 + np.arange(n, dtype=np.int64) * REC)
                    pos_stm.append((1 - side.astype(np.int64)).astype(np.int8))
                    pos_game.append(np.full(n, n_games, np.int64))
                    n_games += 1
                p += 16 + n * REC
    if not pos_file:
        raise RuntimeError("no positions found")
    pos_file, pos_off = np.concatenate(pos_file), np.concatenate(pos_off)
    pos_stm, pos_game = np.concatenate(pos_stm), np.concatenate(pos_game)
    eval_game = rng.random(n_games) < EVAL_GAME_FRACTION
    is_eval = eval_game[pos_game]
    train_idx = np.nonzero(~is_eval)[0]
    pool = np.nonzero(is_eval)[0]
    eval_idx = rng.choice(pool, size=min(EVAL_POSITIONS, len(pool)), replace=False) if len(pool) else pool
    say(f"{n_games} games, {len(pos_file)} positions; {len(train_idx)} for training, "
        f"{len(eval_idx)} held out")

    def batch(idx):
        boards = []
        for i in idx:
            b = np.frombuffer(maps[pos_file[i]], np.uint16, 1296, int(pos_off[i]))
            if pos_stm[i] == 0:                    # white to move: rotate, flip colours
                b = b[::-1]
                b = np.where((b >> 2) != 0, b ^ 2, b)
            boards.append(b)
        return torch.from_numpy(((np.stack(boards).astype(np.int64) >> 1) - 1).reshape(-1, 36, 36)).to(device)

    # ---- teacher and student --------------------------------------------------------
    sys.path.insert(0, os.path.abspath(MODEL_DIR))
    from model import TaikyokuShogiBot as _OldBot
    from model_policy import TaikyokuShogiBot as _NewBot
    teacher = _OldBot().to(device)
    tk = torch.load(TEACHER, map_location=device, weights_only=True)
    teacher.load_state_dict(tk["model"] if "model" in tk else tk)
    teacher.eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)

    student = _NewBot().to(device)
    opt = torch.optim.AdamW(student.parameters(), LR, weight_decay=WEIGHT_DECAY)
    if RESUME and os.path.exists(CHECKPOINT_OUT):
        ck = torch.load(CHECKPOINT_OUT, map_location=device, weights_only=True)
        student.load_state_dict(ck["model"])
        if "optim" in ck:
            opt.load_state_dict(ck["optim"])
        say(f"resumed from {CHECKPOINT_OUT}")
    else:
        rename = {"conv_stem.0.": "conv_norm.0.", "conv_stem.1.": "conv_norm.1.",
                  "conv_stem.3.": "conv_down.0.", "conv_stem.4.": "conv_down.1."}
        old = teacher.state_dict()
        new = student.state_dict()
        copied, skipped = [], []
        for k, v in old.items():
            nk = k
            for a, b in rename.items():
                if k.startswith(a):
                    nk = b + k[len(a):]
            if nk in new and new[nk].shape == v.shape:
                new[nk] = v.clone()
                copied.append(nk)
            else:
                skipped.append(k)
        for name in new:
            if ".linear2." in name:                # fresh feed-forward branch adds nothing at first
                new[name] = torch.zeros_like(new[name])
        student.load_state_dict(new)
        fresh = [k for k in new if k not in copied]
        say(f"warm start: copied {len(copied)} tensors from {TEACHER}; not copied from the old "
            f"model: {', '.join(sorted({k.rsplit('.', 1)[0] for k in skipped}))}")
        say(f"  new and untrained here: {', '.join(sorted({k.rsplit('.', 1)[0] for k in fresh}))}")

    def report(step):
        student.eval()
        s_all, t_all = [], []
        with torch.no_grad():
            for j in range(0, len(eval_idx), 256):
                x = batch(eval_idx[j:j + 256])
                s_all.append(student(x)[0].float().squeeze(-1).cpu().numpy())
                t_all.append(teacher(x)[0].float().squeeze(-1).cpu().numpy())
        student.train()
        s, t = np.concatenate(s_all), np.concatenate(t_all)
        d = np.abs(s - t)
        out = {"step": step, "mse": float((d ** 2).mean()),
               "corr": float(np.corrcoef(s, t)[0, 1]) if s.std() > 0 and t.std() > 0 else float("nan"),
               "sign": float((np.sign(s) == np.sign(t)).mean()), "p95": float(np.percentile(d, 95))}
        say(f"  [held out, step {step}] MSE {out['mse']:.5f}  corr {out['corr']:.4f}  "
            f"same sign {out['sign']:.1%}  |student - teacher| p95 {out['p95']:.4f}  "
            f"(teacher sd {t.std():.4f})")
        return out

    def save():
        torch.save({"model": student.state_dict(), "optim": opt.state_dict()}, CHECKPOINT_OUT)

    # ---- distillation ---------------------------------------------------------------
    history = [report(0)]
    student.train()
    step = acc = 0
    run, run_n = 0.0, 0
    t0 = time.time()
    done = False
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(train_idx)
        for j in range(0, len(order), BATCH_SIZE):
            x = batch(order[j:j + BATCH_SIZE])
            with torch.no_grad():
                target = teacher(x)[0]
            loss = F.mse_loss(student(x)[0], target)
            (loss / STEP_SIZE).backward()
            run += float(loss); run_n += 1
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
                    say(f"epoch {epoch} step {step}  distillation MSE {run / run_n:.5f}  "
                        f"({el / step:.2f}s/step, ETA {left * el / step / 60:.0f} min)")
                    run, run_n = 0.0, 0
                if step % EVAL_EVERY == 0:
                    history.append(report(step))
                if step % SAVE_EVERY == 0:
                    save()
                if MAX_STEPS and step >= MAX_STEPS:
                    done = True
                    break
        if done:
            break
    if history[-1].get("step") != step:
        history.append(report(step))
    save()
    say(f"saved {CHECKPOINT_OUT} after {step} optimizer steps ({(time.time() - t0) / 60:.1f} min)")
    return {"student": student, "optimizer": opt, "history": history}


distilled = _distill_value()
