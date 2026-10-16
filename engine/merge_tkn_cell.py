def merge_tkn(inputs, out="output.tkn.tgz", keep_duplicates=False, header_from=1, level=6,
              remove_inputs=False):
    """Join nnplay .tkn files and archives into one, with their .kids side files.

    A marimo cell: paste it on its own. It defines this one function and imports nothing
    from the project, so any other cell can call ``merge_tkn([...], "out.tkn.tgz")``.

    inputs   .tkn and .kids files, and .tgz/.tar.gz/.tar archives holding either, in any
             mix; games are written in input order
    out      "x.tkn.tgz" writes an archive holding x.tkn (and x.kids, if any input had
             children) -- the shape the training cell reads. "x.tkn" writes the bare file,
             with x.kids beside it.

    As in merge_tkn.py: counts are recomputed by walking the games (a killed run leaves
    0 in its header and maybe half a game, which is dropped); the header's settings come
    from input number ``header_from`` and every field that differs is listed; a game seen
    twice -- same header and moves -- is kept once. A .kids game is keyed by (seed,
    plies, hash of its moves) exactly as training matches it, so dropping a duplicate drops
    its children with it. Returns a dict of the counts.
    """
    import hashlib
    import os
    import struct
    import tarfile
    import time

    MAGIC, KMAGIC = b"TKYNNSP\x00", b"TKYKIDS\x00"
    FILE_HDR, GAME_HDR, REC = 128, 16, 1296 * 2 + 16    # the header is 256 from version 7
    OUT_HDR = 256
    KFILE_HDR, KGAME_HDR = 32, 24
    MOVE_OFF = 1296 * 2

    def fields_of(raw):
        u32 = lambda o: struct.unpack_from("<I", raw, o)[0]
        f32 = lambda o: struct.unpack_from("<f", raw, o)[0]
        ver, rules = u32(8), u32(80)
        return {"version": ver, "value_fp": u32(24), "mat_shift": u32(28),
                "seed_base": struct.unpack_from("<Q", raw, 32)[0],
                "temperature": struct.unpack_from("<d", raw, 56)[0],
                "max_plies": u32(64), "rep_limit": u32(68), "no_progress_limit": u32(72),
                "stalemate_loses": u32(76), "regicide": rules & 1,
                "king_safety": (rules >> 1) & 1 if ver >= 3 else 0,
                "cap_by_material": (rules >> 2) & 1 if ver >= 4 else 0,
                "resign_lead": (rules >> 8) & 0xFFFF if ver >= 4 else 0,
                "resign_plies": rules >> 24 if ver >= 4 else 0,
                "opening_plies": struct.unpack_from("<H", raw, 92)[0] if ver >= 2 else 0,
                "opening_temperature": struct.unpack_from("<H", raw, 94)[0] / 1000 if ver >= 2 else None,
                "sub_batch": u32(96), "fp16": u32(100) & 1,
                "policy_top": (u32(100) >> 16) / 1000 if ver >= 6 else None,
                "policy_explore": ((u32(100) >> 8) & 0xFF) / 1000 if ver >= 6 else None,
                "drop_capped": (u32(100) >> 1) & 1, "endgames": (u32(100) >> 2) & 1,
                "sims": u32(128) if ver >= 7 else 0, "leaves": u32(132) if ver >= 7 else 0,
                "cpuct": round(f32(136), 4) if ver >= 7 else 0.0,
                "noise_frac": round(f32(140), 4) if ver >= 7 else 0.0,
                "noise_alpha": round(f32(144), 4) if ver >= 7 else 0.0,
                "games_par": u32(148) if ver >= 7 else 1, "opp_sims": u32(152) if ver >= 7 else 0,
                "contempt": round(f32(160), 4) if ver >= 7 else 0.0,
                "contempt_flat": (u32(156) >> 3) & 1 if ver >= 7 else 0,
                "soft_scale": round(f32(164), 4) if len(raw) >= 256 else 0.0,
                "model": raw[104:128].split(b"\x00")[0].decode(errors="replace")}

    def soft_hard(gh):
        """A game from a file without soft labels: its label (bytes 6..7, 1/10000) is the result."""
        return gh[:6] + struct.pack("<h", struct.unpack_from("<b", gh, 4)[0] * 10000) + gh[8:]


    def soft_header(hdr, headers):
        """The first soft-labelled input's scale and pawns, so a mixed merge keeps them; every
        other input's games were given their result as the label on the way in."""
        for raw in headers:
            if len(raw) >= 256 and struct.unpack_from("<f", raw, 164)[0] > 0:
                hdr[164:172] = raw[164:172]
                return

    def read_exact(f, n):
        parts, got = [], 0
        while got < n:
            b = f.read(n - got)
            if not b:
                break
            parts.append(b)
            got += len(b)
        return b"".join(parts)

    def streams(path):
        """(name, kind, stream) for every .tkn and .kids in an archive, or the file itself."""
        if path.endswith((".tgz", ".tar.gz", ".tar")):
            with tarfile.open(path, "r|*") as tf:
                for m in tf:
                    if m.isfile() and m.name.endswith((".tkn", ".kids")):
                        yield f"{path}:{m.name}", m.name.rsplit(".", 1)[1], tf.extractfile(m)
        else:
            with open(path, "rb") as f:
                yield path, path.rsplit(".", 1)[1], f

    if isinstance(inputs, str):
        inputs = [inputs]
    missing = [p for p in inputs if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("not found: " + ", ".join(missing))
    archive = out.endswith((".tgz", ".tar.gz", ".tar"))
    stem = os.path.basename(out)
    for ext in (".tgz", ".tar.gz", ".tar", ".tkn"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    if stem.endswith(".tkn"):
        stem = stem[:-4]
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    tkn_path = os.path.join(out_dir, stem + ".tkn" + (".partial" if archive else ""))
    kids_path = os.path.join(out_dir, stem + ".kids" + (".partial" if archive else ""))
    cleanup = [tkn_path, kids_path] + ([out + ".partial"] if archive else [])
    for p in inputs:
        if os.path.abspath(p) in (os.path.abspath(out), tkn_path, kids_path):
            raise ValueError(f"{p} is also an output")

    t0 = time.time()
    headers, differs = [], {}
    seen, seen_k = set(), set()
    tot = {"games": 0, "positions": 0, "duplicates": 0, "duplicate_positions": 0,
           "truncated": 0, "kids_games": 0, "kids_duplicates": 0, "kids_truncated": 0}
    kids_hdr = None
    try:
        with open(tkn_path, "wb") as w, open(kids_path, "wb") as kw:
            w.write(b"\x00" * OUT_HDR)                  # patched once the totals are known
            kw.write(b"\x00" * KFILE_HDR)
            for path in inputs:
                for name, kind, f in streams(path):
                    if kind == "tkn":
                        raw = read_exact(f, FILE_HDR)
                        if len(raw) != FILE_HDR or raw[:8] != MAGIC:
                            raise ValueError(f"{name}: not an nnplay .tkn file")
                        hs = struct.unpack_from("<III", raw, 12)
                        if hs[0] not in (FILE_HDR, OUT_HDR) or hs[1:] != (GAME_HDR, REC):
                            raise ValueError(f"{name}: unexpected header/game/ply sizes")
                        raw += read_exact(f, hs[0] - FILE_HDR)
                        fl = fields_of(raw)
                        if headers and (fl["mat_shift"], fl["value_fp"]) != \
                                (fields_of(headers[0])["mat_shift"], fields_of(headers[0])["value_fp"]):
                            raise ValueError(f"{name}: a different material scale from the first "
                                             f"input; the stored material would be misread")
                        headers.append(raw)
                        for k, v in fl.items():
                            differs.setdefault(k, {}).setdefault(v, []).append(len(headers))
                        g = n = d = 0
                        while True:
                            gh = read_exact(f, GAME_HDR)
                            if not gh:
                                break
                            np_ = struct.unpack_from("<I", gh, 0)[0] if len(gh) == GAME_HDR else 0
                            body = read_exact(f, np_ * REC)
                            if len(gh) < GAME_HDR or len(body) < np_ * REC:
                                tot["truncated"] += 1           # the run died mid-game
                                break
                            if not fl["soft_scale"]:
                                gh = soft_hard(gh)
                            if not keep_duplicates:
                                h = hashlib.blake2b(gh, digest_size=16)
                                h.update(b"".join(body[i + MOVE_OFF:i + MOVE_OFF + 4]
                                                  for i in range(0, len(body), REC)))
                                if h.digest() in seen:
                                    d += 1
                                    tot["duplicate_positions"] += np_
                                    continue
                                seen.add(h.digest())
                            w.write(gh)
                            w.write(body)
                            g += 1
                            n += np_
                        tot["games"] += g
                        tot["positions"] += n
                        tot["duplicates"] += d
                        print(f"  [{len(headers)}] {name}: {g} games, {n} positions"
                              + (f", {d} duplicates dropped" if d else "")
                              + f"  [{fl['model']}, T={fl['temperature']:g}"
                              + (", endgames" if fl["endgames"] else "") + "]", flush=True)
                    else:
                        raw = read_exact(f, KFILE_HDR)
                        if len(raw) != KFILE_HDR or raw[:8] != KMAGIC:
                            raise ValueError(f"{name}: not a .kids file")
                        if kids_hdr and raw[8:20] != kids_hdr[8:20]:
                            raise ValueError(f"{name}: .kids version {struct.unpack_from('<I', raw, 8)[0]}"
                                             f", but an earlier input is version "
                                             f"{struct.unpack_from('<I', kids_hdr, 8)[0]}: one "
                                             f"file cannot hold both")
                        kids_hdr = kids_hdr or raw
                        child = struct.unpack_from("<I", raw, 16)[0]   # 6 in version 1, 8 in 2
                        g = d = 0
                        while True:
                            gh = read_exact(f, KGAME_HDR)
                            if not gh:
                                break
                            if len(gh) < KGAME_HDR:
                                tot["kids_truncated"] += 1
                                break
                            np_ = struct.unpack_from("<I", gh, 0)[0]
                            parts, ok = [], True
                            for _ in range(np_):
                                c = read_exact(f, 2)
                                k = child * struct.unpack_from("<H", c, 0)[0] if len(c) == 2 else 0
                                kids = read_exact(f, k)
                                if len(c) < 2 or len(kids) < k:
                                    ok = False
                                    break
                                parts += [c, kids]
                            if not ok:
                                tot["kids_truncated"] += 1
                                break
                            key = (np_,) + struct.unpack_from("<QQ", gh, 8)
                            if not keep_duplicates and key in seen_k:
                                d += 1
                                continue
                            seen_k.add(key)
                            kw.write(gh)
                            kw.write(b"".join(parts))
                            g += 1
                        tot["kids_games"] += g
                        tot["kids_duplicates"] += d
                        print(f"  {name}: {g} games of children"
                              + (f", {d} duplicates dropped" if d else ""), flush=True)

            if not headers:
                raise ValueError("no .tkn data in the inputs")
            if not 1 <= header_from <= len(headers):
                raise ValueError(f"header_from={header_from}: there are {len(headers)} .tkn inputs")
            # always 256 bytes: an older header is padded, and readers gate the new fields
            # on the version, so they read the padding as nothing
            hdr = bytearray(headers[header_from - 1].ljust(OUT_HDR, b"\x00"))
            struct.pack_into("<I", hdr, 12, OUT_HDR)
            soft_header(hdr, headers)
            struct.pack_into("<QQ", hdr, 40, tot["games"], tot["positions"])
            w.seek(0)
            w.write(bytes(hdr))
            if kids_hdr:
                kh = bytearray(kids_hdr)
                struct.pack_into("<Q", kh, 24, tot["kids_games"])
                kw.seek(0)
                kw.write(bytes(kh))

        print(f"\n{tot['games']} games, {tot['positions']} positions in {time.time() - t0:.0f}s"
              + (f"; dropped {tot['duplicates']} duplicate games ({tot['duplicate_positions']} "
                 f"positions)" if tot["duplicates"] else "")
              + (f"; dropped {tot['truncated']} truncated game(s)" if tot["truncated"] else ""))
        if kids_hdr:
            print(f"children for {tot['kids_games']} games"
                  + (f"; dropped {tot['kids_duplicates']} duplicates" if tot["kids_duplicates"] else "")
                  + (f"; dropped {tot['kids_truncated']} truncated" if tot["kids_truncated"] else ""))
            if tot["kids_games"] != tot["games"]:
                print(f"  {tot['games'] - tot['kids_games']:+d} games have no children recorded: "
                      f"training scores those itself")
        mixed = {k: v for k, v in differs.items() if len(v) > 1 and k != "seed_base"}
        if mixed:
            print(f"\nThe inputs were not all generated the same way; the header carries input "
                  f"[{header_from}]'s settings. Differences:")
            for k, vals in mixed.items():
                print(f"    {k:20s} " + "   ".join(f"{v} [{','.join(map(str, ix))}]"
                                                    for v, ix in vals.items()))

        if not kids_hdr:
            os.remove(kids_path)
        if archive:
            t1 = time.time()
            with tarfile.open(out + ".partial", "w:gz", compresslevel=level) as tf:
                tf.add(tkn_path, arcname=stem + ".tkn")
                if kids_hdr:
                    tf.add(kids_path, arcname=stem + ".kids")
            os.replace(out + ".partial", out)
            for p in (tkn_path, kids_path):
                if os.path.exists(p):
                    os.remove(p)
            print(f"{out}: {os.path.getsize(out) / 1e6:.1f} MB, compressed in "
                  f"{time.time() - t1:.0f}s")
        else:
            print(f"wrote {tkn_path}" + (f" and {kids_path}" if kids_hdr else ""))
    except BaseException:
        for p in cleanup:                   # never leave a multi-GB half-file behind
            if os.path.exists(p):
                os.remove(p)
        raise
    if remove_inputs:
        for p in inputs:
            os.remove(p)
    return tot
