"""Join any number of nnplay self-play archives into one.

    python3 merge_tkn.py selfplay.tkn.tgz "selfplay.tkn (2).tgz" more.tkn.tgz
    python3 merge_tkn.py -o all.tkn.tgz batch_*.tkn.tgz

Writes ``output.tkn.tgz`` holding a single ``output.tkn`` -- the same shape as the archives
nnplay's output is shipped in, so ``tar xf`` and ``unpack_nn("output.tkn")`` work unchanged.
Inputs may be .tgz / .tar.gz / .tar archives (every .tkn inside is taken) or bare .tkn
files, in any mix. Games are written in input order.

A merged file has one header, so three things are handled rather than copied blindly:

  * **Counts are recomputed.** nnplay patches the game and position counts into the header
    only when it exits cleanly; a session that was killed leaves them at 0 and can leave
    half of its last game on disk. Games are counted by walking them, and a truncated last
    game is dropped and reported.
  * **Settings come from one input**, the first unless --header-from says otherwise.
    Model, temperature, rules and the rest are per-run, and the output can only carry one
    set. Every field that differs between the inputs is listed so nothing is mixed
    silently. The material scale (mat_shift) must
    match: the stored material values are meaningless under a different one.
  * **Repeated games are dropped.** Every session starts from the same default seed, so two
    sessions with the same model and settings replay the same games, which would double the
    weight of their positions in training. A game is identified by its header (seed,
    length, result) and its full move sequence -- not its bytes, because the stored network
    values of a replayed game can differ by ~1e-4 on a ply or two from GPU rounding, while
    every board, move and result is the same. Games that share a seed but play different
    moves are kept and counted. --keep-duplicates keeps everything.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
import sys
import tarfile
import time

MAGIC = b"TKYNNSP\x00"
MOVE_OFF = 1296 * 2                      # the u32 move follows the board in each ply record
FILE_HDR, GAME_HDR = 128, 16
REC = 1296 * 2 + 16
_CLEANUP: list[str] = []          # half-written outputs to remove if the merge fails


def header_fields(raw: bytes) -> dict:
    u32 = lambda o: struct.unpack_from("<I", raw, o)[0]
    ver = u32(8)
    rules = u32(80)
    f = {"version": ver, "value_fp": u32(24), "mat_shift": u32(28),
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
         "model": raw[104:128].split(b"\x00")[0].decode(errors="replace")}
    return f


def check_header(raw: bytes, name: str) -> None:
    if len(raw) != FILE_HDR or raw[:8] != MAGIC:
        raise SystemExit(f"{name}: not an nnplay .tkn file (magic {raw[:8]!r})")
    sizes = struct.unpack_from("<III", raw, 12)
    if sizes != (FILE_HDR, GAME_HDR, REC):
        raise SystemExit(f"{name}: header/game/ply sizes {sizes}, expected "
                         f"{(FILE_HDR, GAME_HDR, REC)}")


def streams(path: str):
    """Yield (name, binary stream) for every .tkn in an archive, or the file itself."""
    if path.endswith((".tgz", ".tar.gz", ".tar")):
        with tarfile.open(path, "r|*") as tf:            # streaming: no seeking needed
            found = False
            for m in tf:
                if m.isfile() and m.name.endswith(".tkn"):
                    found = True
                    yield f"{path}:{m.name}", tf.extractfile(m)
                elif m.isfile() and m.name.endswith(".kids"):
                    print(f"  warning: {path} holds {m.name}, the search's scored children. "
                          f"Those are not merged; train the policy on the original archive, "
                          f"which the training cell reads directly", file=sys.stderr)
            if not found:
                print(f"  warning: {path} holds no .tkn file", file=sys.stderr)
    else:
        with open(path, "rb") as f:
            yield path, f


def read_exact(f, n: int) -> bytes:
    """tarfile streams can return short reads; keep reading until n bytes or EOF."""
    parts, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            break
        parts.append(b)
        got += len(b)
    return b"".join(parts)


class Progress:
    """A file wrapper that prints how far a long read has got, for the compression step."""

    def __init__(self, f, total: int, label: str):
        self.f, self.total, self.label = f, total, label
        self.done, self.next, self.t0 = 0, 0.0, time.time()

    def read(self, n=-1):
        b = self.f.read(n)
        self.done += len(b)
        frac = self.done / self.total if self.total else 1.0
        if frac >= self.next or not b:
            el = time.time() - self.t0
            eta = el / frac * (1 - frac) if frac > 0 else 0
            tty = sys.stderr.isatty()          # redraw in place only where that works
            print(f"{chr(13) if tty else ''}  {self.label}: {frac:6.1%}  {self.done / 1e9:6.2f} "
                  f"of {self.total / 1e9:.2f} GB  ({el:.0f}s, ETA {eta:.0f}s)",
                  end="" if tty else "\n", file=sys.stderr, flush=True)
            self.next = frac + (0.02 if tty else 0.10)
        return b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", help=".tgz/.tar.gz/.tar archives or .tkn files")
    ap.add_argument("-o", "--out", default="output.tkn.tgz",
                    help="output archive; the .tkn inside is named after it (default "
                         "output.tkn.tgz -> output.tkn)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="keep repeated games (same seed and moves) instead of dropping them")
    ap.add_argument("--keep-tkn", action="store_true",
                    help="also leave the uncompressed .tkn next to the archive")
    ap.add_argument("--level", type=int, default=6, help="gzip level, 1 fast .. 9 small")
    ap.add_argument("--header-from", type=int, default=1, metavar="N",
                    help="take the output header's settings from the Nth .tkn read, "
                         "counting from 1 in the order listed (default 1)")
    a = ap.parse_args()

    out = a.out
    member = os.path.basename(out)
    for ext in (".tgz", ".tar.gz", ".tar"):
        if member.endswith(ext):
            member = member[: -len(ext)]
            break
    if not member.endswith(".tkn"):
        member += ".tkn"
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    tmp_tkn = os.path.join(out_dir, member + ".partial")
    part = out + ".partial"
    _CLEANUP.extend([tmp_tkn, part])
    missing = [p for p in a.inputs if not os.path.isfile(p)]
    if missing:
        raise SystemExit("not found: " + ", ".join(missing))
    for p in a.inputs:
        if os.path.abspath(p) == os.path.abspath(out):
            raise SystemExit(f"{p} is also the output")

    first_raw, first_fields = None, None
    headers = []                          # every input's raw header, in order read
    differs = {}                          # field -> {value: [input numbers]}
    seen_games, seen_seeds = set(), set()
    tot_games = tot_plies = dups = dup_plies = seed_shared = truncated = 0
    t0 = time.time()

    with open(tmp_tkn, "wb") as w:
        w.write(b"\x00" * FILE_HDR)                        # patched once the totals are known
        for path in a.inputs:
            for name, f in streams(path):
                raw = read_exact(f, FILE_HDR)
                check_header(raw, name)
                fields = header_fields(raw)
                headers.append(raw)
                idx = len(headers)
                if first_raw is None:
                    first_raw, first_fields, first_name = raw, fields, name
                elif fields["mat_shift"] != first_fields["mat_shift"] \
                        or fields["value_fp"] != first_fields["value_fp"]:
                    raise SystemExit(
                        f"{name}: material scale mat_shift={fields['mat_shift']} value_fp="
                        f"{fields['value_fp']}, but {first_name} has mat_shift="
                        f"{first_fields['mat_shift']} value_fp={first_fields['value_fp']}. "
                        f"The stored material would be misread; these cannot share a file.")
                for k, v in fields.items():
                    differs.setdefault(k, {}).setdefault(v, []).append(idx)

                g = p_ = d = s = 0
                while True:
                    gh = read_exact(f, GAME_HDR)
                    if not gh:
                        break
                    if len(gh) < GAME_HDR:
                        truncated += 1
                        break
                    n_plies = struct.unpack_from("<I", gh, 0)[0]
                    body = read_exact(f, n_plies * REC)
                    if len(body) < n_plies * REC:
                        truncated += 1               # the session died mid-game
                        break
                    seed = struct.unpack_from("<Q", gh, 8)[0]
                    if not a.keep_duplicates:
                        h = hashlib.blake2b(gh, digest_size=16)
                        # the moves alone, so GPU noise in the stored values cannot make one
                        # game look like two; seed + moves fix every board and the result
                        h.update(b"".join(body[i + MOVE_OFF:i + MOVE_OFF + 4]
                                          for i in range(0, len(body), REC)))
                        key = h.digest()
                        if key in seen_games:
                            d += 1
                            dup_plies += n_plies
                            continue
                        seen_games.add(key)
                    if seed in seen_seeds:
                        s += 1
                    seen_seeds.add(seed)
                    w.write(gh)
                    w.write(body)
                    g += 1
                    p_ += n_plies
                dups += d
                seed_shared += s
                tot_games += g
                tot_plies += p_
                hdr_games = struct.unpack_from("<Q", raw, 40)[0]
                note = ""
                if hdr_games != g + d:
                    note = (f"  (its header says {hdr_games} games: "
                            + ("the run did not exit cleanly)" if hdr_games == 0 else "mismatch)"))
                print(f"  [{idx}] {name}: {g} games, {p_} positions"
                      + (f", {d} duplicate games dropped" if d else "")
                      + f"  [{fields['model']}, T={fields['temperature']:g}]{note}",
                      file=sys.stderr, flush=True)

        if first_raw is None:
            os.remove(tmp_tkn)
            raise SystemExit("no .tkn data in the inputs")
        if not 1 <= a.header_from <= len(headers):
            raise SystemExit(f"--header-from {a.header_from}: there are {len(headers)} inputs")
        hdr = bytearray(headers[a.header_from - 1])
        struct.pack_into("<Q", hdr, 40, tot_games)
        struct.pack_into("<Q", hdr, 48, tot_plies)
        w.seek(0)
        w.write(bytes(hdr))

    print(f"\n{tot_games} games, {tot_plies} positions in {time.time() - t0:.0f}s"
          + (f"; dropped {dups} duplicate games ({dup_plies} positions)" if dups else "")
          + (f"; dropped {truncated} truncated game(s)" if truncated else ""),
          file=sys.stderr)
    if seed_shared:
        print(f"  {seed_shared} kept games share a seed with an earlier one but play different "
              f"moves, so they are genuinely different games",
              file=sys.stderr)

    mixed = {k: v for k, v in differs.items() if len(v) > 1}
    if mixed:
        print(f"\n  The inputs were not all generated the same way. The output header carries "
              f"input [{a.header_from}]'s\n  settings (--header-from picks another); the games "
              f"themselves are unaffected. Differences:", file=sys.stderr)
        for k, vals in mixed.items():
            cells = "   ".join(f"{v} [{','.join(map(str, ix))}]" for v, ix in vals.items())
            print(f"    {k:20s} {cells}", file=sys.stderr)

    size = os.path.getsize(tmp_tkn)
    t1 = time.time()
    with tarfile.open(part, "w:gz", compresslevel=a.level) as tf, open(tmp_tkn, "rb") as src:
        ti = tarfile.TarInfo(member)
        ti.size = size
        ti.mtime = int(time.time())
        ti.mode = 0o644
        tf.addfile(ti, Progress(src, size, "compressing"))
    os.replace(part, out)
    print(f"\n  {out}: {os.path.getsize(out) / 1e6:.1f} MB, holds {member} "
          f"({size / 1e9:.2f} GB), compressed in {time.time() - t1:.0f}s", file=sys.stderr)
    if a.keep_tkn:
        os.replace(tmp_tkn, os.path.join(out_dir, member))
        print(f"  kept {os.path.join(out_dir, member)}", file=sys.stderr)
    else:
        os.remove(tmp_tkn)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # a failed or interrupted merge must not leave a multi-GB half-file behind
        for p in _CLEANUP:
            if os.path.exists(p):
                os.remove(p)
        raise
