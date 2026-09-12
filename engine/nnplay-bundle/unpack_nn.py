"""Reader for the .tkn self-play format that nnplay writes.

One record per ply, holding the board *after* the move, the network's value for the leaf
it chose, and who moved; one result per game. Little-endian throughout, fixed strides, so
a trainer can mmap a shard and index straight to a ply.

The board is stored absolute -- black's frame, black's King at (0,17) -- exactly as
unpack.py stores it, whoever moved. ``canonical=True`` (the default) rotates it into the
frame of the side to move in that position, which is the opponent of the mover, and is the
frame the network was given. So after canonicalisation ``info.value`` is the value of the
board you are holding, with no sign to chase.

    from unpack_nn import unpack_nn
    for board, info in unpack_nn("selfplay.tkn"):
        x = embedding_index(board)     # exactly what nnplay fed the network
        v = info.value                 # the network's value of this board
        z = info.result_stm            # +1 if the side to move went on to win
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
from dataclasses import dataclass, asdict

import numpy as np

from unpack import (VALUE_FP, TERM_NAMES, canonical_board, rot180_move, square_name,
                    unpack_move)

NSQ = 1296

MAGIC = b"TKYNNSP\x00"
FILE_HDR = 128
GAME_HDR = 16
BOARD_BYTES = NSQ * 2
REC = BOARD_BYTES + 16

PF_CAPTURE, PF_PROMOTION, PF_FORCED_KING, PF_TERMINAL = 1, 2, 4, 8

REC_DTYPE = np.dtype([("board", "<u2", (NSQ,)), ("move", "<u4"), ("value", "<f4"),
                      ("n_legal", "<u2"), ("material", "<i2"), ("side", "u1"),
                      ("flags", "u1"), ("pad", "<u2")])
assert REC_DTYPE.itemsize == REC, REC_DTYPE.itemsize


@dataclass
class Header:
    version: int
    seed_base: int
    games: int
    plies: int
    temperature: float
    max_plies: int
    rep_limit: int
    no_progress_limit: int
    stalemate_loses: int
    regicide: int
    board_size: int
    piece_types: int
    sub_batch: int
    fp16: int
    model: str            # "<graph>@<hash>" -- the hash identifies the weights, not the name
    mat_shift: int
    value_fp: int


@dataclass
class NNInfo:
    """Everything stored with one ply, plus the game it came from."""
    path: str
    game_index: int
    seed: int
    ply: int
    plies_total: int
    side: int                # the mover, in this frame: 1 black, 0 white (always 0 once canonical)
    side_black: int          # always absolute: 1 if black made this move
    move: dict               # special, origin, target of the move that produced this board
    value: float             # the network's value of this board, in its side-to-move frame
    n_legal: int             # legal moves the mover chose from
    material: float          # pawns, black minus white (side to move minus opponent if canonical)
    result: int              # 1 the side to move won, -1 lost, 0 draw (absolute: 1 = black)
    result_black: int        # always absolute: 1 black won, -1 white won, 0 draw
    termination: str
    forced_king_capture: bool
    was_capture: bool
    was_promotion: bool
    canonical: bool

    @property
    def side_to_move(self) -> int:
        """Who is to move in this position: the mover's opponent, and 1 once canonical."""
        return 1 - self.side

    @property
    def result_stm(self) -> int:
        """+1 if the side to move in this position went on to win."""
        return self.result if self.side_to_move == 1 else -self.result

    @property
    def value_mover(self) -> float:
        """The same value from the point of view of the player who just moved."""
        return -self.value

    @property
    def plies_remaining(self) -> int:
        return self.plies_total - self.ply - 1

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(side_to_move=self.side_to_move, result_stm=self.result_stm,
                 value_mover=self.value_mover, plies_remaining=self.plies_remaining)
        return d


def read_header(f) -> Header:
    raw = f.read(FILE_HDR)
    if len(raw) != FILE_HDR or raw[:8] != MAGIC:
        raise ValueError(f"not a .tkn shard (magic {raw[:8]!r})")
    u32 = lambda o: struct.unpack_from("<I", raw, o)[0]
    u64 = lambda o: struct.unpack_from("<Q", raw, o)[0]
    if u32(12) != FILE_HDR or u32(16) != GAME_HDR or u32(20) != REC:
        raise ValueError(f"unexpected record sizes {u32(12)}/{u32(16)}/{u32(20)}")
    return Header(version=u32(8), seed_base=u64(32), games=u64(40), plies=u64(48),
                  temperature=struct.unpack_from("<d", raw, 56)[0],
                  max_plies=u32(64), rep_limit=u32(68), no_progress_limit=u32(72),
                  stalemate_loses=u32(76), regicide=u32(80), board_size=u32(84),
                  piece_types=u32(88), sub_batch=u32(96), fp16=u32(100),
                  model=raw[104:128].split(b"\x00")[0].decode(), mat_shift=u32(28),
                  value_fp=u32(24))


def shard_paths(path: str) -> list[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.tkn")))
    return [path]


def _canonicalise(board: np.ndarray, info: NNInfo) -> tuple[np.ndarray, NNInfo]:
    """Into the frame of the side to move, which is the frame the network saw.

    ``move``, ``material`` and ``result`` are rewritten into the same frame, matching
    unpack.py, so ``result == 1`` means the side to move went on to win. ``result_black``
    keeps the absolute answer. The mover is by definition not the side to move, so ``side``
    comes out 0 either way; ``result_black`` and ``move`` are the only things that still
    remember which way round the board was.
    """
    info.canonical = True
    if info.side == 0:                 # white moved, so black is to move: already canonical
        return board, info
    o, m = rot180_move(info.move["origin"], info.move["special"] * 1296 + info.move["target"])
    info.move = {"special": m // 1296, "origin": o, "target": m % 1296}
    info.material = -info.material
    info.result = -info.result
    info.side = 0
    return canonical_board(board, 0), info


def unpack_nn(path: str, canonical: bool = True, terminal_only: bool = False,
              copy: bool = True, limit: int | None = None):
    """Yield ``(board, NNInfo)`` for every ply of every game in a shard, file or directory.

    ``terminal_only`` keeps just the last ply of each game -- the final position, the one
    the result is actually about.
    """
    n = 0
    for p in shard_paths(path):
        with open(p, "rb") as f:
            hdr = read_header(f)
            gi = 0
            while True:
                gh = f.read(GAME_HDR)
                if len(gh) < GAME_HDR:
                    break
                n_plies, result, term = struct.unpack_from("<IbB", gh, 0)
                seed = struct.unpack_from("<Q", gh, 8)[0]
                raw = f.read(n_plies * REC)
                if len(raw) < n_plies * REC:
                    raise ValueError(f"{p}: game {gi} truncated")
                recs = np.frombuffer(raw, dtype=REC_DTYPE)
                for i, r in enumerate(recs):
                    if terminal_only and i != n_plies - 1:
                        continue
                    board = r["board"].reshape(36, 36)
                    info = NNInfo(
                        path=p, game_index=gi, seed=seed, ply=i, plies_total=n_plies,
                        side=int(r["side"]), side_black=int(r["side"]),
                        move={k: unpack_move(int(r["move"]))[k] for k in ("special", "origin", "target")},
                        value=float(r["value"]), n_legal=int(r["n_legal"]),
                        material=float(r["material"]) / (1 << hdr.mat_shift),
                        result=int(result), result_black=int(result),
                        termination=TERM_NAMES[term],
                        forced_king_capture=bool(r["flags"] & PF_FORCED_KING),
                        was_capture=bool(r["flags"] & PF_CAPTURE),
                        was_promotion=bool(r["flags"] & PF_PROMOTION),
                        canonical=False)
                    if canonical:
                        board, info = _canonicalise(board, info)
                    yield (board.copy() if copy else board), info
                    n += 1
                    if limit is not None and n >= limit:
                        return
                gi += 1


def games(path: str):
    """Yield ``(header, game_index, seed, result, termination, n_plies)`` per game."""
    for p in shard_paths(path):
        with open(p, "rb") as f:
            hdr = read_header(f)
            gi = 0
            while True:
                gh = f.read(GAME_HDR)
                if len(gh) < GAME_HDR:
                    break
                n_plies, result, term = struct.unpack_from("<IbB", gh, 0)
                seed = struct.unpack_from("<Q", gh, 8)[0]
                f.seek(n_plies * REC, os.SEEK_CUR)
                yield hdr, gi, seed, result, TERM_NAMES[term], n_plies
                gi += 1


def main():
    ap = argparse.ArgumentParser(description="inspect a .tkn self-play shard")
    ap.add_argument("path")
    ap.add_argument("--show", type=int, default=3, help="plies to print")
    ap.add_argument("--board", action="store_true", help="print the board too")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--absolute", action="store_true", help="do not canonicalise")
    ap.add_argument("--count", action="store_true", help="summarise only")
    a = ap.parse_args()

    for p in shard_paths(a.path):
        with open(p, "rb") as f:
            print(f"{p}: {json.dumps(asdict(read_header(f)))}")

    gs = list(games(a.path))
    if gs:
        res = np.array([g[3] for g in gs])
        pl = np.array([g[5] for g in gs])
        print(f"{len(gs)} games, {pl.sum()} plies (mean {pl.mean():.1f}, max {pl.max()}); "
              f"black {(res == 1).sum()} white {(res == -1).sum()} draw {(res == 0).sum()}")
        from collections import Counter
        print("  terminations:", dict(Counter(g[4] for g in gs)))
    if a.count:
        return

    shown = 0
    vals = []
    for board, info in unpack_nn(a.path, canonical=not a.absolute):
        vals.append(info.value)
        if shown < a.show:
            shown += 1
            if a.json:
                print(json.dumps(info.as_dict()))
            else:
                m = info.move
                print(f"game {info.game_index} ply {info.ply}/{info.plies_total} "
                      f"mover={'black' if info.side_black else 'white'} "
                      f"move={m['special']}:{m['origin']}->{m['target']} "
                      f"value={info.value:+.4f} n_legal={info.n_legal} "
                      f"material={info.material:+.1f} result={info.result}"
                      f"{' [forced king capture]' if info.forced_king_capture else ''}")
            if a.board:
                for x in range(36):
                    print(" ".join(f"{square_name(int(c)):>4.4}" for c in board[x]))
    v = np.array(vals)
    print(f"values: n={len(v)} mean={v.mean():+.4f} sd={v.std():.4f} "
          f"min={v.min():+.4f} max={v.max():+.4f}")


if __name__ == "__main__":
    main()
