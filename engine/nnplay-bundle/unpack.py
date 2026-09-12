"""Unpacks either packed corpus format into (board, info) pairs.

Both formats are handled by one entry point:

    from unpack import unpack
    for board, info in unpack("corpus/"):
        board          # (36, 36) uint16, board[x][y]: x is the rank, y the file
        info.material  # ... and everything else stored with that position

* ``.tky`` (``--format games``) stores whole games as move lists. Positions are recovered
  by replaying, which needs no move *generation* -- only the three move-application rules
  and the promotion table, both of which live in this file.
* ``.tkp`` (``--format positions``) stores the sampled positions as board snapshots at a
  fixed stride, so records are read straight out of an mmap.

Only numpy and ``tables.json`` (generated from ``game.js`` by ``gen_tables.js``) are needed.

Run it directly to inspect a corpus::

    python3 unpack.py corpus/ --show 3 --board
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------- static tables
_T = json.loads((Path(__file__).resolve().parent / "tables.json").read_text())

IDS: list[int] = _T["ids"]                         # piece index -> the id used in game.js
NPIECE = len(IDS)
EMPTY_IDX = IDS.index(0)
KING_IDX = IDS.index(1000)
PROMOTE_IDX = np.array(_T["promote_idx"], dtype=np.int16)
PIECE_NAME: list[str] = [_T["defs"][i * 2 + 1]["name"] for i in range(NPIECE)]

VALUE_FP = 1000                                    # piece values are in 1/1000 pawn
EMPTY_CODE = (EMPTY_IDX << 2) | (1 << 1)           # game.js gives empty squares colour 1
INIT_BOARD = np.array([(c[0] << 2) | (c[1] << 1) | c[2] for c in _T["init"]], dtype=np.uint16)

# Empty squares carry a colour bit like any other square, and game.js pins it to 1, so
# "empty with colour 0" is a hole in code>>1 that no position can ever contain. Empty is
# piece index 0, so subtracting one closes it and leaves a dense range.
assert EMPTY_IDX == 0 and (EMPTY_CODE >> 1) == 1, "embedding_index assumes empty is index 0, colour 1"
NUM_EMBEDDING = NPIECE * 2 - 1                     # 603: 301 real pieces x 2 colours, + empty

# material value per packed code; the King is excluded, both sides hold one until the end
MATVAL = np.zeros(NPIECE * 4, dtype=np.int32)
for _i, _id in enumerate(IDS):
    _nat, _pro = _T["values"][str(_id)]
    for _c in (0, 1):
        MATVAL[(_i << 2) | (_c << 1)] = _nat
        MATVAL[(_i << 2) | (_c << 1) | 1] = _pro

DIRX = (1, 1, 0, -1, -1, -1, 0, 1)
DIRY = (0, 1, 1, 1, 0, -1, -1, -1)

RESULT_NAMES = ("white", "black", "draw")
TERM_NAMES = ("royal_capture", "stalemate", "repetition", "no_progress", "ply_cap")

MAGIC_GAMES = b"TKYSHOGI"
MAGIC_POSITIONS = b"TKYPOSNS"
MOVE_SELECTED = 8                                  # bit 3 of the move word
TKP_REC = 1296 * 2 + 32

_TKP_DTYPE = np.dtype([
    ("board", "<u2", 1296), ("seed", "<u8"), ("game_index", "<u4"), ("plies_total", "<u4"),
    ("ply", "<u2"), ("nmoves", "<u2"), ("mat", "<i2"), ("n_occupied", "<u2"),
    ("side_to_move", "u1"), ("result", "u1"), ("termination", "u1"), ("bin", "u1"),
    ("reserved", "<u4"),
])
assert _TKP_DTYPE.itemsize == TKP_REC

_PLY_DTYPE = np.dtype([("move", "<u4"), ("mat", "<i2"), ("nmoves", "<u2")])


# ---------------------------------------------------------------- square decoding
# A square is one uint16: code = piece_index << 2 | colour << 1 | promoted.
def piece_index(board: np.ndarray) -> np.ndarray:
    """Piece index, 0..301. Index into PIECE_NAME / IDS."""
    return (board >> 2).astype(np.int32)


def piece_id(board: np.ndarray) -> np.ndarray:
    """The id used in game.js (1 = Pawn, 1000 = King, 0 = empty)."""
    return np.asarray(IDS, dtype=np.int32)[piece_index(board)]


def colour(board: np.ndarray) -> np.ndarray:
    """1 = black, 0 = white. Meaningless where the square is empty."""
    return ((board >> 1) & 1).astype(np.int8)


def promoted(board: np.ndarray) -> np.ndarray:
    return (board & 1).astype(bool)


def occupied(board: np.ndarray) -> np.ndarray:
    return piece_index(board) != EMPTY_IDX


def embedding_index(board: np.ndarray) -> np.ndarray:
    """Dense 0..NUM_EMBEDDING-1 (0..602): 0 is empty, then piece_index * 2 + colour.

    The input to a single ``nn.Embedding(NUM_EMBEDDING, d)``. Every value in the range is
    reachable -- all 301 piece indices occur (209 in the initial setup plus 146 promotion
    targets), each in both colours, plus one code for empty.

    The promoted bit is dropped because it never changes how a piece moves; feed it as a
    separate plane if the value head wants it, since promoted and unpromoted values differ.
    """
    return (board >> 1).astype(np.int32) - 1


def material_plane(board: np.ndarray) -> np.ndarray:
    """Signed material per square in pawns, black positive. Sums to ``info.material``."""
    v = MATVAL[board.astype(np.int64)].astype(np.float32) / VALUE_FP
    return np.where(colour(board) == 1, v, -v) * occupied(board)


def square_name(code: int) -> str:
    idx = code >> 2
    if idx == EMPTY_IDX:
        return "."
    return f"{'B' if (code >> 1) & 1 else 'W'} {PIECE_NAME[idx]}{'+' if code & 1 else ''}"


# ---------------------------------------------------------------- canonicalisation
def rot180_square(sq: int) -> int:
    """(x, y) -> (35-x, 35-y) on the x*36+y index."""
    return 1295 - sq


def rot180_move(origin: int, mv_code: int) -> tuple[int, int]:
    """The same move seen from the other end of the board.

    Special 1-8 carries a DIRS index for the first leg, and a 180 degree rotation maps
    DIRS[i] to DIRS[i+4], which is why invert_color() rotates the 8-arrays by 4. Specials
    0 and 9 encode no direction and pass through.
    """
    special, target = divmod(mv_code, 1296)
    if 1 <= special <= 8:
        special = ((special - 1 + 4) % 8) + 1
    return rot180_square(origin), special * 1296 + rot180_square(target)


def canonical_board(board: np.ndarray, side_to_move: int) -> np.ndarray:
    """Return the position with the side to move playing as black, up the board.

    Black to move is returned unchanged; white to move is rotated 180 degrees and has every
    colour bit flipped. That is exact rather than approximate here: the setup is a 180
    degree rotation (not a mirror), and a white piece's move table is by construction the
    180 degree rotation of the black one, so a rotated white piece *is* the black piece of
    the same id. The promotion zones map onto each other too -- white's x <= 10 becomes
    x >= 25. run_tests.sh checks the claim by asking the engine for the legal moves of a
    position and of its canonical form and requiring one to be the rotation of the other.

    Use it to spare the network from learning both perspectives; the cost is that the
    absolute frame is gone, so keep side_to_move if anything downstream needs it.
    """
    if side_to_move == 1:
        return board
    b = board[::-1, ::-1].copy()
    return np.where((b >> 2) != EMPTY_IDX, b ^ 2, b).astype(np.uint16)  # flip colour only


# ---------------------------------------------------------------- move decoding
def unpack_move(mv: int) -> dict:
    """-> special, origin, target, capture and promotion flags, and the selected bit."""
    return {
        "special": (mv >> 28) & 0xF,   # 0 ordinary, 1-8 two-step, 9 trample / ranging
        "origin": (mv >> 17) & 0x7FF,  # x*36 + y
        "target": (mv >> 6) & 0x7FF,
        "was_capture": bool(mv >> 5 & 1),
        "was_promotion": bool(mv >> 4 & 1),
        "selected": bool(mv & MOVE_SELECTED),
    }


def is_terminal(mv: int) -> bool:
    """Terminal records carry no move; ``special`` never exceeds 9 on a real one."""
    return (mv >> 28) == 0xF


def apply_move(board: np.ndarray, mv: int) -> None:
    """Board.move from game.js, in place on a flat 1296 array. No legality knowledge."""
    special, origin, target = (mv >> 28) & 0xF, (mv >> 17) & 0x7FF, (mv >> 6) & 0x7FF
    if special == 0:
        board[target] = board[origin]
        board[origin] = EMPTY_CODE
    elif special <= 8:
        d = special - 1
        mid = (origin // 36 + DIRX[d]) * 36 + (origin % 36 + DIRY[d])
        board[mid] = board[origin]
        board[origin] = EMPTY_CODE
        board[target] = board[mid]
        board[mid] = EMPTY_CODE
    else:
        ox, oy = divmod(origin, 36)
        tx, ty = divmod(target, 36)
        dx, dy = tx - ox, ty - oy
        ux, uy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
        board[target] = board[origin]                       # target is written first
        for i in range(max(abs(dx), abs(dy))):              # then [origin, target) cleared
            board[(ox + i * ux) * 36 + (oy + i * uy)] = EMPTY_CODE
    # promotion is automatic, forced, and evaluated only on the ending square
    code = int(board[target])
    col, prom = (code >> 1) & 1, code & 1
    if not prom and ((col == 1 and target // 36 >= 25) or (col == 0 and target // 36 <= 10)):
        nxt = int(PROMOTE_IDX[code >> 2])
        if nxt >= 0:
            board[target] = (nxt << 2) | (col << 1) | 1


# ---------------------------------------------------------------- header and info
@dataclass
class Header:
    kind: str                  # "games" or "positions"
    version: int
    file_hdr: int
    a16: int                   # games: game header bytes; positions: record bytes
    a20: int                   # games: ply record bytes; positions: board squares
    value_fp: int
    mat_shift: int             # material unit is 1 >> mat_shift pawns
    seed_base: int
    first_game: int
    count: int                 # games: games in shard; positions: records in shard
    second: int                # games: plies; positions: positions examined by the sampler
    tanh_scale_fp: int
    policy: int
    max_plies: int
    rep_limit: int
    no_progress_limit: int
    stalemate_loses: int
    board_size: int
    npiece: int
    forced_king_capture: int
    save_percentage: float
    sel_bins: int

    @property
    def tanh_scale_pawns(self) -> float:
        return self.tanh_scale_fp / self.value_fp

    @property
    def material_unit_pawns(self) -> float:
        return 1.0 / (1 << self.mat_shift)

    @property
    def policy_name(self) -> str:
        return "uniform-move" if self.policy == 0 else "uniform-piece"


@dataclass
class Info:
    """Everything stored alongside the board, plus the few things derived from it."""
    source: str                # file the record came from
    kind: str                  # "games" or "positions"
    game_index: int
    seed: int
    ply: int
    plies_total: int
    side_to_move: int          # 1 = black, 0 = white
    material: float            # pawns, black minus white, King excluded
    label: float               # tanh(material / S), black's point of view
    label_bin: int             # which of header.sel_bins the sampler put it in
    n_legal: int               # legal moves for the side to move
    n_occupied: int
    result: int                # 0 white wins, 1 black wins, 2 draw
    result_name: str
    termination: int
    termination_name: str
    selected: bool             # kept by the --save-percentage sampler
    canonical: bool = True     # board is in the side-to-move's frame (see unpack)
    move: dict | None = field(default=None)   # the move played from here; None at the end

    @property
    def label_stm(self) -> float:
        """Label from the side-to-move's point of view."""
        return self.label if self.side_to_move == 1 else -self.label

    @property
    def result_stm(self) -> float:
        """+1 win / -1 loss / 0 draw, from the side-to-move's point of view."""
        return 0.0 if self.result == 2 else (1.0 if self.result == self.side_to_move else -1.0)

    @property
    def plies_remaining(self) -> int:
        return self.plies_total - self.ply

    def as_dict(self) -> dict:
        return asdict(self)


def _read_header(buf: np.ndarray, path: Path) -> Header:
    magic = bytes(buf[:8])
    if magic == MAGIC_GAMES:
        kind = "games"
    elif magic == MAGIC_POSITIONS:
        kind = "positions"
    else:
        raise ValueError(f"{path}: not a .tky or .tkp shard (magic {magic!r})")
    f = struct.unpack_from("<7I", buf, 8)
    q = struct.unpack_from("<4Q", buf, 32)
    h = struct.unpack_from("<11I", buf, 64)
    return Header(kind=kind, version=f[0], file_hdr=f[1], a16=f[2], a20=f[3], value_fp=f[4],
                  mat_shift=f[5], seed_base=q[0], first_game=q[1], count=q[2], second=q[3],
                  tanh_scale_fp=h[0], policy=h[1], max_plies=h[2], rep_limit=h[3],
                  no_progress_limit=h[4], stalemate_loses=h[5], board_size=h[6], npiece=h[7],
                  forced_king_capture=h[8], save_percentage=h[9] / 100.0, sel_bins=h[10] or 64)


def read_header(path: str | os.PathLike) -> Header:
    p = Path(path)
    return _read_header(np.memmap(p, dtype=np.uint8, mode="r"), p)


def shard_paths(root: str | os.PathLike) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    paths = sorted(root.glob("*.tky")) + sorted(root.glob("*.tkp"))
    if not paths:
        raise ValueError(f"no .tky or .tkp shards under {root}")
    return paths


# ---------------------------------------------------------------- the unpackers
def _canonicalise(board: np.ndarray, info: Info) -> tuple[np.ndarray, Info]:
    """Rewrite a (board, info) pair into the side-to-move's frame."""
    if info.side_to_move == 1:
        info.canonical = True
        return board, info
    board = canonical_board(board, 0)
    info.material = -info.material
    info.label = -info.label
    info.label_bin = _label_bin(info.label, _CANON_BINS[0])
    info.result = 2 if info.result == 2 else (1 if info.result == info.side_to_move else 0)
    info.result_name = RESULT_NAMES[info.result]
    info.side_to_move = 1
    info.canonical = True
    if info.move is not None:
        o, m = rot180_move(info.move["origin"], info.move["special"] * 1296 + info.move["target"])
        info.move = dict(info.move, origin=o, special=m // 1296, target=m % 1296)
    return board, info


_CANON_BINS = [64]


def _label_bin(label: float, bins: int) -> int:
    b = int((label + 1.0) * 0.5 * bins)
    return 0 if b < 0 else (bins - 1 if b >= bins else b)


def _unpack_games(path: Path, buf: np.ndarray, h: Header, selected_only: bool,
                  copy: bool) -> Iterator[tuple[np.ndarray, Info]]:
    S = h.tanh_scale_pawns
    unit = h.material_unit_pawns
    off, total = h.file_hdr, len(buf)
    for gi in range(h.count):
        if off + h.a16 > total:
            break
        n_plies, result, term = struct.unpack_from("<IBB", buf, off)
        seed = struct.unpack_from("<Q", buf, off + 8)[0]
        recs = np.frombuffer(buf, dtype=_PLY_DTYPE, count=n_plies + 1, offset=off + h.a16)
        off += h.a16 + (n_plies + 1) * h.a20

        board = INIT_BOARD.copy()
        for ply in range(n_plies + 1):
            mv = int(recs["move"][ply])
            sel = bool(mv & MOVE_SELECTED)
            if sel or not selected_only:
                mat = float(recs["mat"][ply]) * unit
                label = float(np.tanh(mat / S))
                grid = board.reshape(36, 36)
                yield (grid.copy() if copy else grid, Info(
                    source=str(path), kind="games",
                    game_index=int(h.first_game) + gi, seed=seed,
                    ply=ply, plies_total=n_plies, side_to_move=1 - (ply & 1),
                    material=mat, label=label, label_bin=_label_bin(label, h.sel_bins),
                    n_legal=int(recs["nmoves"][ply]),
                    n_occupied=int(np.count_nonzero(occupied(board))),
                    result=result, result_name=RESULT_NAMES[result],
                    termination=term, termination_name=TERM_NAMES[term], selected=sel,
                    move=None if is_terminal(mv) else unpack_move(mv)))
            if is_terminal(mv):
                break
            apply_move(board, mv)


def _unpack_positions(path: Path, buf: np.ndarray, h: Header, copy: bool
                      ) -> Iterator[tuple[np.ndarray, Info]]:
    if h.a16 != TKP_REC:
        raise ValueError(f"{path}: record size {h.a16}, expected {TKP_REC}")
    n = (len(buf) - h.file_hdr) // TKP_REC
    recs = np.frombuffer(buf, dtype=_TKP_DTYPE, count=n, offset=h.file_hdr)
    S = h.tanh_scale_pawns
    unit = h.material_unit_pawns
    for r in recs:
        mat = float(r["mat"]) * unit
        label = float(np.tanh(mat / S))
        grid = np.asarray(r["board"]).reshape(36, 36)
        yield (grid.copy() if copy else grid, Info(
            source=str(path), kind="positions", game_index=int(r["game_index"]),
            seed=int(r["seed"]), ply=int(r["ply"]), plies_total=int(r["plies_total"]),
            side_to_move=int(r["side_to_move"]), material=mat, label=label,
            label_bin=int(r["bin"]), n_legal=int(r["nmoves"]),
            n_occupied=int(r["n_occupied"]), result=int(r["result"]),
            result_name=RESULT_NAMES[int(r["result"])], termination=int(r["termination"]),
            termination_name=TERM_NAMES[int(r["termination"])], selected=True, move=None))


def unpack(root: str | os.PathLike, selected_only: bool | None = None,
           copy: bool = True, limit: int | None = None, canonical: bool = True
           ) -> Iterator[tuple[np.ndarray, Info]]:
    """Yields ``(board, info)`` for every position in a shard, file or directory.

    ``board`` is ``(36, 36)`` uint16 of packed codes, indexed ``board[x][y]`` -- x is the
    rank, y the file, matching ``board[x][y]`` in game.js. Decode it with ``piece_id``,
    ``colour``, ``promoted`` or ``embedding_index``.

    ``selected_only`` applies to the game format only, and defaults to True when the shard
    was written with a ``--save-percentage`` below 100, so a training loop gets the sampled,
    label-balanced subset without asking. The position format only ever holds selected
    positions.

    ``copy=False`` yields a view of the array being replayed, which is faster but is
    overwritten by the next step -- only use it if each board is consumed before the next
    ``next()``.

    ``canonical`` defaults to True: every white-to-move position is rotated 180 degrees
    with its colours swapped, so the side to move always plays as black up the board and the
    network never has to learn two perspectives. ``material``, ``label``, ``result`` and
    ``move`` are rewritten into that frame and ``side_to_move`` is pinned to 1, so ``label``
    equals ``label_stm`` and ``result == 1`` means the side to move won. Pass
    ``canonical=False`` for the absolute board as stored, in black's frame.
    """
    seen = 0
    for path in shard_paths(root):
        buf = np.memmap(path, dtype=np.uint8, mode="r")
        h = _read_header(buf, path)
        if h.kind == "positions":
            it = _unpack_positions(path, buf, h, copy)
        else:
            so = selected_only
            if so is None:
                so = 0.0 < h.save_percentage < 100.0
            it = _unpack_games(path, buf, h, so, copy)
        _CANON_BINS[0] = h.sel_bins
        for item in it:
            yield _canonicalise(*item) if canonical else item
            seen += 1
            if limit is not None and seen >= limit:
                return


# ---------------------------------------------------------------- CLI
def _render(board: np.ndarray) -> str:
    """Compact 36x36 map: '#' black, 'o' white, 'K'/'k' the Kings, '.' empty."""
    idx, col = piece_index(board), colour(board)
    out = ["    " + "".join(str(y // 10 % 10) if y % 5 == 0 else " " for y in range(36)),
           "    " + "".join(str(y % 10) if y % 5 == 0 else " " for y in range(36))]
    for x in range(36):
        row = []
        for y in range(36):
            if idx[x, y] == EMPTY_IDX:
                row.append(".")
            elif idx[x, y] == KING_IDX:
                row.append("K" if col[x, y] == 1 else "k")
            else:
                row.append("#" if col[x, y] == 1 else "o")
        out.append(f"{x:3d} " + "".join(row))
    return "\n".join(out)


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", help="a .tky/.tkp file, or a directory of shards")
    ap.add_argument("--show", type=int, default=2, help="entries to print in full")
    ap.add_argument("--stride", type=int, default=1, help="print every Nth entry")
    ap.add_argument("--limit", type=int, default=None, help="stop after N entries")
    ap.add_argument("--board", action="store_true", help="also draw each shown board")
    ap.add_argument("--all", action="store_true", help="game format: every ply, not just selected")
    ap.add_argument("--absolute", action="store_true",
                    help="board as stored, in black's frame (default is the side-to-move's)")
    ap.add_argument("--json", action="store_true", help="print info as JSON")
    ap.add_argument("--count", action="store_true", help="just count entries and summarise")
    a = ap.parse_args()

    for p in shard_paths(a.path):
        h = read_header(p)
        print(f"{p.name}: {h.kind}, v{h.version}, {h.count:,} "
              f"{'games' if h.kind == 'games' else 'positions'}, "
              f"S={h.tanh_scale_pawns:g} pawns, material unit {h.material_unit_pawns} pawn, "
              f"save={h.save_percentage:g}%, policy={h.policy_name}, "
              f"forced_king_capture={bool(h.forced_king_capture)}")
        break

    shown = n = 0
    mats, labels = [], []
    for board, info in unpack(a.path, selected_only=False if a.all else None,
                              limit=a.limit, canonical=not a.absolute):
        if n % a.stride == 0:
            mats.append(info.material)
            labels.append(info.label)
            if shown < a.show and not a.count:
                print(f"\n--- entry {n}")
                if a.json:
                    print(json.dumps(info.as_dict(), indent=2, default=str))
                else:
                    d = info.as_dict()
                    d.pop("source")
                    print(f"  board {board.shape} {board.dtype}, "
                          f"{int(np.count_nonzero(occupied(board)))} occupied, "
                          f"material recomputed {material_plane(board).sum():+.3f} pawns")
                    for k, v in d.items():
                        print(f"  {k:16s} {v}")
                    print(f"  {'label_stm':16s} {info.label_stm:+.4f}")
                    print(f"  {'result_stm':16s} {info.result_stm:+.0f}")
                if a.board:
                    print(_render(board))
                shown += 1
        n += 1
    if not n:
        print("no entries")
        return
    mats, labels = np.array(mats), np.array(labels)
    print(f"\n{n:,} entries; material {mats.mean():+.2f} +/- {mats.std():.2f} pawns "
          f"[{mats.min():+.1f}, {mats.max():+.1f}]; label sd {labels.std():.3f}")
    hist, _ = np.histogram(labels, bins=8, range=(-1, 1))
    print("label histogram (8 bins on [-1,1]): " + " ".join(f"{c:,}" for c in hist))


if __name__ == "__main__":
    _main()
