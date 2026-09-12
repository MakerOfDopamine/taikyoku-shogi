"""Reader for the .tky corpus format.

A shard stores games as (header, move list). Positions are recovered by replaying the
moves, which needs no move *generation* -- only the three move-application rules and the
promotion table -- so a training pipeline never has to link the engine.

Typical use::

    from tky_reader import Corpus
    c = Corpus("out/")
    for pos in c.positions(shuffle_games=True, seed=0):
        x     = pos.board          # int16[1296], piece codes; 0..NDEF-1 index an embedding
        stm   = pos.side_to_move   # 1 = black, 0 = white
        y     = pos.label          # tanh(material / S), from BLACK's point of view
        ...

Everything is derived from tables.json, which gen_tables.js writes by executing game.js.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_T = json.loads((_HERE / "tables.json").read_text())

IDS: list[int] = _T["ids"]                       # piece index -> game.js id
NPIECE = len(IDS)
EMPTY_IDX = IDS.index(0)
KING_IDX = IDS.index(1000)
PROMOTE_IDX = np.array(_T["promote_idx"], dtype=np.int16)   # piece index -> promoted index, -1 if none
PIECE_NAME = {IDS[i]: n for i, n in enumerate([])}  # filled below if needed

# code = idx<<2 | color<<1 | promoted
EMPTY_CODE = (EMPTY_IDX << 2) | (1 << 1)          # game.js gives empty squares colour 1
INIT_BOARD = np.array([(c[0] << 2) | (c[1] << 1) | c[2] for c in _T["init"]], dtype=np.uint16)

# material value per code, in 1/1000 pawn; the King is excluded (both sides always have one)
VALUE_FP = 1000
_MATVAL = np.zeros(NPIECE * 4, dtype=np.int32)
for _i, _id in enumerate(IDS):
    _nat, _pro = _T["values"][str(_id)]
    for _c in (0, 1):
        _MATVAL[(_i << 2) | (_c << 1) | 0] = _nat
        _MATVAL[(_i << 2) | (_c << 1) | 1] = _pro

DIRX = (1, 1, 0, -1, -1, -1, 0, 1)
DIRY = (0, 1, 1, 1, 0, -1, -1, -1)

RESULT_NAMES = ("white", "black", "draw")
TERM_NAMES = ("royal_capture", "stalemate", "repetition", "no_progress", "ply_cap")

MAGIC = b"TKYSHOGI"
MAGIC_POS = b"TKYPOSNS"
MOVE_SELECTED = 8            # bit 3 of the move word: this position was kept by the sampler


def is_terminal(mv: int) -> bool:
    """A terminal record carries no move; special is never 0xF on a real one (max 9)."""
    return (mv >> 28) == 0xF


@dataclass
class Header:
    version: int
    file_hdr: int
    game_hdr: int
    ply_rec: int
    value_fp: int
    mat_shift: int
    seed_base: int
    first_game: int
    n_games: int
    n_plies: int
    tanh_scale_fp: int
    policy: int
    max_plies: int
    rep_limit: int
    no_progress_limit: int
    stalemate_loses: int
    board: int
    npiece: int
    forced_king_capture: int
    save_percentage: float
    sel_bins: int

    @property
    def tanh_scale_pawns(self) -> float:
        return self.tanh_scale_fp / self.value_fp

    @property
    def mat_unit_pawns(self) -> float:
        """One unit of the stored material field, in pawns."""
        return 1.0 / (1 << self.mat_shift)


@dataclass
class Position:
    board: np.ndarray      # uint16[1296], packed codes, indexed x*36+y
    side_to_move: int      # 1 = black, 0 = white
    ply: int
    plies_total: int
    material: float        # pawns, black - white, King excluded
    label: float           # tanh(material / S), black's point of view
    n_legal: int           # legal moves for the side to move (see header.policy)
    selected: bool         # kept by the --save-percentage sampler
    result: int            # 0 white, 1 black, 2 draw -- for the whole game
    termination: int
    game_index: int
    move: int | None       # the move played from here, packed; None at the terminal ply

    @property
    def label_stm(self) -> float:
        """Label from the side-to-move's point of view."""
        return self.label if self.side_to_move == 1 else -self.label

    @property
    def result_stm(self) -> float:
        """+1 win / -1 loss / 0 draw, from the side-to-move's point of view."""
        if self.result == 2:
            return 0.0
        return 1.0 if self.result == self.side_to_move else -1.0


def unpack_move(mv: int) -> tuple[int, int, int, bool, bool]:
    """-> (special, origin_sq, target_sq, was_capture, was_promotion)"""
    return (mv >> 28) & 0xF, (mv >> 17) & 0x7FF, (mv >> 6) & 0x7FF, bool(mv >> 5 & 1), bool(mv >> 4 & 1)


def apply_move(board: np.ndarray, mv: int) -> None:
    """Board.move from game.js, in place. Needs no legality knowledge."""
    special, origin, target, _, _ = unpack_move(mv)
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
        size = max(abs(dx), abs(dy))
        board[target] = board[origin]                      # target is written first
        for i in range(size):
            board[(ox + i * ux) * 36 + (oy + i * uy)] = EMPTY_CODE
    # promotion: automatic, forced, evaluated only on the ending square
    code = int(board[target])
    color, promoted = (code >> 1) & 1, code & 1
    tx = target // 36
    if not promoted and ((color == 1 and tx >= 25) or (color == 0 and tx <= 10)):
        nxt = int(PROMOTE_IDX[code >> 2])
        if nxt >= 0:
            board[target] = (nxt << 2) | (color << 1) | 1


class Shard:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.buf = np.memmap(self.path, dtype=np.uint8, mode="r")
        if bytes(self.buf[:8]) != MAGIC:
            raise ValueError(f"{path}: not a .tky shard")
        f = struct.unpack_from("<7I", self.buf, 8)
        q = struct.unpack_from("<4Q", self.buf, 32)
        h2 = struct.unpack_from("<11I", self.buf, 64)
        self.header = Header(
            version=f[0], file_hdr=f[1], game_hdr=f[2], ply_rec=f[3], value_fp=f[4],
            mat_shift=f[5], seed_base=q[0], first_game=q[1], n_games=q[2], n_plies=q[3],
            tanh_scale_fp=h2[0], policy=h2[1], max_plies=h2[2], rep_limit=h2[3],
            no_progress_limit=h2[4], stalemate_loses=h2[5], board=h2[6], npiece=h2[7],
            forced_king_capture=h2[8], save_percentage=h2[9] / 100.0, sel_bins=h2[10] or 64)
        if self.header.ply_rec != 8 or self.header.game_hdr != 16:
            raise ValueError("unexpected record sizes")
        self._index: list[int] | None = None

    def index(self) -> list[int]:
        """Byte offset of every game header. One linear pass, then O(1) seeks."""
        if self._index is None:
            off, n, idx = self.header.file_hdr, self.header.n_games, []
            total = len(self.buf)
            for _ in range(n):
                if off + self.header.game_hdr > total:
                    break
                idx.append(off)
                n_plies = int(struct.unpack_from("<I", self.buf, off)[0])
                off += self.header.game_hdr + (n_plies + 1) * self.header.ply_rec
            self._index = idx
        return self._index

    def game(self, i: int):
        off = self.index()[i]
        n_plies, result, term = struct.unpack_from("<IBB", self.buf, off)
        seed = struct.unpack_from("<Q", self.buf, off + 8)[0]
        recs = np.frombuffer(self.buf, dtype=np.dtype([("move", "<u4"), ("mat", "<i2"), ("nmoves", "<u2")]),
                             count=n_plies + 1, offset=off + self.header.game_hdr)
        return n_plies, result, term, seed, recs

    def __len__(self):
        return len(self.index())


class Corpus:
    def __init__(self, root: str | os.PathLike):
        root = Path(root)
        self.shards = [Shard(p) for p in sorted(root.glob("*.tky"))] if root.is_dir() else [Shard(root)]
        if not self.shards:
            raise ValueError(f"no .tky shards under {root}")
        self.header = self.shards[0].header

    @property
    def n_games(self) -> int:
        return sum(len(s) for s in self.shards)

    def games(self, shuffle_games: bool = False, seed: int = 0):
        pairs = [(si, gi) for si, s in enumerate(self.shards) for gi in range(len(s))]
        if shuffle_games:
            np.random.default_rng(seed).shuffle(pairs)
        for si, gi in pairs:
            yield si, gi, self.shards[si].game(gi)

    def positions(self, shuffle_games: bool = False, seed: int = 0, stride: int = 1,
                  verify: bool = False, selected_only: bool | None = None):
        """Replays each game and yields Positions.

        selected_only defaults to True when the shard was written with --save-percentage
        below 100, so a training loop gets exactly the sampled, label-balanced subset
        without asking. Pass False to walk every ply.

        Replay is one game pass, so the amortised cost per position is a handful of array
        writes -- do NOT seek to a single ply by replaying from the start each time.
        """
        if selected_only is None:
            selected_only = 0.0 < self.header.save_percentage < 100.0
        h = self.header
        S = h.tanh_scale_pawns
        unit = h.mat_unit_pawns
        for si, gi, (n_plies, result, term, seed_, recs) in self.games(shuffle_games, seed):
            board = INIT_BOARD.copy()
            for ply in range(n_plies + 1):
                mv = int(recs["move"][ply])
                mat = float(recs["mat"][ply]) * unit
                if verify:
                    got = float(_MATVAL[board.astype(np.int64)].sum()) / VALUE_FP
                    sign = np.where(((board >> 1) & 1) == 1, 1, -1)
                    got = float((_MATVAL[board.astype(np.int64)] * sign).sum()) / VALUE_FP
                    if abs(got - mat) > unit / 2 + 1e-9:
                        raise AssertionError(f"shard {si} game {gi} ply {ply}: material {got} != stored {mat}")
                sel = bool(mv & MOVE_SELECTED)
                if ply % stride == 0 and (sel or not selected_only):
                    yield Position(
                        board=board, side_to_move=1 - (ply & 1), ply=ply, plies_total=n_plies,
                        material=mat, label=float(np.tanh(mat / S)), n_legal=int(recs["nmoves"][ply]),
                        selected=sel, result=result, termination=term, game_index=gi,
                        move=None if is_terminal(mv) else mv)
                if is_terminal(mv):
                    break
                apply_move(board, mv)


TKP_REC = 1296 * 2 + 32
_TKP_DTYPE = np.dtype([
    ("board", "<u2", 1296), ("seed", "<u8"), ("game_index", "<u4"), ("plies_total", "<u4"),
    ("ply", "<u2"), ("nmoves", "<u2"), ("mat", "<i2"), ("n_occupied", "<u2"),
    ("side_to_move", "u1"), ("result", "u1"), ("termination", "u1"), ("bin", "u1"),
    ("reserved", "<u4"),
])
assert _TKP_DTYPE.itemsize == TKP_REC, _TKP_DTYPE.itemsize


class PositionCorpus:
    """A --format positions corpus: fixed-stride board snapshots, no replay.

    Records are self-contained and all the same size, so `corpus.records` is a plain numpy
    struct array over an mmap -- index it however the sampler likes, including a full
    shuffle every epoch.
    """

    def __init__(self, root: str | os.PathLike):
        root = Path(root)
        paths = sorted(root.glob("*.tkp")) if root.is_dir() else [root]
        if not paths:
            raise ValueError(f"no .tkp shards under {root}")
        self.headers, self.parts = [], []
        for p in paths:
            buf = np.memmap(p, dtype=np.uint8, mode="r")
            if bytes(buf[:8]) != MAGIC_POS:
                raise ValueError(f"{p}: not a .tkp shard")
            f = struct.unpack_from("<7I", buf, 8)
            q = struct.unpack_from("<4Q", buf, 32)
            h2 = struct.unpack_from("<11I", buf, 64)
            if f[2] != TKP_REC:
                raise ValueError(f"{p}: record size {f[2]}, expected {TKP_REC}")
            self.headers.append(Header(
                version=f[0], file_hdr=f[1], game_hdr=f[2], ply_rec=f[3], value_fp=f[4],
                mat_shift=f[5], seed_base=q[0], first_game=q[1], n_games=q[2], n_plies=q[3],
                tanh_scale_fp=h2[0], policy=h2[1], max_plies=h2[2], rep_limit=h2[3],
                no_progress_limit=h2[4], stalemate_loses=h2[5], board=h2[6], npiece=h2[7],
                forced_king_capture=h2[8], save_percentage=h2[9] / 100.0, sel_bins=h2[10] or 64))
            n = (len(buf) - f[1]) // TKP_REC
            self.parts.append(np.frombuffer(buf, dtype=_TKP_DTYPE, count=n, offset=f[1]))
        self.header = self.headers[0]
        self.records = np.concatenate(self.parts) if len(self.parts) > 1 else self.parts[0]

    def __len__(self):
        return len(self.records)

    @property
    def positions_sampled(self) -> int:
        """How many positions were looked at to produce these, across all shards."""
        return sum(h.n_plies for h in self.headers)

    def labels(self) -> np.ndarray:
        """tanh(material / S) for every record, from black's point of view."""
        S = self.header.tanh_scale_pawns
        return np.tanh(self.records["mat"].astype(np.float32) * self.header.mat_unit_pawns / S)

    def position(self, i: int) -> Position:
        r = self.records[i]
        S = self.header.tanh_scale_pawns
        mat = float(r["mat"]) * self.header.mat_unit_pawns
        return Position(
            board=np.asarray(r["board"]), side_to_move=int(r["side_to_move"]), ply=int(r["ply"]),
            plies_total=int(r["plies_total"]), material=mat, label=float(np.tanh(mat / S)),
            n_legal=int(r["nmoves"]), selected=True, result=int(r["result"]),
            termination=int(r["termination"]), game_index=int(r["game_index"]), move=None)


NUM_EMBEDDING = NPIECE * 2 - 1        # 603; see unpack.embedding_index


def to_index_planes(board: np.ndarray, side_to_move: int) -> np.ndarray:
    """36x36 int16 in 0..NUM_EMBEDDING-1: 0 is empty, then piece index * 2 + colour.

    Note this takes the board as stored, in black's absolute frame. unpack.unpack()
    canonicalises to the side-to-move's frame by default; this module does not.

    One nn.Embedding(NUM_EMBEDDING, d) covers the board. The raw `code>>1` leaves a hole at
    0 -- empty squares carry a colour bit that game.js pins to 1, so "empty, colour 0"
    never occurs -- and the -1 closes it. The promoted bit is dropped because it never
    changes how a piece moves; feed it as a separate binary plane if the value head wants
    it (VALUES_PROMOTED differs)."""
    return (board >> 1).astype(np.int16).reshape(36, 36) - 1


def to_feature_planes(board: np.ndarray, side_to_move: int) -> np.ndarray:
    """A small, dense alternative to one-hotting 604 piece types: 5 float32 planes of
    36x36 -- own piece, enemy piece, promoted, per-square material value (pawns, signed
    for the side to move), and a constant side-to-move plane."""
    idx = (board >> 2).astype(np.int32)
    color = ((board >> 1) & 1).astype(np.int32)
    prom = (board & 1).astype(np.float32)
    occ = idx != EMPTY_IDX
    val = _MATVAL[board.astype(np.int64)].astype(np.float32) / VALUE_FP
    own = (occ & (color == side_to_move)).astype(np.float32)
    opp = (occ & (color != side_to_move)).astype(np.float32)
    signed = val * np.where(color == side_to_move, 1.0, -1.0) * occ
    stm = np.full(1296, float(side_to_move), dtype=np.float32)
    out = np.stack([own, opp, prom * occ, signed, stm]).reshape(5, 36, 36)
    return out


if __name__ == "__main__":
    import sys
    c = Corpus(sys.argv[1] if len(sys.argv) > 1 else "out")
    h = c.header
    print(f"shards={len(c.shards)} games={c.n_games} plies={sum(s.header.n_plies for s in c.shards)}")
    print(f"tanh scale S = {h.tanh_scale_pawns:.1f} pawns   material unit = {h.mat_unit_pawns} pawn")
    print(f"policy={'uniform-move' if h.policy == 0 else 'uniform-piece'} max_plies={h.max_plies} "
          f"rep_limit={h.rep_limit} stalemate_loses={h.stalemate_loses} "
          f"forced_king_capture={bool(h.forced_king_capture)}")
    print(f"save_percentage={h.save_percentage}")
    n = sel = 0
    for pos in c.positions(verify=True, selected_only=False):
        n += 1
        sel += pos.selected
    print(f"replayed {n} positions, material verified against the stored label at every ply; "
          f"{sel} ({100 * sel / max(1, n):.2f}%) flagged as selected")
    p = next(c.positions(stride=97))
    print(f"example: ply {p.ply}/{p.plies_total} stm={p.side_to_move} mat={p.material:+.2f} "
          f"label={p.label:+.4f} legal={p.n_legal} result={RESULT_NAMES[p.result]} "
          f"term={TERM_NAMES[p.termination]}")
    print("index planes", to_index_planes(p.board, p.side_to_move).shape,
          "feature planes", to_feature_planes(p.board, p.side_to_move).shape)
