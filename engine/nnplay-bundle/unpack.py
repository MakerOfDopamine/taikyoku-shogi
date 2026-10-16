"""The board encoding shared by nnplay's readers and exporters: piece tables from
``tables.json`` (generated from ``game.js`` by ``gen_tables.js``), square decoding, the
network's embedding index, the canonical side-to-move frame, and game.js's Board.move.

Only numpy and ``tables.json`` are needed.
"""
from __future__ import annotations

import json
from pathlib import Path

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

TERM_NAMES = ("royal_capture", "stalemate", "repetition", "no_progress", "ply_cap",
              "resignation")


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
    """A packed move word -> special, origin, target."""
    return {
        "special": (mv >> 28) & 0xF,   # 0 ordinary, 1-8 two-step, 9 trample / ranging
        "origin": (mv >> 17) & 0x7FF,  # x*36 + y
        "target": (mv >> 6) & 0x7FF,
    }


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
