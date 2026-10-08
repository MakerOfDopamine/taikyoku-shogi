"""One-cell .tkn reader for a marimo notebook: self-contained, and a single global name.

Two deliberate differences from unpack_nn.py:

  * it replaces unpack.py *and* unpack_nn.py, so neither needs to be in the notebook and
    the names they share cannot collide. The board decoders are inlined; the ones that
    need piece tables read tables.json lazily, so nothing is loaded unless asked for;
  * everything hangs off one class, so pasting it into a cell defines exactly one name,
    ``Tkn``. Even the imports live inside the methods, so no ``np`` or ``struct`` leaks
    into the cell and nothing can collide with another cell.

    for board, info in Tkn.unpack("selfplay.tkn"):
        x = Tkn.embedding_index(board)   # (36,36) int32 in 0..602 -- what nnplay fed the net
        y = info.value                   # the net's value of this board
        z = info.result_stm              # +1 if the side to move went on to win

Boards are stored absolute, in black's frame, whoever moved. ``Tkn.unpack`` canonicalises
to the side to move by default -- the frame the network was given -- so ``info.value`` is
the value of the board you are holding, with no sign to chase.

Games can end by adjudication (format version 4 on): ``termination == "resignation"`` when
one side held a large material lead, and ``"ply_cap"`` scored for the side ahead on
material -- a draw under a 50-pawn lead from version 5. ``Tkn.header(path)`` says which
rules a shard was played under. For a soft target on those games, e.g.

        z = np.tanh(info.final_material_stm / scale) if info.termination == "ply_cap" \
            else info.result_stm

``final_material_stm`` is the material balance the game ended on, for the side to move.
"""


class Tkn:
    NSQ = 1296
    MAGIC = b"TKYNNSP\x00"
    FILE_HDR = 128
    GAME_HDR = 16
    REC = NSQ * 2 + 16
    NUM_EMBEDDING = 603       # 301 piece types x 2 colours, plus empty: the nn.Embedding size
    EMPTY_IDX = 0             # IDS.index(0) in tables.json; empty squares are piece index 0
    TERM_NAMES = ("royal_capture", "stalemate", "repetition", "no_progress", "ply_cap",
                  "resignation")
    F_CAPTURE, F_PROMOTION, F_FORCED_KING, F_TERMINAL, F_KING_SAFETY, F_OPPONENT = \
        1, 2, 4, 8, 16, 32

    _dtype_cache = None
    _tables_cache = None
    TABLES_PATH = "tables.json"   # set this if tables.json is not in the working directory

    class Info:
        """One ply. Fields are set by Tkn.unpack; everything below them is derived.

        side            the mover in this frame: 1 black, 0 white (always 0 once canonical)
        side_black      absolute: 1 if black made this move
        value           the net's value of this board, in its side-to-move frame
        n_legal         legal moves the mover chose from
        n_scored        children actually evaluated (== n_legal unless --subset-fraction)
        material        pawns after this move: black minus white, or side to move minus
                        opponent once canonical
        final_material  the balance the game ended on, in the same frame as ``material``
        result          1 the side to move won, -1 lost, 0 drew (absolute: 1 = black)
        result_black    absolute, never reframed
        termination     royal_capture, ply_cap, resignation (or the never-seen stalemate,
                        repetition, no_progress)
        forced_king_capture   this move was a King capture the rules forced
        king_safety_override  the King-safety rule changed the pick on this ply
        opponent_move   from nnplay elo --out: the baseline or random mover played it
        """

        def __init__(self, **kw):
            self.__dict__.update(kw)

        @property
        def side_to_move(self):
            return 1 - self.side

        @property
        def result_stm(self):
            return self.result if self.side_to_move == 1 else -self.result

        @property
        def final_material_stm(self):
            """The balance the game ended on, for the side to move in this position."""
            return self.final_material if self.side_to_move == 1 else -self.final_material

        @property
        def value_mover(self):
            return -self.value

        @property
        def plies_remaining(self):
            return self.plies_total - self.ply - 1

        @property
        def scored_fraction(self):
            return self.n_scored / self.n_legal if self.n_legal else 1.0

        def as_dict(self):
            d = dict(self.__dict__)
            d.update(side_to_move=self.side_to_move, result_stm=self.result_stm,
                     value_mover=self.value_mover, plies_remaining=self.plies_remaining,
                     scored_fraction=self.scored_fraction,
                     final_material_stm=self.final_material_stm)
            return d

        def __repr__(self):
            return (f"Tkn.Info(game={self.game_index} ply={self.ply}/{self.plies_total} "
                    f"value={self.value:+.4f} n_legal={self.n_legal} n_scored={self.n_scored} "
                    f"result_stm={self.result_stm:+d} {self.termination})")

    @staticmethod
    def dtype():
        import numpy as np
        if Tkn._dtype_cache is None:
            d = np.dtype([("board", "<u2", (Tkn.NSQ,)), ("move", "<u4"), ("value", "<f4"),
                          ("n_legal", "<u2"), ("material", "<i2"), ("side", "u1"),
                          ("flags", "u1"), ("n_scored", "<u2")])
            assert d.itemsize == Tkn.REC, d.itemsize
            Tkn._dtype_cache = d
        return Tkn._dtype_cache

    # ---- board helpers, inlined from unpack.py so tables.json is not needed -------------
    @staticmethod
    def embedding_index(board):
        """(36,36) int32 in 0..602: 0 empty, then piece_index * 2 + colour.

        The promoted bit is dropped because it never changes how a piece moves. Feed it as
        a separate plane if the value head wants it, since promoted values differ.
        """
        return (board >> 1).astype("int32") - 1

    @staticmethod
    def canonical_board(board, side_to_move):
        """The side to move playing as black, up the board.

        Black to move is unchanged; white to move is rotated 180 degrees with every colour
        bit flipped. Exact rather than approximate: the setup is a rotation and not a
        mirror, a white piece's move table is by construction the rotation of the black
        one, and the promotion zones map onto each other.
        """
        import numpy as np
        if side_to_move == 1:
            return board
        b = board[::-1, ::-1].copy()
        return np.where((b >> 2) != Tkn.EMPTY_IDX, b ^ 2, b).astype("uint16")

    @staticmethod
    def rot180_move(origin, mv_code):
        """The same move from the other end of the board.

        Specials 1-8 carry a DIRS index for the first leg, and a 180 degree rotation sends
        DIRS[i] to DIRS[i+4]; 0 and 9 encode no direction and pass through.
        """
        special, target = divmod(mv_code, 1296)
        if 1 <= special <= 8:
            special = ((special - 1 + 4) % 8) + 1
        return 1295 - origin, special * 1296 + (1295 - target)

    @staticmethod
    def unpack_move(mv):
        return {"special": (mv >> 28) & 0xF, "origin": (mv >> 17) & 0x7FF,
                "target": (mv >> 6) & 0x7FF}

    # ---- piece tables, loaded only when something needs them ---------------------------
    @staticmethod
    def tables():
        """tables.json, cached. Only the decoders below need it; boards do not."""
        import json
        import numpy as np
        if Tkn._tables_cache is None:
            path = Tkn.TABLES_PATH
            if not __import__("os").path.exists(path) and "__file__" in globals():
                path = __import__("os").path.join(
                    __import__("os").path.dirname(__import__("os").path.abspath(__file__)),
                    "tables.json")
            t = json.loads(open(path).read())
            ids = t["ids"]
            npiece = len(ids)
            assert ids.index(0) == Tkn.EMPTY_IDX, "empty is not piece index 0"
            matval = np.zeros(npiece * 4, dtype=np.int32)
            for i, pid in enumerate(ids):
                nat, pro = t["values"][str(pid)]
                for c in (0, 1):
                    matval[(i << 2) | (c << 1)] = nat
                    matval[(i << 2) | (c << 1) | 1] = pro
            Tkn._tables_cache = {
                "ids": ids, "npiece": npiece, "king_idx": ids.index(1000),
                "piece_name": [t["defs"][i * 2 + 1]["name"] for i in range(npiece)],
                "matval": matval,
                "promote_idx": np.array(t["promote_idx"], dtype=np.int16),
                "init_board": np.array([(c[0] << 2) | (c[1] << 1) | c[2] for c in t["init"]],
                                       dtype=np.uint16),
            }
        return Tkn._tables_cache

    @staticmethod
    def piece_index(board):
        return (board >> 2).astype("int32")

    @staticmethod
    def piece_id(board):
        """The id game.js uses: 1 = Pawn, 1000 = King, 0 = empty. Needs tables.json."""
        import numpy as np
        return np.asarray(Tkn.tables()["ids"], dtype=np.int32)[Tkn.piece_index(board)]

    @staticmethod
    def colour(board):
        """1 black, 0 white. Meaningless where the square is empty."""
        return ((board >> 1) & 1).astype("int8")

    @staticmethod
    def promoted(board):
        return (board & 1).astype(bool)

    @staticmethod
    def occupied(board):
        return Tkn.piece_index(board) != Tkn.EMPTY_IDX

    @staticmethod
    def material_plane(board):
        """Signed material per square in pawns, black positive. Needs tables.json."""
        import numpy as np
        v = Tkn.tables()["matval"][board.astype(np.int64)].astype(np.float32) / 1000.0
        return np.where(Tkn.colour(board) == 1, v, -v) * Tkn.occupied(board)

    @staticmethod
    def square_name(code):
        """"B Pawn+" and so on; "." for empty. Needs tables.json."""
        idx = code >> 2
        if idx == Tkn.EMPTY_IDX:
            return "."
        return (f"{'B' if (code >> 1) & 1 else 'W'} "
                f"{Tkn.tables()['piece_name'][idx]}{'+' if code & 1 else ''}")

    @staticmethod
    def init_board():
        """The 1296-square starting position, absolute. Needs tables.json."""
        return Tkn.tables()["init_board"].copy()

    # ---- files ---------------------------------------------------------------------------
    @staticmethod
    def shards(path):
        import glob
        import os
        if os.path.isdir(path):
            return sorted(glob.glob(os.path.join(path, "*.tkn")))
        return [path]

    @staticmethod
    def header(path):
        import struct
        with open(path, "rb") as f:
            raw = f.read(Tkn.FILE_HDR)
        if len(raw) != Tkn.FILE_HDR or raw[:8] != Tkn.MAGIC:
            raise ValueError(f"not a .tkn shard (magic {raw[:8]!r})")
        u32 = lambda o: struct.unpack_from("<I", raw, o)[0]
        u64 = lambda o: struct.unpack_from("<Q", raw, o)[0]
        if u32(12) != Tkn.FILE_HDR or u32(16) != Tkn.GAME_HDR or u32(20) != Tkn.REC:
            raise ValueError(f"unexpected record sizes {u32(12)}/{u32(16)}/{u32(20)}")
        return {"version": u32(8), "value_fp": u32(24), "mat_shift": u32(28),
                "seed_base": u64(32), "games": u64(40), "plies": u64(48),
                "temperature": struct.unpack_from("<d", raw, 56)[0],
                "max_plies": u32(64), "rep_limit": u32(68), "no_progress_limit": u32(72),
                "stalemate_loses": u32(76), "regicide": u32(80) & 1, "board_size": u32(84),
                # bit 1 of the regicide word, version 3 on
                "king_safety": (u32(80) >> 1) & 1 if u32(8) >= 3 else 0,
                # version 4 on: ply cap scored on material, and material resignation
                "cap_by_material": (u32(80) >> 2) & 1 if u32(8) >= 4 else 0,
                # the ply cap is a draw under this lead: 0 in version 4, 50 from 5
                "cap_draw_margin": ((50 if u32(8) >= 5 else 0)
                                    if u32(8) >= 4 and (u32(80) >> 2) & 1 else None),
                "resign_lead": (u32(80) >> 8) & 0xFFFF if u32(8) >= 4 else 0,
                "resign_plies": u32(80) >> 24 if u32(8) >= 4 else 0,
                "piece_types": u32(88), "sub_batch": u32(96), "fp16": u32(100) & 1,
                # version 6 on: the children scored were the policy's top this much of the
                # legal moves, plus this much at random
                "policy_top": (u32(100) >> 16) / 1000 if u32(8) >= 6 else None,
                "policy_explore": ((u32(100) >> 8) & 0xFF) / 1000 if u32(8) >= 6 else None,
                # 92..95 held NSQ before version 2; from 2 on it is the opening schedule
                "opening_plies": struct.unpack_from("<H", raw, 92)[0] if u32(8) >= 2 else 0,
                "opening_temperature": (struct.unpack_from("<H", raw, 94)[0] / 1000.0
                                        if u32(8) >= 2 else None),
                "model": raw[104:128].split(b"\x00")[0].decode()}

    MATERIAL_OFF = 1296 * 2 + 4 + 4 + 2   # board, move, value, n_legal: then i16 material

    @staticmethod
    def games(path):
        """Yield one dict per game without decoding any ply. ``final_material`` is read
        from the last record alone: pawns, black minus white."""
        import os
        import struct
        for p in Tkn.shards(path):
            shift = Tkn.header(p)["mat_shift"]
            with open(p, "rb") as f:
                f.seek(Tkn.FILE_HDR)
                gi = 0
                while True:
                    gh = f.read(Tkn.GAME_HDR)
                    if len(gh) < Tkn.GAME_HDR:
                        break
                    n_plies, result, term = struct.unpack_from("<IbB", gh, 0)
                    seed = struct.unpack_from("<Q", gh, 8)[0]
                    final = 0.0
                    if n_plies:
                        f.seek((n_plies - 1) * Tkn.REC + Tkn.MATERIAL_OFF, os.SEEK_CUR)
                        final = struct.unpack("<h", f.read(2))[0] / (1 << shift)
                        f.seek(Tkn.REC - Tkn.MATERIAL_OFF - 2, os.SEEK_CUR)
                    yield {"path": p, "game_index": gi, "seed": seed, "n_plies": n_plies,
                           "result": result, "termination": Tkn.TERM_NAMES[term],
                           "final_material": final}
                    gi += 1

    @staticmethod
    def unpack(path, canonical=True, terminal_only=False, copy=True, limit=None):
        """Yield ``(board, Tkn.Info)`` for every ply of every game in a shard or directory.

        ``terminal_only`` keeps just the last ply of each game, the position the result is
        actually about. ``canonical=False`` leaves the board in the stored absolute frame.
        """
        import struct
        import numpy as np
        n = 0
        for p in Tkn.shards(path):
            hdr = Tkn.header(p)
            with open(p, "rb") as f:
                f.seek(Tkn.FILE_HDR)
                gi = 0
                while True:
                    gh = f.read(Tkn.GAME_HDR)
                    if len(gh) < Tkn.GAME_HDR:
                        break
                    n_plies, result, term = struct.unpack_from("<IbB", gh, 0)
                    seed = struct.unpack_from("<Q", gh, 8)[0]
                    raw = f.read(n_plies * Tkn.REC)
                    if len(raw) < n_plies * Tkn.REC:
                        raise ValueError(f"{p}: game {gi} truncated")
                    recs = np.frombuffer(raw, dtype=Tkn.dtype())
                    final = (float(recs["material"][-1]) / (1 << hdr["mat_shift"])
                             if n_plies else 0.0)
                    for i, r in enumerate(recs):
                        if terminal_only and i != n_plies - 1:
                            continue
                        board = r["board"].reshape(36, 36)
                        info = Tkn.Info(
                            path=p, game_index=gi, seed=seed, ply=i, plies_total=n_plies,
                            side=int(r["side"]), side_black=int(r["side"]),
                            move=Tkn.unpack_move(int(r["move"])),
                            value=float(r["value"]), n_legal=int(r["n_legal"]),
                            # 0 was the reserved pad before --subset-fraction existed
                            n_scored=int(r["n_scored"]) or int(r["n_legal"]),
                            material=float(r["material"]) / (1 << hdr["mat_shift"]),
                            final_material=final,
                            result=int(result), result_black=int(result),
                            termination=Tkn.TERM_NAMES[term],
                            forced_king_capture=bool(r["flags"] & Tkn.F_FORCED_KING),
                            king_safety_override=bool(r["flags"] & Tkn.F_KING_SAFETY),
                            opponent_move=bool(r["flags"] & Tkn.F_OPPONENT),
                            was_capture=bool(r["flags"] & Tkn.F_CAPTURE),
                            was_promotion=bool(r["flags"] & Tkn.F_PROMOTION),
                            canonical=False)
                        if canonical:
                            info.canonical = True
                            if info.side != 0:      # black moved, so white is to move
                                o, m = Tkn.rot180_move(
                                    info.move["origin"],
                                    info.move["special"] * 1296 + info.move["target"])
                                info.move = {"special": m // 1296, "origin": o,
                                             "target": m % 1296}
                                info.material = -info.material
                                info.final_material = -info.final_material
                                info.result = -info.result
                                info.side = 0
                                board = Tkn.canonical_board(board, 0)
                        yield (board.copy() if copy else board), info
                        n += 1
                        if limit is not None and n >= limit:
                            return
                    gi += 1

    @staticmethod
    def summary(path):
        """Counts and value statistics, for a quick look in a cell."""
        import numpy as np
        gs = list(Tkn.games(path))
        res = np.array([g["result"] for g in gs])
        pl = np.array([g["n_plies"] for g in gs])
        vals, frac = [], []
        for _b, info in Tkn.unpack(path):
            vals.append(info.value)
            frac.append(info.scored_fraction)
        v = np.array(vals)
        h = Tkn.header(Tkn.shards(path)[0])
        out = {"shards": len(Tkn.shards(path)), "games": len(gs), "plies": int(pl.sum()),
               "rules": {k: h[k] for k in ("version", "temperature", "max_plies", "regicide",
                                           "king_safety", "cap_by_material",
                                           "cap_draw_margin", "resign_lead",
                                           "resign_plies")},
               "mean_plies": float(pl.mean()) if len(pl) else 0.0,
               "black": int((res == 1).sum()), "white": int((res == -1).sum()),
               "draw": int((res == 0).sum()),
               "value_mean": float(v.mean()), "value_sd": float(v.std()),
               "value_min": float(v.min()), "value_max": float(v.max()),
               "scored_fraction_mean": float(np.mean(frac))}
        for g in gs:
            out.setdefault("terminations", {})
            out["terminations"][g["termination"]] = out["terminations"].get(g["termination"], 0) + 1
        return out
