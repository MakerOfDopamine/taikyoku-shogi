#!/bin/bash
# Full validation. Every layer is checked against the layer below it:
#   game.js  <- probe.js / xcheck.js -> tky_ref (straight transliteration)
#   tky_ref  <- difftest.sh          -> tky     (optimised)
#   tky      <- tky_reader.py        -> the corpus format
set -u
cd "$(dirname "$0")"
SC="$(mktemp -d)"; trap 'rm -rf "$SC"' EXIT
T="${THREADS:-4}"
fails=0
step() { printf '\n== %s\n' "$1"; }
ok()   { [ "$1" -eq 0 ] && echo "   PASS" || { echo "   FAIL"; fails=$((fails+1)); }; }

step "known anchors (CLAUDE.md section 10)"
out=$(./tky perft 2)
echo "$out"
echo "$out" | grep -q "perft(1) = 262"   && echo "$out" | grep -q "perft(2) = 68614" \
  && echo "$out" | grep -q "0:68 9:194"; ok $?

step "every piece type, both colours, alone on the board vs game.js"
node probe.js lone | tail -2; ok ${PIPESTATUS[0]}

step "dense random boards vs game.js"
for s in 1 2 3; do node probe.js random $s 30 | tail -1; done; ok $?

step "seeded playouts vs game.js (full sorted move list, material and turn every ply)"
for s in 1 5 11 42; do
    ./tky trace $s 200 1 > "$SC/t$s.txt"
    node --max-old-space-size=4096 xcheck.js "$SC/t$s.txt" | tail -1
done; ok $?

step "optimised engine vs straight transliteration"
./difftest.sh 30 250; ok $?

step "forced King capture: fast detector vs reference, cache, and the rule itself"
./tky kingtest 200; ok $?

step "corpus round trip: python reader rebuilds the engine's board every ply"
for s in 1 2 3; do ./tky trace $s 600 > "$SC/r$s.txt"; done
python3 - "$SC"/r1.txt "$SC"/r2.txt "$SC"/r3.txt <<'PY'
import sys, numpy as np
from tky_reader import INIT_BOARD, EMPTY_IDX, apply_move, _MATVAL
bad = plies = 0
for path in sys.argv[1:]:
    b = INIT_BOARD.copy()
    for line in open(path):
        t = line.split()
        if t[0] == 'PLY':
            kv = dict(x.split('=') for x in t[2:])
            sign = np.where(((b >> 1) & 1) == 1, 1, -1)
            if int((_MATVAL[b.astype(np.int64)] * sign).sum()) != int(kv['mat']) \
               or int(((b >> 2) != EMPTY_IDX).sum()) != int(kv['pieces']):
                print(f"  {path} ply {t[1]}: reader board disagrees with engine"); bad += 1
            plies += 1
        elif t[0] == 'PICK':
            o, m = int(t[1]), int(t[2])
            apply_move(b, ((m // 1296) << 28) | (o << 17) | ((m % 1296) << 6))
print(f"   {plies} plies replayed, {bad} mismatched")
sys.exit(1 if bad else 0)
PY
ok $?

step "generate a corpus and verify the stored labels against a fresh replay"
rm -rf "$SC/corpus"; mkdir -p "$SC/corpus"
./tky gen --games 40 --threads "$T" --out "$SC/corpus" --max-plies 400 --save-percentage 100 2>&1 | tail -1
python3 tky_reader.py "$SC/corpus" | tail -3; ok ${PIPESTATUS[0]}

step "position sampling: 5% selection, label balance, and snapshot == replay"
rm -rf "$SC/sg" "$SC/sp"; mkdir -p "$SC/sg" "$SC/sp"
./tky gen --games 400 --threads "$T" --out "$SC/sg" 2>&1 | grep -E "selection:" | sed 's/^ */   /'
./tky gen --games 400 --threads "$T" --out "$SC/sp" --format positions >/dev/null 2>&1
python3 - "$SC" <<'PY2'
import sys, numpy as np
from tky_reader import Corpus, PositionCorpus, EMPTY_IDX
SC = sys.argv[1]
pc, gc = PositionCorpus(f"{SC}/sp"), Corpus(f"{SC}/sg")
rep = [(p.ply, p.board.copy(), p.material, p.n_legal, p.result, p.termination) for p in gc.positions()]
bad = 0
if len(rep) != len(pc):
    print(f"   selection differs: .tky {len(rep)} vs .tkp {len(pc)}"); bad += 1
else:
    for i, (ply, b, mat, nl, res, term) in enumerate(rep):
        r = pc.records[i]
        if (not np.array_equal(b, r["board"])
                or (int(r["ply"]), int(r["nmoves"]), int(r["result"]), int(r["termination"])) != (ply, nl, res, term)
                or abs(float(r["mat"]) * pc.header.mat_unit_pawns - mat) > 1e-9):
            bad += 1
    occ = ((pc.records["board"] >> 2) != EMPTY_IDX).sum(axis=1)
    if not (occ == pc.records["n_occupied"]).all():
        print("   n_occupied field disagrees with the stored board"); bad += 1
lab = pc.labels()
h, _ = np.histogram(lab, bins=16, range=(-1, 1))
spread = h.max() / max(1, h.min())
rate = 100 * len(pc) / pc.positions_sampled
print(f"   {len(rep)} snapshots vs replay: {bad} mismatched; kept {rate:.2f}%; "
      f"label bin spread {spread:.2f}x")
if spread > 1.6:
    print("   label distribution is not flat enough"); bad += 1
if not (3.5 < rate < 6.5):
    print("   kept fraction is off target"); bad += 1
sys.exit(1 if bad else 0)
PY2
ok $?

step "unpack.py: both formats agree, and agree with tky_reader.py"
python3 - "$SC" <<'PY3'
import sys, numpy as np
from unpack import unpack, material_plane, piece_id, occupied
from tky_reader import Corpus
SC = sys.argv[1]
g, p = list(unpack(f"{SC}/sg", canonical=False)), list(unpack(f"{SC}/sp", canonical=False))
bad = 0
if len(g) != len(p):
    print(f"   entry counts differ: .tky {len(g)} vs .tkp {len(p)}"); bad += 1
else:
    for i, ((bg, ig), (bp, ip)) in enumerate(zip(g, p)):
        if not np.array_equal(bg, bp): bad += 1; break
        if any(getattr(ig, f) != getattr(ip, f) for f in
               ("game_index", "seed", "ply", "plies_total", "side_to_move", "material",
                "label", "label_bin", "n_legal", "n_occupied", "result", "termination")):
            bad += 1; break
rd = [(x.board.copy(), x.material, x.ply) for x in Corpus(f"{SC}/sg").positions()]
for (bg, ig), (br, mat, ply) in zip(g, rd):
    if not np.array_equal(bg.reshape(-1), br) or abs(ig.material - mat) > 1e-9 or ig.ply != ply:
        bad += 1; break
err = max(abs(material_plane(b).sum() - i.material) for b, i in g)
if err > 0.0626:
    print(f"   material recomputed from the board is off by {err:.4f} pawns"); bad += 1
from unpack import embedding_index, NUM_EMBEDDING
e = np.concatenate([embedding_index(b).ravel() for b, _ in g[:200]])
if e.min() < 0 or e.max() >= NUM_EMBEDDING:
    print(f"   embedding index out of [0,{NUM_EMBEDDING}): {e.min()}..{e.max()}"); bad += 1
# canonicalisation must be exact: the engine's move set for the rotated position has to be
# the rotation of the move set for the original
import subprocess
from unpack import canonical_board, rot180_move, EMPTY_IDX as _E
def _probe(board, turn):
    flat = board.reshape(-1)
    lines = [f"turn {turn}"] + [f"{sq} {int(flat[sq])>>2} {(int(flat[sq])>>1)&1} {int(flat[sq])&1}"
                                for sq in np.flatnonzero((flat >> 2) != _E)]
    out = subprocess.run(["./tky", "probe"], input="\n".join(lines) + "\n",
                         capture_output=True, text=True).stdout.strip().split("\n")
    return {tuple(map(int, l.split(":"))) for l in out[1:] if l}
ncanon = 0
for b, i in g[::max(1, len(g) // 8)][:8]:
    # force the rotation regardless of whose turn it is, and require the engine's move set
    # for the rotated position to be exactly the rotation of the original's
    A = _probe(b, i.side_to_move)
    B = _probe(canonical_board(b, 0), 1 - i.side_to_move)
    if B != {rot180_move(o, m) for o, m in A}:
        print(f"   rotated form of ply {i.ply} has a different move set"); bad += 1
    # and canonical_board must leave a black-to-move position alone
    if not np.array_equal(canonical_board(b, 1), b):
        print(f"   canonical_board altered a black-to-move position at ply {i.ply}"); bad += 1
    ncanon += 1
b0, i0 = next(iter(unpack(f"{SC}/sg", selected_only=False, canonical=False)))
ids = piece_id(b0)
if b0.shape != (36, 36) or int(occupied(b0).sum()) != 804 or i0.n_legal != 262:
    print(f"   start position wrong: {b0.shape}, {int(occupied(b0).sum())} pieces, {i0.n_legal} moves"); bad += 1
if [tuple(map(int, v)) for v in np.argwhere((ids == 1000) & ((b0 >> 1 & 1) == 1))] != [(0, 17)]:
    print("   black King is not at (0,17)"); bad += 1
print(f"   {len(g)} entries via both formats, {bad} mismatched; "
      f"material round-trip max error {err:.4f} pawns; "
      f"{ncanon} rotations verified against the engine")
sys.exit(1 if bad else 0)
PY3
ok $?

step "shard fusing: fused corpus == the shards it replaced"
mkdir -p "$SC/fz" "$SC/fk" "$SC/pz" "$SC/pk"
./tky gen --games 200 --threads 5 --max-plies 300 --seed 4242 --out "$SC/fz"        >/dev/null 2>&1
./tky gen --games 200 --threads 5 --max-plies 300 --seed 4242 --out "$SC/fk" --keep-shards >/dev/null 2>&1
./tky gen --games 150 --threads 4 --max-plies 300 --seed 4242 --format positions --out "$SC/pz"        >/dev/null 2>&1
./tky gen --games 150 --threads 4 --max-plies 300 --seed 4242 --format positions --out "$SC/pk" --keep-shards >/dev/null 2>&1
python3 - "$SC" <<'PY4'
import glob, hashlib, sys
import numpy as np
from unpack import unpack, read_header, shard_paths

SC = sys.argv[1]
bad = 0

for kind, fused_dir, kept_dir, ext in (("games", f"{SC}/fz", f"{SC}/fk", "tky"),
                                       ("positions", f"{SC}/pz", f"{SC}/pk", "tkp")):
    # the fused payload must be exactly the shard payloads, in worker order
    def payload(p):
        with open(p, "rb") as f:
            f.seek(128)
            return f.read()
    one = hashlib.sha256(payload(f"{fused_dir}/corpus.{ext}")).hexdigest()
    many = hashlib.sha256()
    parts = sorted(glob.glob(f"{kept_dir}/shard_*.{ext}"))
    for p in parts:
        many.update(payload(p))
    if one != many.hexdigest():
        print(f"   {kind}: fused payload differs from the concatenated shards"); bad += 1

    # and the header counts must be the sums, with the first-game index reset
    hf = read_header(shard_paths(fused_dir)[0])
    tc = ts = 0
    for p in shard_paths(kept_dir):
        h = read_header(p); tc += h.count; ts += h.second
    if (hf.count, hf.second, hf.first_game) != (tc, ts, 0):
        print(f"   {kind}: fused header {hf.count}/{hf.second}/{hf.first_game} "
              f"but shards sum to {tc}/{ts}"); bad += 1

    # readers must see the same thing either way, game indices included
    a = list(unpack(fused_dir, canonical=False))
    b = list(unpack(kept_dir, canonical=False))
    if len(a) != len(b):
        print(f"   {kind}: {len(a)} positions fused vs {len(b)} sharded"); bad += 1
    for (ba, ia), (bb, ib) in zip(a, b):
        if not np.array_equal(ba, bb) or ia.game_index != ib.game_index or ia.ply != ib.ply:
            print(f"   {kind}: position {ia.game_index}/{ia.ply} differs"); bad += 1
            break
    gi = [i.game_index for _, i in a]
    if gi != sorted(gi) or gi[0] != 0:
        print(f"   {kind}: fused game indices are not the original global order"); bad += 1
    print(f"   {kind}: {len(parts)} shards -> 1 file, {len(a)} positions, identical payload")

sys.exit(1 if bad else 0)
PY4
ok $?

step "value targets: result_stm is the side to move's own result"
mkdir -p "$SC/vt"
./tky gen --games 120 --threads 4 --max-plies 1200 --seed 20260909 --format positions \
          --save-percentage 100 --out "$SC/vt" >/dev/null 2>&1
python3 - "$SC" <<'PY5'
import sys
import numpy as np
from unpack import unpack, piece_id, colour

SC = sys.argv[1]
bad = 0

# 1. result_stm against ground truth recomputed from the absolute frame. result is
#    0 white won / 1 black won / 2 draw and side_to_move is 1 black / 0 white, so the side
#    to move won exactly when the two are equal.
absolute = [(i.result, i.side_to_move, i.ply) for _, i in unpack(f"{SC}/vt", canonical=False)]
canon = [(i.result_stm, i.side_to_move) for _, i in unpack(f"{SC}/vt")]
nw = nb = wrong = pin = parity = 0
for (res, stm, ply), (rstm, cstm) in zip(absolute, canon):
    want = 0.0 if res == 2 else (1.0 if res == stm else -1.0)
    wrong += rstm != want
    pin += cstm != 1                       # canonicalising must leave black to move
    parity += stm != 1 - (ply & 1)         # black moves first
    nw += stm == 0; nb += stm == 1
if wrong or pin or parity:
    print(f"   result_stm wrong {wrong}, side_to_move not pinned {pin}, ply parity {parity}")
    bad += 1
if not nw or not nb:
    print(f"   only one side to move present ({nw} white, {nb} black)"); bad += 1

# 2. and the stored result itself against the board: after a royal capture the loser's King
#    is gone. Only id 1000 is royal here -- the Crown Prince is not (CLAUDE.md section 9).
term = res_bad = stm_bad = 0
for board, i in unpack(f"{SC}/vt", canonical=False):
    if i.ply != i.plies_total or i.termination != 0:
        continue
    term += 1
    ids, col = piece_id(board), colour(board)
    kb = int(((ids == 1000) & (col == 1)).sum())
    kw = int(((ids == 1000) & (col == 0)).sum())
    if (kb, kw) not in ((1, 0), (0, 1)):
        res_bad += 1
        continue
    winner = 1 if kw == 0 else 0
    res_bad += i.result != winner
    stm_bad += i.side_to_move != 1 - winner     # the loser is the one left to move
if not term:
    print("   no terminal positions to check"); bad += 1
if res_bad or stm_bad:
    print(f"   stored result disagrees with the board {res_bad}, side to move {stm_bad}"); bad += 1
tv = {i.result_stm for _, i in unpack(f"{SC}/vt") if i.ply == i.plies_total and i.termination == 0}
if tv != {-1.0}:
    print(f"   canonical result_stm at terminal positions is {tv}, want {{-1.0}}"); bad += 1

print(f"   {len(canon)} positions ({nw} white to move, {nb} black), "
      f"{term} terminal checked against King counts")
sys.exit(1 if bad else 0)
PY5
ok $?

printf '\n== %d failing step(s)\n' "$fails"
exit $((fails > 0))
