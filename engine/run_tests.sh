#!/bin/bash
# Full validation. Every layer is checked against the layer below it:
#   game.js  <- probe.js / xcheck.js -> tky_ref (straight transliteration)
#   tky_ref  <- difftest.sh          -> tky     (optimised)
#   tky      <- unpack.py            -> the python board, move application and canonical frame
set -u
cd "$(dirname "$0")"
SC="$(mktemp -d)"; trap 'rm -rf "$SC"' EXIT
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

if [ -x ./nnplay ]; then
    step "nnplay King safety: exact vs the slow detector, the sweep, the sampling distribution"
    ./nnplay safetytest; ok $?
fi

step "unpack.py rebuilds the engine's board every ply, and the canonical frame is exact"
for s in 1 2 3; do ./tky trace $s 600 > "$SC/r$s.txt"; done
python3 - "$SC"/r1.txt "$SC"/r2.txt "$SC"/r3.txt <<'PY'
import subprocess, sys
import numpy as np
from unpack import (INIT_BOARD, EMPTY_IDX, MATVAL, NUM_EMBEDDING, apply_move, canonical_board,
                    embedding_index, rot180_move)

def probe(board, turn):
    lines = [f"turn {turn}"] + [f"{sq} {int(board[sq]) >> 2} {(int(board[sq]) >> 1) & 1} {int(board[sq]) & 1}"
                                for sq in np.flatnonzero((board >> 2) != EMPTY_IDX)]
    out = subprocess.run(["./tky", "probe"], input="\n".join(lines) + "\n",
                         capture_output=True, text=True).stdout.strip().split("\n")
    return {tuple(map(int, l.split(":"))) for l in out[1:] if l}

bad = plies = ncanon = 0
for path in sys.argv[1:]:
    b = INIT_BOARD.copy()
    for line in open(path):
        t = line.split()
        if t[0] == 'PLY':
            kv = dict(x.split('=') for x in t[2:])
            sign = np.where(((b >> 1) & 1) == 1, 1, -1)
            if int((MATVAL[b.astype(np.int64)] * sign).sum()) != int(kv['mat']) \
               or int(((b >> 2) != EMPTY_IDX).sum()) != int(kv['pieces']):
                print(f"  {path} ply {t[1]}: python board disagrees with engine"); bad += 1
            e = embedding_index(b)
            if e.min() < 0 or e.max() >= NUM_EMBEDDING:
                print(f"  embedding index out of [0,{NUM_EMBEDDING}) at ply {t[1]}"); bad += 1
            # the move set of the rotated position must be the rotation of the original's
            if int(t[1]) % 150 == 75:
                turn = int(kv['turn'])
                A = probe(b, turn)
                B = probe(canonical_board(b.reshape(36, 36), 0).reshape(-1), 1 - turn)
                if B != {rot180_move(o, m) for o, m in A}:
                    print(f"  {path} ply {t[1]}: the rotated position has a different move set"); bad += 1
                if not np.array_equal(canonical_board(b.reshape(36, 36), 1).reshape(-1), b):
                    print(f"  canonical_board altered a black-to-move position"); bad += 1
                ncanon += 1
            plies += 1
        elif t[0] == 'PICK':
            o, m = int(t[1]), int(t[2])
            apply_move(b, ((m // 1296) << 28) | (o << 17) | ((m % 1296) << 6))
print(f"   {plies} plies replayed, {bad} mismatched; {ncanon} rotations verified against the engine")
sys.exit(1 if bad else 0)
PY
ok $?

printf '\n== %d failing step(s)\n' "$fails"
exit $((fails > 0))
