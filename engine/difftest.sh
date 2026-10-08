#!/bin/bash
# Diffs the optimised engine against the straight transliteration (tky_ref) on positions
# taken from real playouts and on random boards. Any divergence is a rules bug.
set -u
cd "$(dirname "$0")"
SC="${TMPDIR:-/tmp}/tkydiff.$$"; mkdir -p "$SC"; trap 'rm -rf "$SC"' EXIT
fail=0; n=0

# 1. traces: identical seeded playouts must agree ply for ply (digest covers every move)
for s in $(seq 1 "${1:-40}"); do
    ./tky     trace "$s" "${2:-150}" 0 1 > "$SC/a" 2>&1
    ./tky_ref trace "$s" "${2:-150}" 0 1 > "$SC/b" 2>&1
    if ! cmp -s "$SC/a" "$SC/b"; then echo "TRACE MISMATCH seed=$s"; diff "$SC/a" "$SC/b" | head -6; fail=1; fi
    n=$((n+1))
done
echo "traces compared: $n"

# 2. random positions through probe
python3 - "$SC" <<'PY'
import random, subprocess, sys, json
sc = sys.argv[1]
T = json.load(open('tables.json'))
IDS = T['ids']
special = [276,286,287,288,289,290,291,292,293,294,295,297,298,299,300,1000]
bad = 0
for trial in range(300):
    rnd = random.Random(trial)
    used = set(); cells = []
    for _ in range(rnd.randint(10, 500)):
        sq = rnd.randrange(1296)
        if sq in used: continue
        used.add(sq)
        cells.append((sq, rnd.randrange(1, len(IDS)), rnd.randrange(2), rnd.randrange(2)))
    turn = rnd.randrange(2)
    for pid in special:
        sq = rnd.randrange(1296)
        while sq in used: sq = rnd.randrange(1296)
        used.add(sq); cells.append((sq, IDS.index(pid), turn, 0))
    stdin = f"turn {turn}\n" + "\n".join(f"{a} {b} {c} {d}" for a,b,c,d in cells) + "\n"
    a = subprocess.run(['./tky','probe'], input=stdin, capture_output=True, text=True).stdout
    b = subprocess.run(['./tky_ref','probe'], input=stdin, capture_output=True, text=True).stdout
    if a != b:
        bad += 1
        la, lb = set(a.split('\n')), set(b.split('\n'))
        print(f"PROBE MISMATCH trial={trial}: opt-only={sorted(la-lb)[:6]} ref-only={sorted(lb-la)[:6]}")
        if bad > 3: break
print(f"random positions compared: 300, mismatched: {bad}")
sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && fail=1
exit $fail
