#!/bin/sh
# Regenerates tables.h from game.js, then builds both engines.
set -e
cd "$(dirname "$0")"
node gen_tables.js
gcc -O3 -march=native -flto -pthread -o tky     tky.c     -lm
gcc -O3 -march=native        -pthread -o tky_ref tky_ref.c -lm   # differential oracle
echo "built $(pwd)/tky and tky_ref"
