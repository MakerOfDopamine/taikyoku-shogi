#!/bin/sh
# One-shot setup for an nnplay bundle: GPU runtime, build, proof. Safe to re-run.
#
#   sh setup.sh [PYTHON]
#
# From a notebook, pass sys.executable as PYTHON. It is only used to *download* the runtime
# and to look for CUDA libraries -- nothing is installed into its environment, so it does
# not matter which environment it belongs to.
#
# Why the runtime lives in ./ortlib: onnxruntime and onnxruntime-gpu install into the same
# package directory, so whichever went in last wins, and notebook package managers --
# marimo's sandbox among them -- reinstall the CPU wheel whenever a cell imports
# onnxruntime. Every earlier version of this script installed into the notebook's
# environment and lost to that. Here the GPU runtime goes into a directory nothing else
# knows about, and nnplay finds it by a path relative to itself.
#
#   ORT_SPEC   pip requirement for the runtime  (default: onnxruntime-gpu>=1.22)
#   FORCE=1    download the runtime again even if ortlib/ already has one
set -u
cd "$(dirname "$0")" || exit 1
HERE=$(pwd)
PY=${1:-python3}
ORT_SPEC=${ORT_SPEC:-onnxruntime-gpu>=1.22}
LIB=$HERE/ortlib
CAPI=$LIB/onnxruntime/capi
LOG=$HERE/setup.log
: > "$LOG"

say() { printf '\n== %s\n' "$*"; }
die() { printf '\nSETUP FAILED: %s\n' "$*"; exit 1; }

real_lib() {   # the fully versioned libonnxruntime.so.X.Y.Z, never one of our aliases
    for f in "$CAPI"/libonnxruntime.so.*.*.*; do
        if [ -f "$f" ] && [ ! -L "$f" ]; then echo "$f"; return 0; fi
    done
    return 1
}
have_gpu_runtime() { [ -f "$CAPI/libonnxruntime_providers_cuda.so" ] && real_lib >/dev/null; }

attempt() {
    rm -rf "$LIB"
    printf '   %s ... ' "$*"
    if "$@" >>"$LOG" 2>&1; then echo ok; return 0; fi
    echo no
    return 1
}

# ---- 1. the GPU runtime, into ./ortlib ---------------------------------------------------
if have_gpu_runtime && [ "${FORCE:-0}" != 1 ]; then
    say "GPU runtime already in ortlib/: $(basename "$(real_lib)")   (FORCE=1 to replace)"
else
    say "downloading $ORT_SPEC into ortlib/ -- no Python environment is modified"
    # --no-deps: only the shared libraries inside the wheel are used, never its Python
    # half. Any pip will do, since --target writes here rather than into its environment.
    got=""
    if attempt "$PY" -m pip install --no-deps --target "$LIB" "$ORT_SPEC"; then got=1
    elif command -v uv >/dev/null 2>&1 \
         && attempt uv pip install --python "$PY" --no-deps --target "$LIB" "$ORT_SPEC"; then got=1
    elif attempt python3 -m pip install --no-deps --target "$LIB" "$ORT_SPEC"; then got=1
    elif command -v pip3 >/dev/null 2>&1 && attempt pip3 install --no-deps --target "$LIB" "$ORT_SPEC"; then got=1
    elif command -v pip  >/dev/null 2>&1 && attempt pip  install --no-deps --target "$LIB" "$ORT_SPEC"; then got=1
    fi
    [ -n "$got" ] || { tail -25 "$LOG"; die "no pip or uv could download $ORT_SPEC (full log: setup.log)"; }
    have_gpu_runtime || die "the downloaded runtime has no libonnxruntime_providers_cuda.so, so it is a
                CPU-only build. ORT_SPEC must name onnxruntime-gpu."
    echo "   runtime: $(basename "$(real_lib)")"
fi

# The two names the wheel does not provide -- the SONAME nnplay asks for, and the bare
# name -l wants -- made *inside* capi, beside the provider libraries. ONNX Runtime loads
# its providers from the directory its own libonnxruntime was loaded from; aliasing it into
# the bundle directory instead is what produced
#   Failed to load library .../libonnxruntime_providers_shared.so
REAL=$(basename "$(real_lib)")
ln -sf "$REAL" "$CAPI/libonnxruntime.so"
ln -sf "$REAL" "$CAPI/libonnxruntime.so.1"
for f in "$HERE"/libonnxruntime*.so*; do [ -L "$f" ] && rm -f "$f"; done   # earlier setups' aliases

# ---- 2. CUDA libraries --------------------------------------------------------------------
say "looking for CUDA libraries"
CANDIDATES=$(
    for P in "$PY" python3 /usr/local/bin/python3 /usr/bin/python3; do
        { command -v "$P" >/dev/null 2>&1 || [ -x "$P" ]; } || continue
        "$P" - 2>/dev/null <<'PYEOF'
import glob, os
try:
    import nvidia
    for r in nvidia.__path__:
        for d in glob.glob(os.path.join(r, "*", "lib")):
            print(d)
except Exception:
    pass
PYEOF
    done
    for d in /usr/local/lib/python3*/site-packages/nvidia/*/lib \
             /usr/local/lib/python3*/dist-packages/nvidia/*/lib \
             /usr/lib/python3*/site-packages/nvidia/*/lib \
             /usr/lib/python3*/dist-packages/nvidia/*/lib \
             "${HOME:-/nonexistent}"/.local/lib/python3*/site-packages/nvidia/*/lib \
             /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib; do
        [ -d "$d" ] && echo "$d"
    done
)
RPATHS=""
LDPATH=""
for d in $(printf '%s\n' "$CANDIDATES" | awk 'NF && !seen[$0]++'); do
    ls "$d"/*.so* >/dev/null 2>&1 || continue
    RPATHS="$RPATHS -Wl,-rpath,$d"
    LDPATH="${LDPATH:+$LDPATH:}$d"
    echo "   $d"
done
[ -n "$RPATHS" ] || echo "   none in the usual places; relying on the system loader path"

# ---- 3. build -----------------------------------------------------------------------------
say "compiling nnplay"
# --disable-new-dtags emits DT_RPATH, not DT_RUNPATH: the CUDA provider is dlopened by
# libonnxruntime.so, and only DT_RPATH is inherited that far down.
# shellcheck disable=SC2086
gcc -O3 -march=native -pthread -o nnplay nnplay.c -I. -L"$CAPI" -lonnxruntime -lm -ldl \
    -Wl,--disable-new-dtags -Wl,-rpath,'$ORIGIN/ortlib/onnxruntime/capi' $RPATHS >>"$LOG" 2>&1 \
    || { tail -30 "$LOG"; die "gcc failed (full log: setup.log)"; }
echo "   ok"

# ---- 4. prove it ------------------------------------------------------------------------
say "self-check: the CUDA provider, the exported weights, and this GPU's throughput"
./nnplay selfcheck
rc=$?
if [ "$rc" -ne 0 ]; then
    echo
    if command -v ldd >/dev/null 2>&1; then
        missing=$(LD_LIBRARY_PATH="$LDPATH${LDPATH:+:}$CAPI" ldd "$CAPI/libonnxruntime_providers_cuda.so" 2>&1 \
                  | grep "not found")
        if [ -n "$missing" ]; then
            echo "   libraries the CUDA provider cannot find:"
            printf '%s\n' "$missing" | sed 's/^[[:space:]]*/     /'
        fi
    fi
    die "the self-check did not pass (exit $rc) -- see above"
fi

cat <<'DONE'

== READY

nnplay is built, runs on CUDA, and reproduces the exported weights. Use the
--sub-batch figure the self-check printed to skip the probe on every run.

  ./nnplay play --games 20 --max-plies 400 --sub-batch 32 --temperature 0.3 \
                --progress-every 25 --out selfplay.tkn
  ./nnplay elo  --games 200 --sub-batch 32 --temperature 0.3
  python3 unpack_nn.py selfplay.tkn --show 3

README.md has the same commands as notebook cells.
DONE
