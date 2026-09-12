#!/bin/sh
# Builds nnplay on the far end. Run once, from inside this directory.
set -e
cd "$(dirname "$0")"

pip install "onnxruntime-gpu>=1.22"

CAPI=$(python3 -c "import onnxruntime, os; print(os.path.dirname(onnxruntime.__file__))")/capi
[ -e "$CAPI/libonnxruntime_providers_cuda.so" ] || cat <<'WARN'

WARNING: no libonnxruntime_providers_cuda.so in the wheel -- this is the CPU onnxruntime
         package, not onnxruntime-gpu, and nnplay will fall back to the CPU. The two
         install into the same directory, so fix it with:
             pip uninstall -y onnxruntime onnxruntime-gpu && pip install "onnxruntime-gpu>=1.22"

WARN

# Every provider library, not just libonnxruntime itself. ONNX Runtime finds its provider
# shims next to the libonnxruntime.so it was loaded from, so linking that one alone into
# this directory makes it look here and find nothing:
#   Failed to load library .../libonnxruntime_providers_shared.so
# The C API tarball ships all of them in one directory; that co-location is the contract.
ln -sf "$CAPI"/*.so* .
# and the two names the wheel does not provide: the SONAME the binary asks for, and the
# bare name -l wants at link time.
SO=$(ls "$CAPI"/libonnxruntime.so.[0-9]* | head -1)
ln -sf "$SO" libonnxruntime.so
ln -sf "$SO" libonnxruntime.so.1

# cuDNN and cuBLAS, wherever they are -- on a box with no system CUDA they come from the
# nvidia wheels torch pulls in. Baked in so no LD_LIBRARY_PATH is needed at run time.
NVRPATH=$(python3 - <<'PYEOF'
import glob, os
try:
    import nvidia
    dirs = [d for r in nvidia.__path__ for d in sorted(glob.glob(os.path.join(r, "*", "lib")))
            if glob.glob(os.path.join(d, "*.so*"))]
except ImportError:
    dirs = []
print(" ".join("-Wl,-rpath," + d for d in dirs))
PYEOF
)

# --disable-new-dtags emits DT_RPATH rather than DT_RUNPATH: the CUDA provider is dlopened
# by libonnxruntime.so, and only DT_RPATH is inherited that far.
gcc -O3 -march=native -pthread -o nnplay nnplay.c -I. -L. -lonnxruntime -lm -ldl \
    -Wl,--disable-new-dtags -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$CAPI" $NVRPATH

# Does this runtime actually reproduce the values the bundle was built with? The graphs
# carry a contrib op, so this is checked rather than assumed.
python3 - <<'CHECK'
import subprocess, sys
sys.path.insert(0, ".")
from unpack import INIT_BOARD, embedding_index, canonical_board

board = embedding_index(canonical_board(INIT_BOARD.reshape(36, 36), 1)).reshape(-1)
out = subprocess.run(["./nnplay", "eval", "--precision", "fp16", "--quiet"],
                     input=" ".join(map(str, board)), capture_output=True, text=True)
if out.returncode:
    print(out.stderr[-1500:]); sys.exit(1)
import json
man = json.load(open("net_manifest.json"))["selfcheck"]
got, want, tol = float(out.stdout.split()[0]), man["start_position_value"], man["tolerance"]
d = abs(got - want)
print(f"\nself-check: start position value {got:+.9f}, expected {want:+.9f}, deviation {d:.2e}")
print("  PASS" if d < tol else
      "  FAIL -- this runtime does not reproduce the bundled graph.\n"
      "  Fix it with:  pip install onnx && python3 fuse.py net_fp16_raw.onnx net_fp16.onnx\n"
      "  then run ./setup.sh again.")
sys.exit(0 if d < tol else 1)
CHECK

echo
echo "built ./nnplay -- try:"
echo "  ./nnplay play --games 4 --max-plies 400 --progress-every 25 --out selfplay.tkn"
echo "  python3 unpack_nn.py selfplay.tkn --show 3"
