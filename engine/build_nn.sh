#!/bin/sh
# Builds nnplay, the neural-network self-play driver.
#
# Kept separate from build.sh because it needs the ONNX Runtime GPU C API in
# third_party/onnxruntime; the corpus generator does not. Fetch it with:
#
#   V=$(cat third_party/onnxruntime/VERSION_NUMBER 2>/dev/null || echo 1.20.1)
#   curl -L -o /tmp/ort.tgz \
#     https://github.com/microsoft/onnxruntime/releases/download/v$V/onnxruntime-linux-x64-gpu-$V.tgz
#   mkdir -p third_party && tar xzf /tmp/ort.tgz -C third_party \
#     && mv third_party/onnxruntime-linux-x64-gpu-$V third_party/onnxruntime
#
# The CUDA provider needs libcudnn.so.9 and the cuBLAS/cuFFT/cuRAND runtimes. Rather than
# demand LD_LIBRARY_PATH at every invocation, their directories are resolved here and
# baked into the binary as an rpath -- including the ones torch ships in its own wheels,
# which is where they live on a machine with no system CUDA install. Rebuild after moving
# the tree or changing the torch install.
set -e
cd "$(dirname "$0")"

ORT=third_party/onnxruntime
[ -d "$ORT" ] || { echo "missing $ORT -- see the header of this script"; exit 1; }

RPATH="-Wl,--disable-new-dtags -Wl,-rpath,\$ORIGIN/$ORT/lib"
for d in $(python3 - <<'PY'
import glob, os
try:
    import nvidia
    for root in nvidia.__path__:
        for lib in sorted(glob.glob(os.path.join(root, "*", "lib"))):
            if glob.glob(os.path.join(lib, "*.so*")):
                print(lib)
except ImportError:
    pass
PY
); do
    RPATH="$RPATH -Wl,-rpath,$d"
done

node gen_tables.js
# shellcheck disable=SC2086
gcc -O3 -march=native -pthread -o nnplay nnplay.c -I$ORT/include -L$ORT/lib -lonnxruntime -lm -ldl $RPATH
echo "built $(pwd)/nnplay"
echo "  ort      $(cat $ORT/VERSION_NUMBER)"
echo "  rpath    $(echo "$RPATH" | tr ' ' '\n' | sed 's/-Wl,-rpath,/           /' | tail -n +2 | head -4)"
