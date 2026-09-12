#!/bin/sh
# Assembles the smallest tree that can run self-play somewhere else -- a remote notebook,
# a rented GPU box -- as one directory with one entry point: setup.sh.
#
#   ./bundle.sh [DIR] [--with-fp32] [--with-tests] [--with-export]
#
# The default bundle needs no node, no torch and no checkpoint.pt on the far end: the rules
# are baked into tables.h and the weights into net_fp16.onnx. The GPU runtime is *not*
# shipped -- it is hundreds of MB -- setup.sh downloads it into the bundle itself.
set -e
cd "$(dirname "$0")"

OUT=""
TESTS=0
EXPORT=0
FP32=0
for a in "$@"; do
    case "$a" in
        --with-fp32)   FP32=1 ;;
        --with-tests)  TESTS=1 ;;
        --with-export) EXPORT=1 ;;
        --*) echo "unknown option $a"; exit 1 ;;
        *) OUT="$a" ;;
    esac
done
[ -n "$OUT" ] || OUT=nnplay-bundle

[ -f net_manifest.json ] || { echo "missing net_manifest.json -- run export_onnx.py first"; exit 1; }
[ -f third_party/onnxruntime/include/onnxruntime_c_api.h ] \
    || { echo "missing third_party/onnxruntime -- see the header of build_nn.sh"; exit 1; }

rm -rf "$OUT"
mkdir -p "$OUT"
# engine: nnplay.c includes tky.c, which includes tables.h. That is the whole rules stack.
cp nnplay.c tky.c tables.h "$OUT"/
# The header only. The C API is versioned inside it and newer runtimes still serve older
# API versions, so it works with whichever onnxruntime-gpu setup.sh downloads.
cp third_party/onnxruntime/include/onnxruntime_c_api.h "$OUT"/
cp net_fp16.onnx net_fp16_raw.onnx "$OUT"/
if [ "$FP32" = 1 ]; then cp net_fp32.onnx "$OUT"/; fi
# the Elo baseline, when one has been exported
if [ -f baseline_fp16.onnx ]; then
    cp baseline_fp16.onnx "$OUT"/
    if [ "$FP32" = 1 ] && [ -f baseline_fp32.onnx ]; then cp baseline_fp32.onnx "$OUT"/; fi
else
    echo "  (no baseline_fp16.onnx: 'nnplay elo' in this bundle will need --baseline random)"
fi
cp unpack.py unpack_nn.py tables.json fuse.py net_manifest.json "$OUT"/
cp bundle_setup.sh "$OUT"/setup.sh
chmod +x "$OUT"/setup.sh

if [ "$TESTS" = 1 ]; then
    cp test_nnplay.py model.py checkpoint.pt "$OUT"/
    cp shard_0000.tkp "$OUT"/ 2>/dev/null || echo "  (no shard_0000.tkp: the encode and torch checks need one)"
fi
if [ "$EXPORT" = 1 ]; then
    cp export_onnx.py export_rewrite.py model.py checkpoint.pt "$OUT"/
    if [ -f baseline.pt ]; then cp baseline.pt "$OUT"/; fi
fi

cat > "$OUT/README.md" <<'EOF'
# nnplay bundle

Upload this directory, run `setup.sh` once, then use `nnplay`. `setup.sh` downloads the
GPU runtime into `ortlib/` inside this directory, compiles, and proves the result: CUDA
loads, the weights reproduce, and it prints this GPU's throughput. It stops with the
reason if any of that fails, and it is safe to run again.

## From a notebook

Each cell uses only names starting with `_`, which marimo keeps private to the cell, so
none of them clash. Change `/marimo/nnplay-bundle` if yours lives elsewhere.

**Do not `import onnxruntime` in the notebook.** A notebook package manager that sees the
import may install the CPU `onnxruntime` wheel to satisfy it. nnplay no longer uses the
notebook's packages at all, but there is no reason to invite it.

Setup, once:

```python
import subprocess as _sp_setup, sys as _sys_setup
_setup = _sp_setup.Popen(["sh", "/marimo/nnplay-bundle/setup.sh", _sys_setup.executable],
                         stdout=_sp_setup.PIPE, stderr=_sp_setup.STDOUT, text=True, bufsize=1)
for _setup_line in _setup.stdout:
    print(_setup_line, end="", flush=True)
print("setup exit code:", _setup.wait())
```

Self-play, with live output:

```python
import subprocess as _sp_play
_play = _sp_play.Popen(
    ["./nnplay", "play", "--games", "20", "--max-plies", "400", "--sub-batch", "32",
     "--temperature", "0.3", "--progress-every", "25", "--out", "selfplay.tkn"],
    cwd="/marimo/nnplay-bundle", stdout=_sp_play.PIPE, stderr=_sp_play.STDOUT,
    text=True, bufsize=1)
for _play_line in _play.stdout:
    print(_play_line, end="", flush=True)
print("play exit code:", _play.wait())
```

Elo against the baseline (add `"--baseline", "random"` if the bundle has no
`baseline_fp16.onnx`):

```python
import subprocess as _sp_elo
_elo = _sp_elo.Popen(
    ["./nnplay", "elo", "--games", "200", "--sub-batch", "32", "--temperature", "0.3"],
    cwd="/marimo/nnplay-bundle", stdout=_sp_elo.PIPE, stderr=_sp_elo.STDOUT,
    text=True, bufsize=1)
for _elo_line in _elo.stdout:
    print(_elo_line, end="", flush=True)
print("elo exit code:", _elo.wait())
```

Use the `--sub-batch` the self-check printed.

## If it stops

`setup.sh` names the problem and the fix. The two common ones:

- **"CPU-only build"** means a CPU runtime was downloaded; `ORT_SPEC` must name
  `onnxruntime-gpu`.
- **"libraries the CUDA provider cannot find"** lists them. They come from the CUDA
  toolkit or from the `nvidia-*` wheels a torch install pulls in; `setup.sh` searches both.

nnplay refuses to run on the CPU unless given `--allow-cpu`, because a silent fallback is
~100x slower and looks like a hang.

If the self-check reports a value mismatch rather than a missing library, this runtime
does not reproduce the fused graph; re-fuse it here from the pre-fusion copy:

    pip install onnx && python3 fuse.py net_fp16_raw.onnx net_fp16.onnx && sh setup.sh
EOF

echo "$OUT: $(du -sh "$OUT" | cut -f1), $(find "$OUT" -type f | wc -l) files"
ls -la "$OUT" | awk 'NR>1 && $5 != "" && $9 != "." && $9 != ".." {printf "  %9.1f KB  %s\n", $5/1024, $9}'
echo
echo "tar it up:  tar czf $OUT.tgz $OUT"
