# nnplay bundle

Self-play for Taikyoku Shogi, packaged to run somewhere with a GPU. Built from the engine
directory of the parent repo; see its README for what any of this means.

    ./setup.sh                                              # pip install + gcc
    ./nnplay play --games 4 --max-plies 400 --out sp.tkn
    python3 unpack_nn.py sp.tkn --show 3

**Any onnxruntime-gpu >= 1.22 works.** The graphs carry `com.microsoft.Attention`, a
contrib op, so in principle they are tied to the runtime that defines them. In practice
they are not: the graph these were fused with (1.20.1) was loaded and run by
onnxruntime 1.20.0, 1.22.0, 1.24.4 and 1.29.0, and fp32 came back bit-identical on all
four. `setup.sh` does not take that on trust -- it runs the start position through the
binary it just built and compares against the value recorded when this bundle was made.

If that check ever fails, `net_fp16_raw.onnx` is the same weights before fusion, plain
ONNX with no contrib ops:

    pip install onnx && python3 fuse.py net_fp16_raw.onnx net_fp16.onnx && ./setup.sh

Re-fusing locally reproduced the value exactly on all four versions tested, so this is a
real substitute and not a degraded fallback.

The floor is 1.22 rather than 1.20.0 because 1.20.0's fp16 `SkipLayerNormalization`
kernel is broken (`Missing Input: norm2.weight`); it is fine from 1.22 on.

Needs CUDA 12 and cuDNN 9. Without a working CUDA provider nnplay warns and falls back to
the CPU, which is roughly 100x slower and not worth starting.

`--sub-batch` is measured on whatever GPU is present at startup; the printed curve is
worth reading once, since it is the only thing that tells you whether the card is
saturated.
