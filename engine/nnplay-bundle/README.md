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
