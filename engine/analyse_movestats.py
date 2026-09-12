"""Percentiles and shape of the legal-move count per position, by ply.

Reads movestats.bin, written by `tky movestats`. Counts are exact over every position of
every game -- nothing is subsampled.
"""
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "movestats.bin"
raw = open(path, "rb").read()
nply, nbucket, nbins, binw = np.frombuffer(raw, dtype="<u4", count=4)
o = 16
def take(n, dt):
    global o
    a = np.frombuffer(raw, dtype=dt, count=n, offset=o); o += n * np.dtype(dt).itemsize
    return a
cnt, sm, sq, psum = (take(nply, "<u8") for _ in range(4))
mn, mx = take(nply, "<u4"), take(nply, "<u4")
hist = take(nbucket * nbins, "<u8").reshape(nbucket, nbins)

BUCKETW = nply // nbucket if nply % nbucket == 0 else int(np.ceil(nply / nbucket))
centres = (np.arange(nbins) + 0.5) * binw

def pcts(h, qs):
    tot = h.sum()
    if tot == 0: return [np.nan] * len(qs)
    c = np.cumsum(h)
    return [float(centres[np.searchsorted(c, q * tot)]) for q in qs]

QS = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
print(f"legal moves per position, {cnt[0]:,} games, {cnt.sum():,} positions total\n")
print(f"{'ply range':>13} {'positions':>11} {'games':>7} {'pieces':>7} {'mean':>7} "
      + " ".join(f"{'p'+str(int(q*100)):>6}" for q in QS) + f" {'max':>6}")
groups = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 800),
          (800, 1200), (1200, 1600), (1600, 2000), (2000, 2400), (2400, 2800), (2800, nply)]
for lo, hi in groups:
    b0, b1 = lo // BUCKETW, min(nbucket, (hi + BUCKETW - 1) // BUCKETW)
    h = hist[b0:b1].sum(0)
    n = cnt[lo:hi].sum()
    if n == 0: continue
    mean = sm[lo:hi].sum() / n
    pieces = psum[lo:hi].sum() / n
    alive = cnt[min(hi, nply) - 1]
    print(f"{lo:5d}-{hi:<7d} {n:11,} {alive:7,} {pieces:7.0f} {mean:7.0f} "
          + " ".join(f"{v:6.0f}" for v in pcts(h, QS)) + f" {mx[lo:hi].max():6d}")

tot = hist.sum(0)
n = cnt.sum()
print(f"\nall plies pooled: mean {sm.sum()/n:.0f}, "
      + ", ".join(f"p{int(q*100)} {v:.0f}" for q, v in zip(QS, pcts(tot, QS)))
      + f", max {mx.max()}")
print(f"positions with < 100 legal moves: {tot[:100//binw].sum()/n*100:.3f}%   "
      f"with > 3000: {tot[3000//binw:].sum()/n*100:.3f}%   with 0 (stalemate): {(mn==0).sum()}")

print("\nmean legal moves vs ply (survivors only beyond ~ply 800)")
step = max(1, nply // 60)
peak = int(np.argmax(np.where(cnt > 0, sm / np.maximum(cnt, 1), 0)))
for i in range(0, nply, step):
    j = min(nply, i + step)
    c = cnt[i:j].sum()
    if not c: continue
    m = sm[i:j].sum() / c
    print(f"{i:5d} {m:7.0f} {'#' * int(m / 30)}")
print(f"\npeak mean is {sm[peak]/cnt[peak]:.0f} moves at ply {peak}")
