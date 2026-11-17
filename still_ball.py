import numpy as np
import tracking_io as tio
from games import GAME, GAMES

FPS = GAMES[GAME]["fps"]
L, W = GAMES[GAME]["pitch"]
GRADED = ((11185, 90651), (110808, 190197))
LAG, MAXD, MINLEN, BRIDGE = 50, 1.0, 75, 25   # 2 s lag, <1 m moved, >=3 s run, bridge 1 s gaps

T = tio.load()
bx, by = T["bx"], T["by"]
n = len(bx)
graded = np.zeros(n, bool)
for a, b in GRADED:
    graded[a:b + 1] = True

d = np.full(n, np.nan)
d[:-LAG] = np.hypot(bx[LAG:] - bx[:-LAG], by[LAG:] - by[:-LAG])
still = np.zeros(n, bool)
with np.errstate(invalid="ignore"):
    s = d < MAXD
for i in np.flatnonzero(s):                    # a still pair covers the whole 2 s span
    still[i:i + LAG + 1] = True

# bridge short gaps, then keep long runs
idx = np.flatnonzero(still)
gaps = np.flatnonzero(np.diff(idx) > BRIDGE)
st = np.r_[idx[0], idx[gaps + 1]]
en = np.r_[idx[gaps], idx[-1]] + 1
runs = [(a, b) for a, b in zip(st, en) if b - a >= MINLEN and graded[a:b].all()]
mask = np.zeros(n, bool)
for a, b in runs:
    mask[a:b] = True

dead_sp = np.load("cache/dead_sp.npy")
cx = np.array([np.nanmedian(bx[a:b]) for a, b in runs])
cy = np.array([np.nanmedian(by[a:b]) for a, b in runs])
ln = np.array([(b - a) / FPS for a, b in runs])
edge = np.minimum.reduce([cx, L - cx, cy, W - cy]) <= 2.0
center = np.hypot(cx - L / 2, cy - W / 2) <= 2.0
pen = ((cx < 16.5) | (cx > L - 16.5)) & (np.abs(cy - W / 2) < 20.2)

print("still-ball runs (>=3 s, graded): {}".format(len(runs)))
print("  length s p10/50/90: {}".format(np.percentile(ln, [10, 50, 90]).round(1).tolist()))
print("  share of graded frames: {:.1f}%".format(100 * mask[graded].mean()))
print("  share of those already in set-piece dead: {:.0f}%".format(
    100 * (mask & dead_sp)[graded].sum() / mask[graded].sum()))
print("  where: on/near a line {}, centre spot {}, inside a penalty area {}, elsewhere {}".format(
    int(edge.sum()), int((center & ~edge).sum()), int((pen & ~edge & ~center).sum()),
    int((~edge & ~center & ~pen).sum())))
print("  per half:", [sum(1 for a, b in runs if lo <= a <= hi) for lo, hi in GRADED])
print("  runs with a gap of ball-missing frames inside: {}".format(
    sum(1 for a, b in runs if np.isnan(bx[a:b]).mean() > 0.3)))
np.save("cache/still_runs.npy", np.array(runs))