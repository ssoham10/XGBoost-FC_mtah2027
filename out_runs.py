import numpy as np
import tracking_io as tio
from games import GAME, GAMES

L, W = GAMES[GAME]["pitch"]
GRADED = ((11185, 90651), (110808, 190197))
bx, by = tio.load()["bx"], tio.load()["by"]
n = len(bx)
g = np.zeros(n, bool)
for a, b in GRADED:
    g[a:b + 1] = True
M = 0.5
with np.errstate(invalid="ignore"):
    out = (bx < -M) | (bx > L + M) | (by < -M) | (by > W + M)
idx = np.flatnonzero(out & g)
gaps = np.flatnonzero(np.diff(idx) > 25)
st = np.r_[idx[0], idx[gaps + 1]]
en = np.r_[idx[gaps], idx[-1]] + 1
runs = [(a, b) for a, b in zip(st, en) if b - a >= 50 and g[a:b].all()]
np.save("cache/out_runs.npy", np.array(runs))
print("ball-out runs >=2 s:", len(runs), " frames:", sum(b - a for a, b in runs))