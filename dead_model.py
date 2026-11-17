import warnings
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import tracking_io as tio

GRADED = ((11185, 90651), (110808, 190197))
T = tio.load()
bx, by, ps = T["bx"], T["by"], T["pspeed"]
n = len(bx)
seen = ~np.isnan(bx)
g = np.zeros(n, bool)
for a, b in GRADED:
    g[a:b + 1] = True
pdead = np.load("cache/dead_sp.npy").copy()
for f in ("still_runs", "out_runs"):
    for a, b in np.load("cache/{}.npy".format(f)):
        pdead[a:b] = True
pdead &= g
with np.errstate(invalid="ignore"):
    bs = np.full(n, np.nan); bs[:-5] = np.hypot(bx[5:] - bx[:-5], by[5:] - by[:-5]) * 5
live = g & seen & (bs > 4) & ~pdead

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ms = pd.Series(np.nanmean(ps, 1)); sh = pd.Series((ps < 1).sum(1) / np.maximum((~np.isnan(ps)).sum(1), 1))
    F = np.column_stack([s.rolling(w, center=True, min_periods=w // 5).mean().values
                         for w in (50, 125, 250) for s in (ms, sh)])
ok = ~np.isnan(F).any(1)
block = np.arange(n) // 7500
p = np.full(n, np.nan)
tr = np.flatnonzero((pdead | live) & ok)[::5]
ub = np.unique(block[g])
oof_y, oof_p = [], []
for a, b in GroupKFold(5).split(ub, groups=ub):
    tri = tr[np.isin(block[tr], ub[a])]
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=100, early_stopping=False, random_state=0)
    m.fit(F[tri], pdead[tri].astype(int))
    tei = np.flatnonzero(g & ok & np.isin(block, ub[b]))
    p[tei] = m.predict_proba(F[tei])[:, 1]
    k = tei[(pdead | live)[tei]][::5]
    oof_y += list(pdead[k]); oof_p += list(p[k])
print("out-of-fold AUC (dead vs live): {:.2f}".format(roc_auc_score(oof_y, oof_p)))
ps5 = pd.Series(p).rolling(125, center=True, min_periods=25).mean().values
for thr in (0.7, 0.8, 0.9):
    hi = ps5 > thr
    print("p>{}: pseudo-live flagged {:.1f}% | ball-seen non-pseudo-dead flagged {:.1f}% | ball-missing flagged {:.1f}% of those frames".format(
        thr, 100 * hi[live].mean(), 100 * hi[g & seen & ~pdead].mean(), 100 * hi[g & ~seen & ~pdead].mean()))
hi = ps5 > 0.8
runs = 0; fr = 0
for lo, hh in GRADED:
    e = np.diff(np.r_[0, (hi[lo:hh + 1] & ~seen[lo:hh + 1]).astype(np.int8), 0])
    L = np.flatnonzero(e == -1) - np.flatnonzero(e == 1)
    runs += int((L >= 100).sum()); fr += int(L[L >= 100].sum())
print("ball-missing runs >=4 s with p>0.8: {} ({:.1f}% of graded frames)".format(runs, 100 * fr / g.sum()))

THR, MINLEN = 0.9, 100
mask = np.zeros(n, bool)
with np.errstate(invalid="ignore"):
    for lo, hh in GRADED:
        seg = (ps5[lo:hh + 1] > THR) & ~seen[lo:hh + 1]
        e = np.diff(np.r_[0, seg.astype(np.int8), 0])
        for s, t in zip(np.flatnonzero(e == 1) + lo, np.flatnonzero(e == -1) + lo):
            if t - s >= MINLEN:
                mask[s:t] = True
np.save("cache/model_dead.npy", mask)
print("model-dead runs: {} frames: {} ({:.1f}% of graded)".format(
    int(np.sum(np.diff(np.r_[0, mask.astype(np.int8)]) == 1)), int(mask.sum()), 100 * mask.sum() / g.sum()))

print("\nsensitivity (THR, MINLEN s): runs, frames, % graded")
for thr in (0.8, 0.85, 0.9, 0.95):
    for ml in (75, 100, 150):
        mk = np.zeros(n, bool)
        for lo, hh in GRADED:
            seg = (ps5[lo:hh + 1] > thr) & ~seen[lo:hh + 1]
            e = np.diff(np.r_[0, seg.astype(np.int8), 0])
            for s, t in zip(np.flatnonzero(e == 1) + lo, np.flatnonzero(e == -1) + lo):
                if t - s >= ml:
                    mk[s:t] = True
        r = int(np.sum(np.diff(np.r_[0, mk.astype(np.int8)]) == 1))
        print("  {:.2f}, {:>3} s: {:>3} runs, {:>6} frames, {:.1f}%".format(
            thr, ml / 25, r, int(mk.sum()), 100 * mk.sum() / g.sum()))