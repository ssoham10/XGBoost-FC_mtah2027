import warnings
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

GRADED = ((11185, 90651), (110808, 190197))
TAU = 250


def _feats(T, m):
    px, py, ps = T["px"], T["py"], T["pspeed"]
    x = np.where(m, px, np.nan); y = np.where(m, py, np.nan); s = np.where(m, ps, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cx, cy = np.nanmean(x, 1), np.nanmean(y, 1)
        return [cx, cy, np.nanstd(x, 1), np.nanstd(y, 1), np.nanmean(s, 1),
                np.nanmean(np.hypot(x - cx[:, None], y - cy[:, None]), 1)]


def fill(lab, T):
    """lab: array of 'H'/'A'/'D' from smooth(code, 50); 'D' = no ball evidence. Fills every
    D inside the graded halves with H/A (blend of edge labels and a player-shape model)."""
    team, bx = T["team"], T["bx"]
    n = len(lab)
    seen = ~np.isnan(bx)
    truth = np.where(seen, np.where(lab == "A", 1, np.where(lab == "H", 0, -1)), -1)
    F0, F1 = _feats(T, team == 0), _feats(T, team == 1)
    X = np.column_stack(F0 + F1 + [a - b for a, b in zip(F0, F1)])
    ok = ~np.isnan(X).any(1)
    graded = np.zeros(n, bool)
    for a, b in GRADED:
        graded[a:b + 1] = True
    block = np.arange(n) // 7500

    p = np.full(n, 0.5)  # out-of-fold P(away), split by 5-minute block
    tr_idx = np.flatnonzero(graded & ok & (truth >= 0))[::5]
    ub = np.unique(block[graded])
    for tr_b, te_b in GroupKFold(5).split(ub, groups=ub):
        tri = tr_idx[np.isin(block[tr_idx], ub[tr_b])]
        m = HistGradientBoostingClassifier(max_depth=4, max_iter=150, early_stopping=False, random_state=0).fit(X[tri], truth[tri])
        tei = np.flatnonzero(graded & ok & np.isin(block, ub[te_b]))
        p[tei] = m.predict_proba(X[tei])[:, 1]
    p5 = uniform_filter1d(p, 250)

    out = lab.copy()
    for lo, hi in GRADED:
        seg = lab[lo:hi + 1] == "D"
        e = np.diff(np.r_[0, seg.astype(np.int8), 0])
        for s, t in zip(np.flatnonzero(e == 1) + lo, np.flatnonzero(e == -1) + lo):
            L = t - s
            i = np.arange(L)
            has0 = s > lo and lab[s - 1] != "D"
            has1 = t <= hi and lab[t] != "D"
            c0 = float(lab[s - 1] == "A") if has0 else 0.0
            c1 = float(lab[t] == "A") if has1 else 0.0
            w0 = np.exp(-i / TAU) * has0
            w1 = np.exp(-(L - i) / TAU) * has1
            prob = (w0 * c0 + w1 * c1 + p5[s:t]) / (w0 + w1 + 1)
            out[s:t] = np.where(prob > 0.5, "A", "H")
    return out