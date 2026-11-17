import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import tracking_io as tio

V6, RAW, BRIDGE = "evaluation/results/soham_v6.csv", "cache/v3_raw.csv", 25  # RAW = before 3 s merge

def load(p): return pd.read_csv(p)["label"].values.astype("<U1")

def runs_of(lab):
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    return [(lab[s], s, e) for s, e in zip(np.r_[0, ch], np.r_[ch, len(lab)])]

def poss(lab): return [r for r in runs_of(lab) if r[0] in "HA"]

def agreement(a, b):
    """My approximation of spellF1 between two label files (same-team, 1-to-1 greedy IoU, th .3-.7)."""
    A, B = poss(a), poss(b)
    iou = np.zeros((len(A), len(B)))
    for i, (la, sa, ea) in enumerate(A):
        for j, (lb, sb, eb) in enumerate(B):
            if la == lb and min(ea, eb) > max(sa, sb):
                iou[i, j] = (min(ea, eb) - max(sa, sb)) / (max(ea, eb) - min(sa, sb))
    f = []
    for th in (.3, .4, .5, .6, .7):
        pairs = sorted(((iou[i, j], i, j) for i, j in zip(*np.nonzero(iou >= th))), reverse=True)
        ui, uj = set(), set()
        for _, i, j in pairs:
            if i not in ui and j not in uj:
                ui.add(i); uj.add(j)
        f.append(2 * len(ui) / (len(A) + len(B)))
    return float(np.mean(f))

def nearest_codes(T):
    with np.errstate(invalid="ignore"):
        d = np.hypot(T["px"] - T["bx"][:, None], T["py"] - T["by"][:, None])
    d = np.where(np.isnan(d), np.inf, d)
    return np.where(np.isfinite(d.min(1)), T["team"][d.argmin(1)], -1).astype(np.int8)

def streaks(codes):
    """Runs of the same nearest team on ball-seen frames; a gap > 1 s breaks the run."""
    idx = np.flatnonzero(codes >= 0); c = codes[idx]
    brk = np.flatnonzero((c[1:] != c[:-1]) | (np.diff(idx) > BRIDGE)) + 1
    st, en = np.r_[0, brk], np.r_[brk, len(idx)] - 1
    return idx[st], idx[en], c[st]

def variant(lab, seen_cum, S, W):
    """Keep a hand-over only if the new team is nearest for >= W frames starting by t+W and lasting past t.
    No ball evidence within +/-W of t -> unjudgeable, kept."""
    out, r = lab.copy(), poss(lab)
    cur = [x[0] for x in r]
    st, en, tm = S
    n_h = n_ok = n_rej = n_unj = 0
    for i in range(1, len(r)):
        _, s, e = r[i]
        if r[i - 1][2] != s or cur[i] == cur[i - 1]:
            continue
        n_h += 1
        if W == 0:
            n_ok += 1; continue
        lo, hi = max(s - W, 0), min(s + W, len(seen_cum) - 1)
        if seen_cum[hi] - seen_cum[lo] == 0:
            n_unj += 1; continue
        new = 0 if cur[i] == "H" else 1
        if ((tm == new) & (en - st + 1 >= W) & (st <= s + W) & (en >= s)).any():
            n_ok += 1
        else:
            n_rej += 1; cur[i] = cur[i - 1]; out[s:e] = cur[i]
    return out, (n_h, n_ok, n_rej, n_unj)

v6, raw = load(V6), load(RAW)
T = tio.load()
codes = nearest_codes(T)
seen_cum = np.cumsum(codes >= 0)
S = streaks(codes)

# tests that can fail
assert agreement(v6, v6) == 1.0
shifted = np.r_[np.full(25, "X"), v6[:-25]]
assert 0.3 < agreement(v6, shifted) < 1.0, "1 s shift should hurt but not destroy"
assert (variant(v6, seen_cum, S, 0)[0] == v6).all(), "W=0 must be a no-op"
print("self-tests passed\n")

print("v6 possessions: {}   3 s merge effect (raw->v6 agreement): {:.3f}".format(
    len(poss(v6)), agreement(raw, v6)))
print("\nW(s) handovers confirmed rejected unjudgeable | poss after | agreement with v6")
for sec in (1, 2, 3):
    out, (h, ok, rej, unj) = variant(v6, seen_cum, S, sec * 25)
    print("{:>4} {:>9} {:>9} {:>8} {:>11} | {:>10} | {:.3f}".format(
        sec, h, ok, rej, unj, len(poss(out)), agreement(v6, out)))
    if sec == 2:
        os.makedirs("cache", exist_ok=True)
        pd.DataFrame({"frame": np.arange(len(out)), "label": out}).to_csv("cache/controlled_w2.csv", index=False)