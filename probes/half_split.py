import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import tracking_io as tio

# half 2 starts at the real kickoff: 110808-113741 is a known D block (3.7% of that half)
HALVES = ((11185, 90651), (113742, 190197))
V6 = "evaluation/results/soham_v6.csv"

def runs_of(lab):
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    return [(lab[s], s, e) for s, e in zip(np.r_[0, ch], np.r_[ch, len(lab)])]

def half_stats(lab, codes, lo, hi):
    seg, c = lab[lo:hi + 1], codes[lo:hi + 1]
    r = [x for x in runs_of(seg) if x[0] in "HA"]
    L = np.array([e - s for _, s, e in r]) / 25
    hp = np.isin(seg, ["H", "A"])
    m = hp & (c >= 0)
    pred = (seg == "A").astype(int)
    miss = (c < 0)
    mr = [e - s for v, s, e in runs_of(miss.astype(int)) if v == 1]
    return dict(frames=len(seg), d=(seg == "D").mean(), poss=len(r), med=np.median(L), p90=np.percentile(L, 90),
                h_share=(seg[hp] == "H").mean(), agree=(pred[m] == c[m]).mean(),
                seen=1 - miss.mean(), gap_med=np.median(mr) if mr else 0)

# tests that can fail
lab = np.array(list("H" * 100 + "A" * 50 + "D" * 10))
r = half_stats(lab, np.zeros(160, int), 0, 159)
assert r["poss"] == 2 and abs(r["d"] - 10 / 160) < 1e-9 and abs(r["agree"] - 100 / 150) < 1e-9 and abs(r["h_share"] - 100 / 150) < 1e-9, r
r = half_stats(lab, np.r_[np.zeros(80, int), np.full(80, -1)], 0, 159)
assert abs(r["seen"] - 0.5) < 1e-9 and r["gap_med"] == 80, r
print("self-tests passed\n")

lab = pd.read_csv(V6)["label"].values.astype("<U1")
T = tio.load()
with np.errstate(invalid="ignore"):
    d = np.hypot(T["px"] - T["bx"][:, None], T["py"] - T["by"][:, None])
d = np.where(np.isnan(d), np.inf, d)
codes = np.where(np.isfinite(d.min(1)), T["team"][d.argmin(1)], -1).astype(np.int8)

print("half  frames   D%   poss  med_s  p90_s  H%    agree_w_nearest  ball_seen  gap_med_frames")
for k, (lo, hi) in enumerate(HALVES, 1):
    s = half_stats(lab, codes, lo, hi)
    print("{:<5} {:>6} {:>5.1%} {:>5} {:>6.1f} {:>6.1f} {:>5.1%} {:>10.1%} {:>13.1%} {:>9.0f}".format(
        k, s["frames"], s["d"], s["poss"], s["med"], s["p90"], s["h_share"], s["agree"], s["seen"], s["gap_med"]))