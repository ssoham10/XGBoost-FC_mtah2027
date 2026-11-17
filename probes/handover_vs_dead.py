import numpy as np, pandas as pd

GRADED = ((11185, 90651), (110808, 190197))
W = 50  # 2 s
FILES = {"v6": "evaluation/results/soham_v6.csv", "pre-merge": "cache/v3_raw.csv"}

def load(p): return pd.read_csv(p)["label"].values.astype("<U1")

def stats(lab, lo, hi):
    n = len(lab); hp = np.isin(lab, ["H", "A"])
    dcum = np.r_[0, np.cumsum(lab == "D")]
    idx = np.arange(n)
    near = (dcum[np.minimum(idx + W, n)] - dcum[np.maximum(idx - W, 0)]) > 0
    ho = np.flatnonzero((lab[1:] != lab[:-1]) & hp[1:] & hp[:-1]) + 1   # direct H<->A
    ho = ho[(ho > lo) & (ho <= hi)]
    live = hp.copy(); live[:lo] = False; live[hi + 1:] = False
    seg = (lab[lo:hi + 1] == "D").astype(np.int8)
    e = np.diff(np.r_[0, seg, 0])
    same = diff = 0
    for s, t in zip(np.flatnonzero(e == 1) + lo, np.flatnonzero(e == -1) + lo):
        if s > lo and t <= hi and hp[s - 1] and hp[t]:
            same += lab[s - 1] == lab[t]; diff += lab[s - 1] != lab[t]
    return dict(n_ho=len(ho), ho_near=int(near[ho].sum()), live_near=float(near[live].mean()),
                same=int(same), diff=int(diff))

# tests that can fail
a = np.array(list("H" * 200 + "A" * 200 + "D" * 60 + "A" * 200))
r = stats(a, 0, len(a) - 1)
assert (r["n_ho"], r["ho_near"], r["same"], r["diff"]) == (1, 0, 1, 0), r
b = np.array(list("H" * 200 + "D" * 20 + "H" * 30 + "A" * 200))
r = stats(b, 0, len(b) - 1)
assert (r["n_ho"], r["ho_near"], r["same"], r["diff"]) == (1, 1, 1, 0), r
print("self-tests passed\n")

for name, p in FILES.items():
    lab = load(p)
    tot = {k: 0 for k in ("n_ho", "ho_near", "same", "diff")}
    wn = wt = 0.0
    for lo, hi in GRADED:
        r = stats(lab, lo, hi)
        for k in tot: tot[k] += r[k]
        m = np.isin(lab[lo:hi + 1], ["H", "A"]).sum(); wn += r["live_near"] * m; wt += m
    base = wn / wt
    share = tot["ho_near"] / max(tot["n_ho"], 1)
    print("{}: handovers {}  near D {} ({:.1%})  random live frame near D {:.1%}  ratio {:.2f}".format(
        name, tot["n_ho"], tot["ho_near"], share, base, share / base))
    print("   D runs between two possessions: same team {}  other team {}".format(tot["same"], tot["diff"]))