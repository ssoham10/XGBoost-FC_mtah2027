import numpy as np, pandas as pd
import sys

GRADED = ((11185, 90651), (110808, 190197))
MIN = 75  # frames = 3 s at 25 fps

def runs(a):
    ch = np.flatnonzero(a[1:] != a[:-1]) + 1
    st, en = np.r_[0, ch], np.r_[ch, len(a)]
    return [[a[s], s, e] for s, e in zip(st, en)]

def clean(seg):
    r, skip = runs(seg), set()
    while True:
        cand = [(q[2] - q[1], i) for i, q in enumerate(r)
                if q[0] in "HA" and q[2] - q[1] < MIN and q[1] not in skip]
        if not cand:
            break
        _, i = min(cand)
        nb = [r[j] for j in (i - 1, i + 1) if 0 <= j < len(r) and r[j][0] in "HA"]
        if not nb:                      # boxed in by dead ball: leave it
            skip.add(r[i][1]); continue
        r[i][0] = max(nb, key=lambda q: q[2] - q[1])[0]
        m = [r[0]]
        for q in r[1:]:
            if q[0] == m[-1][0]: m[-1][2] = q[2]
            else: m.append(q)
        r = m
    out = seg.copy()
    for l, s, e in r:
        out[s:e] = l
    return out

df = pd.read_csv(sys.argv[1])

lab = df["label"].values.astype("<U1")
out = lab.copy()
for name, (a, b) in zip(("1st half", "2nd half"), GRADED):
    before = sum(q[0] in "HA" for q in runs(lab[a:b + 1]))
    out[a:b + 1] = clean(lab[a:b + 1])
    after = sum(q[0] in "HA" for q in runs(out[a:b + 1]))
    print(name, "possessions:", before, "->", after)

pd.DataFrame({"frame": np.arange(len(out)), "label": out}).to_csv(sys.argv[2], index=False)
print("wrote", sys.argv[2])