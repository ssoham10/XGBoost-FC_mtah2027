import os
import numpy as np
import tracking_io as tio
import deadball
from starter.baseline import smooth, write_submission, NOBODY
from starter.load_smart_tagging import load as load_tags

GRADED = ((11185, 90651), (110808, 190197))
T = tio.load()
px, py, bx, by, team = T["px"], T["py"], T["bx"], T["by"], T["team"]
n = len(bx)

# 1. nearest-team code per frame (NOBODY where the ball is missing)
with np.errstate(invalid="ignore"):
    d = np.hypot(px - bx[:, None], py - by[:, None])
d = np.where(np.isnan(d), np.inf, d)
codes = np.where(np.isfinite(d.min(axis=1)), team[d.argmin(axis=1)], NOBODY).astype(np.int8)

# 2. smooth, then carry the last H/A through frames with no evidence
"""lab = smooth(codes, 50)
idx = np.where(lab != "D", np.arange(n), 0)
np.maximum.accumulate(idx, out=idx)
lab = lab[idx]"""

import gapfill
lab = smooth(codes, 50)
lab = gapfill.fill(lab, T)


# 3. graded halves only, X elsewhere
out = np.full(n, "X", dtype="<U1")
graded = np.zeros(n, bool)
for a, b in GRADED:
    out[a:b + 1] = lab[a:b + 1]
    graded[a:b + 1] = True

# 4. dead-ball overlays
out[110808:113742] = "D"                       # 2nd half graded but not yet kicked off
_, dead = deadball.build()
out[dead & graded] = "D"
for e in load_tags():                          # after-goal kickoffs: dead until ~1.5 s before the window end
    if (e["tags"].get("Type") or "").lower() == "kick off (after goal)":
        s, t = e["start_frame"], e["end_frame"] - 37
        out[s:t] = np.where(graded[s:t], "D", out[s:t])

# 5. report
def poss(seg):
    ch = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    runs = np.split(seg, ch)
    return sum(1 for r in runs if r[0] in "HA")

print("graded counts:", {c: int(((out == c) & graded).sum()) for c in "HAD"})
for name, (a, b) in zip(("1st half", "2nd half"), GRADED):
    print(name, "possessions:", poss(out[a:b + 1]))

path = write_submission(os.path.join("evaluation", "results", "soham_v1.csv"), out)
print("wrote", path, "rows", len(out))