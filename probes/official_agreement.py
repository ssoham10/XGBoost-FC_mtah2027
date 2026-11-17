import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "evaluation"))
import numpy as np, pandas as pd
import possession_metrics as pm

R = "evaluation/results/"
FILES = {"v6": R + "soham_v6.csv", "v5": R + "soham_v5.csv",
         "pre-merge": "cache/v3_raw.csv", "controlled2s": "cache/controlled_w2.csv"}

def load(p): return pd.read_csv(p)["label"].values.astype("<U1")

def spell(a, b):
    """a plays the role of truth, b of prediction; both are our own files."""
    g, p = pm.label_codes(a), pm.label_codes(b)
    return pm.spell_scores(g, p, pm.valid_runs(g))["mean_f1"]

v6 = load(FILES["v6"])
first = spell(v6, v6)
print("v6 vs v6:", first)

# tests that can fail
a = first
assert abs(a - 1.0) < 1e-9, "identical files must score 1"
late1 = np.r_[np.full(1, "X"), v6[:-1]]
late25 = np.r_[np.full(25, "X"), v6[:-25]]
s1, s25 = float(spell(v6, late1)), float(spell(v6, late25))
print("shift 1 frame: {:.4f}   shift 25 frames: {:.4f}".format(s1, s25))
assert s1 > 0.97, "docs say a 1-frame offset is ~free"
assert 0.3 < s25 < 0.95, "1 s shift should hurt but not destroy"
print("self-tests passed\n")

print("pair                      A->B    B->A")
names = [k for k in FILES if os.path.exists(FILES[k])]
for i, x in enumerate(names):
    for y in names[i + 1:]:
        A, B = load(FILES[x]), load(FILES[y])
        print("{:<24} {:.3f}   {:.3f}".format(x + " vs " + y, float(spell(A, B)), float(spell(B, A))))