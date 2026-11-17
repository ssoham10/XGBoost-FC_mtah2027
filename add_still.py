import numpy as np, pandas as pd
GRADED = ((11185, 90651), (110808, 190197))
lab = pd.read_csv("evaluation/results/soham_v1.csv")["label"].values.astype("<U1")
g = np.zeros(len(lab), bool)
for a, b in GRADED:
    g[a:b + 1] = True
print("graded D share before: {:.1f}%".format(100 * (lab[g] == "D").mean()))
for a, b in np.load("cache/still_runs.npy"):
    lab[a:b] = "D"
for a, b in np.load("cache/out_runs.npy"):
    lab[a:b] = "D"
lab[np.load("cache/model_dead.npy")] = "D"
print("graded D share after:  {:.1f}%".format(100 * (lab[g] == "D").mean()))
pd.DataFrame({"frame": np.arange(len(lab)), "label": lab}).to_csv("cache/v3_raw.csv", index=False)