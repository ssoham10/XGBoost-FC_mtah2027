import hashlib, subprocess, sys

STEPS = [
    ["deadball.py"],              # -> cache/dead_sp.npy (set-piece dead)
    ["still_ball.py"],            # -> cache/still_runs.npy
    ["out_runs.py"],              # -> cache/out_runs.npy
    ["dead_model.py"],            # -> cache/model_dead.npy
    ["make_sub.py"],              # -> evaluation/results/soham_v1.csv
    ["add_still.py"],             # -> cache/v3_raw.csv
    ["post_proc.py", "cache/v3_raw.csv", "evaluation/results/soham_v6_repro.csv"],
    ["evaluation/validate_submission.py", "evaluation/results/soham_v6_repro.csv"],
]
for s in STEPS:
    print("\n=== " + " ".join(s), flush=True)
    subprocess.run([sys.executable] + s, check=True)

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

EXPECTED = "8d7932e46f8014403c380d5c31e2ace6"
got = md5("evaluation/results/soham_v6_repro.csv")
print("\nexpected", EXPECTED, "\ngot     ", got)
print("IDENTICAL" if got == EXPECTED else "DIFFERENT")