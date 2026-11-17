# XGBoost FC — More than a Hack 2027

Frame-by-frame possession labels for the BAR-ATH match, using tactical tracking (ATD) and Smart Tagging. The pipeline combines nearest-player rules, dead-ball detection, and gradient-boosting models trained on algorithmic pseudo-labels. It uses no human-labelled training data, event data, external data, or hosted LLMs.

Each output row contains a zero-based tactical video frame and one label:

| Label | Meaning |
| --- | --- |
| `H` | Home-team possession |
| `A` | Away-team possession |
| `D` | Dead ball |
| `X` | Outside the graded match intervals |

## Reproduce the submission

Run all commands from the repository root. You need **Python 3.11** and access to the gated match-data bundle; the data is not included in this repository. Obtain it through the challenge's authorised download channel.

### 1. Set up the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. The pinned dependencies are NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, and scikit-learn 1.9.1.

### 2. Add the match data

Keep the original filenames and place these three required files under `data/tactical/`:

```text
data/
└── tactical/
    ├── BAR-ATH_tactical_atd.xml
    ├── BAR-ATH_tactical_atd.txt
    └── BAR-ATH_tactical_smart_tagging.xml
```

The homography CSV can be stored alongside them, but this pipeline does not use it. Neither the match video nor broadcast assets are required.

If the bundle is stored elsewhere, set `MTAH_DATA_DIR` to the directory **containing** `tactical/`:

```bash
export MTAH_DATA_DIR=/path/to/match-data
```

Raw match assets and generated tracking caches are ignored by Git.

### 3. Run the full pipeline

```bash
python run_all.py
```

The first run parses the raw tracking into `cache/`; later runs reuse that cache. Allow a few minutes for parsing and model training.

The regenerated submission is written to:

```text
evaluation/results/soham_v6_repro.csv
```

It corresponds to the submission named `XGBoost-FC.csv`. A successful reproduction ends with:

```text
expected 8d7932e46f8014403c380d5c31e2ace6
got      8d7932e46f8014403c380d5c31e2ace6
IDENTICAL
```

`IDENTICAL` means the generated file matches the expected MD5 checksum. `DIFFERENT` means it does not, even if the format validator passes. Check the final line: a checksum mismatch does not make the script exit with an error.

### Verified reproduction

Two full runs with Python 3.11.15 and the pinned dependencies produced the expected checksum, including a repeat run with the tracking cache rebuilt from the raw data. The output passed validation with **199,319 rows** and **373 possessions**.

This verifies reproduction of the submitted file; it does not establish the accuracy of its labels.

## How it works

`run_all.py` executes these steps in order:

| Step | Script | Purpose |
| --- | --- | --- |
| 1 | `deadball.py` | Detect dead-ball windows from Smart Tagging set pieces and ball motion. |
| 2 | `still_ball.py` | Find ball-still intervals lasting at least 3 seconds. |
| 3 | `out_runs.py` | Find ball-outside-pitch intervals lasting at least 2 seconds. |
| 4 | `dead_model.py` | Train a player-speed model on rule-based pseudo-labels to detect additional dead-ball intervals. |
| 5 | `make_sub.py` | Assign the nearest player's team, apply a ±2-second majority vote, fill missing-ball intervals with `gapfill.py`, and overlay dead-ball labels. |
| 6 | `add_still.py` | Combine the still-ball, out-of-pitch, and model-based dead-ball detections. |
| 7 | `post_proc.py` | Merge H/A runs shorter than 3 seconds into an adjacent H/A run where possible. |
| 8 | `evaluation/validate_submission.py` | Validate the output format and report label and possession counts. |

## Validate and evaluate

To validate the regenerated submission separately:

```bash
python evaluation/validate_submission.py evaluation/results/soham_v6_repro.csv
```

To score the CSVs in `evaluation/results/` against the public five-minute ground-truth window:

```bash
python evaluation/run_evaluation.py --gt public
```

The scorer also works without the gated match data: a fresh clone includes four baseline CSVs. Public-window scores cover only a small part of the match and should not be treated as full-match performance.

## Troubleshooting

- **`match data not found`:** check the filenames and directory structure above. If using `MTAH_DATA_DIR`, point it at the parent of `tactical/`.
- **Missing Python packages:** activate `.venv` and install `requirements.txt` with that environment's Python.
- **`DIFFERENT`:** confirm the dependency versions and input bundle. If you changed the tracking data, move the old `cache/` aside before rerunning; cached tracking is reused automatically.

## Further reading

- [TECHNICAL.md](TECHNICAL.md): modelling choices, pseudo-label training, checks, and limitations, including weak dead-ball recall and the unvalidated 3-second merge threshold.
- [NOTES.md](NOTES.md): development history and experiment results.
- [probes/](probes/): consistency-check scripts and saved outputs. Some checks reference historical candidate files that are not included or regenerated by `run_all.py`.
