# XGBoost FC - More than a Hack 2027 (possession labels)

One label (H/A/D/X) per frame of BAR-ATH. Built on the provided tracking (ATD) and Smart Tagging only;
no hand labels, no event data, no external data, no hosted LLMs.

## Reproduce
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy 2.4.6, pandas 3.1.1, scipy 1.17.1, scikit-learn 1.9.1
# put the gated bundle in ./data or set MTAH_DATA_DIR
python run_all.py                      # ends with IDENTICAL if the file matches the submitted one
```
Output: `evaluation/results/soham_v6_repro.csv` (submitted as `XGBoost-FC.csv`). Takes a few minutes; the first
step parses the raw tracking once into `cache/` (git-ignored, never pushed).

## Pipeline (run_all.py order)
1. `deadball.py` - dead windows from Smart Tagging set pieces + ball-motion start/restart detection
2. `still_ball.py` - ball nearly still for >= 3 s
3. `out_runs.py` - ball outside the pitch for >= 2 s
4. `dead_model.py` - gradient boosting on player-speed features, trained on the pseudo-labels above (no human labels)
5. `make_sub.py` - nearest-player team, +/-2 s majority vote, `gapfill.py` for ball-missing stretches, dead overlays
6. `add_still.py` - merge all dead sources
7. `post_proc.py` - merge H/A runs shorter than 3 s into their neighbour
8. `evaluation/validate_submission.py`

`probes/` holds the final consistency checks (outputs saved as `*_output.txt`). See `NOTES.md` and `TECHNICAL.md`.
Match data and derived frame-level data are not in this repo.