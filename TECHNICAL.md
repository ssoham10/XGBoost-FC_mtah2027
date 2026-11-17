# XGBoost FC - Technical document

## Approach
A rule-and-pseudo-label pipeline on the provided ATD. Live frames: the team of the player nearest the ball, majority-voted over +/-2 s. Where the ball is missing, a gradient-boosting model on team-shape features (centroid, spread, speed per team) predicts the team; it is trained only on frames where the nearest-player rule has a ball, and its output is blended with the labels at the edges of each gap (decay constant 250 frames). Dead ball is the union of: Smart Tagging set-piece windows refined by ball motion, ball still >= 3 s, ball outside the pitch >= 2 s, the 2,934 graded frames before the second-half kickoff, after-goal kickoff windows, and a second gradient-boosting model (player speed features) trained on those rule-based labels. Last, any H/A run shorter than 75 frames (3 s) is merged into its longer H/A neighbour, so possessions are intervals, not flicker.Why this and not something else: the ranking metric rewards correct possessions; smoothing and merging dominated per-frame tuning in the baselines.

## Assets used / dropped
Used: tactical ATD (players and ball), Smart Tagging (set-piece windows only). 
Dropped: Smart Tagging team tags (did not help), broadcast video and ATD, video, homography, HMM segmentation (the 2-4 s merge minima agreed at 0.90+, so no gain was shown).

## Training data
No labelled data of any kind. Both models use algorithmic pseudo-labels from the provided game's tracking. No hand labels on this or any other game; no event data or external data. Models were evaluated with 5-fold CV over 5-minute blocks, which measures consistency with the pseudo-labels, not accuracy against truth.

## Possession rules implemented
Possession = a maximal H or A run with the ball alive. Loose balls keep the last team (nearest-player vote plus carry-forward). A hand-over needs the new team to hold the signal for 3 s (the merge). Dead balls are D and no owner is assigned. Throw-in, goal-kick and similar windows come from Smart Tagging plus ball motion. The 3 s minimum comes from our own football judgement (guess), not from tuning on any ground truth.

## Checks (no ground truth used)
Official `spell_scores` between our own files: v6 vs v5 0.968 (inside the 0.04 noise band); v6 vs pre-merge 0.722; a 2 s controlled-possession variant (373 -> 285 possessions, agreement 0.776) was rejected as large and unvalidatable. Hand-overs near D runs: 0 in v6 by construction of the merge (pre-merge ratio 0.95). Halves are consistent: D 12.8% / 13.3%, median possession 9.4 s / 10.6 s, agreement with the nearest-team rule 83.3% / 83.8%.  The Player-shape model achieved 69% vs always-home (56-58%). When blending edge labels with the player-shape HGB model (tau=250), accuracy reached 79% (compared to 65% for simple carry-forward), averaging 0.49 switches per gap. Final file: 373 possessions, D 14.7% of graded frames. The public 5-minute window was used once, as a bug check on the finished file (macroF1 0.683, spellF1 0.482, 17 possessions); no choice was made from it. Exploratory scripts (not in the repo) were used to inspect distributions and pick thresholds, in this order: ball-missing statistics, set-piece restarts, ball-out and still-ball runs, gap-model blending, dead-detector grid (results were flat across the grid), merge minimum. `explore.py`(not in the repo) read the public GT for descriptive statistics only; no threshold was set from it.

## Limitations
- The 3 s merge is the largest unvalidated choice (moves the file by 0.28 agreement).
- Dead-ball recall is weak on the public window (0.31): many dead frames are labelled H/A, and we did not tune it.
- Frames with no ball evidence rely on the gap model, which carries the previous team forward; same-team hand-backs after D runs outnumber team changes about 2:1, which may be biased.
- Possession is nearest-player based; there is no pass or control detection.
A better version would include: a validated control rule and better dead-ball detection from player motion.

## Reproduction
See README.md: `pip install -r requirements.txt`, then `python run_all.py` (ends with IDENTICAL).