# Status
Frames 0-based = frame_id-1. Graded: 11185-90651, 113742(2nd kickoff, 110808-113741 = D)-190197.
Public GT: frames 11185-18684.
# Findings
- Baseline spellF1 (public 5min): smooth125 0.424, smooth50 0.326, raw 0.038
- D weak: only 116/1469 D frames caught by smooth125
- Ball seen: 59.3% of graded frames; 88% in public window; per-5min blocks range 20-88%
- Ball-missing runs: median 268 frames, p90 845, max 5537
- Missing ball != dead: of public no-ball frames, ~55% D, ~44% A
- D: ball out of pitch on 10.7% of D frames; ball speed median 2.7 vs 5.4-6.4 live
- True segments (public): H 8, A 9, D 4; median 7-16 s
# Plan
1. Dead ball: pre-kickoff block, set-piece tags, ball out of pitch
2. Possession in ball gaps: carry last state forward, switch on player-movement evidence
3. HMM over H/A/D with minimum duration
# Done / Next


# Status (14 Nov 2026, deadline 17 Nov)
Frames 0-based = frame_id-1. Graded: 11185-90651, 110808-190197 (110808-113741 = D, pre-kickoff).
Public GT: frames 11185-18684 (bug check only, never tune on it).
Current best candidate: evaluation/results/soham_v3.csv. Fallback: soham_v2.csv.
Public board (bug check only): v3 macro 0.660 / spell 0.491; v2 0.634 / 0.494; v1 0.619 / 0.372.

# Pipeline so far (v3)
1. make_sub.py: nearest-team-to-ball per frame -> majority vote +/-50 -> carry last H/A through gaps -> D overlays (110808-113741, set-piece intervals from deadball.py, after-goal kickoffs) -> v1
2. add_still.py: add 48 still-ball runs (>=3 s) as D to v1 -> cache/v3_raw.csv
3. postproc.py <in> <out>: merge any H/A run < 3 s (75 frames) into its longer neighbour -> v3
   Possessions: v1 ~553 -> v3 322 (173 + 149). D share of graded: 7.0% -> 10.9%.

# Findings
- Smart Tagging SET PIECES: 59 total. Non-kickoff windows are 35 s long, kickoff windows 5 s.
  Window start ~ when the ball goes dead; restart ~4-7 s in (median: throw-in 5.7, corner 6.6,
  goal kick 4.2, free kick 5.8 s). Windows cover 27.7% of graded frames, so they are not dead intervals.
- Throw-in Team is N/A for all 26 and for all kickoffs; corner/goal/free kick have a team.
- Ball seen before restart: corners 1%, throw-ins 39%, goal kicks 75%, free kicks 33%.
- Start-of-dead finder mostly failed (14/53), so intervals start at window start (likely ~2 s late
  for throw-ins). Set-piece dead = only 4.5-5.3% of each half.
- Player speed is a weak dead-ball signal (median 0.96 dead vs 1.40 live, heavy overlap),
  but speed jumps right after the restart (0.96 -> 2.0).
- Still-ball runs (<1 m in 2 s, >=3 s): 48 runs, 4.5% of graded, all with a fully tracked ball, 12% overlap with set-piece mask; 25 elsewhere on pitch, 14 in a penalty area, 8 on a line, 1 centre.
- Min-possession 3 s cut possessions ~45% (chosen from football knowledge, not from the public score).

# Plan / Next
1. Ball-missing stretches (41% of graded): measure player behaviour there (bunched around a spot vs spread).
2. Better H/A during gaps than carry-forward; maybe HMM with min duration.
3. Before 17 Nov: one run_all.py that regenerates the submission from scratch (RULES #3),
   public repo (data/ and cache/ gitignored), technical document, read SUBMISSION.md.


# Status (15 Nov 2026, deadline 17 Nov 23:59 CET)
CANDIDATE: evaluation/results/soham_v6.csv. FALLBACK: soham_v5.csv. run_all.py regenerates v6 (md5 identical).
Env: Python 3.11.0, numpy 2.4.6, pandas 3.1.1, scipy 1.17.1, scikit-learn 1.9.1 (requirements.txt)
Frames 0-based = frame_id-1. Graded 11185-90651, 110808-190197 (110808-113741 = D).
Public GT: bug check only, never tune. v4 public 0.651/0.471 (macro/spell) = sane.

# Pipeline (run_all.py)
1 deadball.py -> cache/dead_sp.npy (set-piece dead)   
2 still_ball.py -> still_runs.npy (>=3 s still ball)
3 out_runs.py -> out_runs.npy (ball off pitch >=2 s)   
4 dead_model.py -> model_dead.npy (player-only dead detector)
5 make_sub.py: nearest team, vote +/-50, gapfill.fill (blend of edge labels + player-shape HGB, TAU=250), D overlays -> v1
6 add_still.py: still + out + model_dead as D -> cache/v3_raw.csv
7 post_proc.py: merge H/A runs < 3 s -> soham_v6.csv. 373 possessions, D 14.7% of graded.

# Measured (no GT; pseudo-labels from own nearest-team rule)
- Ball-gap carry-forward ~ coin flip after >=4 s (team differs 40-48%).
- Player-shape model: 69% vs 56-58% always-home. Gap sim: blend tau=250 79% vs carry 65%, 0.49 switches/gap.
- Dead detector (player speed share <1 m/s, 5 s): OOF AUC 0.88; at p>0.9 flags 0.3% of ball-seen live, 9.8% of ball-missing.
  Rule p>0.9, >=4 s: 16 runs, 3.4% graded, all new vs v5, median 9.8 s. Grid 2.1-6.7%: smooth, not fragile.
- Merge min: 2 s/3 s/4 s agree 0.92/0.90 (spell-style) -> keep 3 s; 5 s = 0.81. HMM skipped for that reason.
- Smart Tagging team tags: mostly name the DEFENDING team; weak in gaps -> dropped.
- After-goal kickoffs: 3 of 4 confirmed (ball leaves centre at 3.4-4.1 s, rule ends D at 3.5 s). Window 62012 unverifiable (ball unseen); 4 GOALS tags vs 2-1 score = possible false tag. Not edited by hand.
- Throw-ins: 13/26 with ball-out evidence; first-out median +0.2 s vs window start; v6 covers out span median 100% -> no change.
- Rules: event data about this game is banned (RULES #2). Do not use it.

# Open / next
1. controlled-possession check
2. Before 17 Nov: requirements.txt, .gitignore, README, tech doc (approach, assets used/dropped, rules, limits, pseudo-label training, repro), probes/ folder, copy v6 to <team_name>.csv + validate.
## Probes (final checks, none changed the pipeline)
Run from the repo root after run_all.py (they read cache/ and evaluation/results/, which are not in the repo).

# Status (17 Nov 2026)
- probes/controlled.py: 2 s controlled-possession variant rejects 24% of hand-overs, 373 -> 285 possessions, agreement with v6 0.776. Large, unvalidatable change. NOT adopted.
- probes/official_agreement.py: official spell_scores between our own files, no truth. v6 vs v5 0.968 (inside the 0.04 noise band); v6 vs pre-merge 0.722 (the 3 s merge is the biggest unvalidated choice); 1-frame shift 1.000.
- probes/handover_vs_dead.py: v6 shows 0 hand-overs near D runs, but this holds by construction because the 3 s merge removes them. Pre-merge ratio 0.95 (no relation to D runs).
- probes/half_split.py: halves consistent (D 12.8% / 13.3%, median possession 9.4 s / 10.6 s, agreement with nearest-team rule 83.3% / 83.8%). No drift found.
- ## Probes (final checks, none changed the pipeline)
Run from the repo root after run_all.py. probes_controlled.py: 2 s variant not adopted (373->285, agreement 0.776). official_agreement.py: v6 vs v5 0.968, v6 vs pre-merge 0.722. handover_vs_dead.py: 0 near D by construction of merge. half_split.py: halves consistent.

