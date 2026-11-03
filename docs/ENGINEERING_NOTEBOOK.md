# Engineering notebook: design iterations

Biomedical Engineering judges look for the design cycle: define → build → test → find failure → redesign → retest. These are the real iterations behind Emotif's signal-cleaning system, with the numbers that triggered each one. Keep adding to this file; date each entry in your own notebook.

## Problem and criteria
**Need.** Consumer EEG headsets are noisy. Blinks, muscle tension and electrode movement are often 10–100× larger than brain signals, and they correlate with emotion (people frown when stressed). An affect-driven system that reads muscles instead of the brain is measuring the wrong thing.

**Design criteria.**
1. Run live in a browser on a school laptop (≥ 10× faster than real time).
2. Work on any headset, from 2 to 64 channels at 128–500 Hz.
3. Repair artifacts without discarding data (no frozen display).
4. Leave clean EEG and **genuine brain-state changes** untouched.
5. Be identical offline (training) and live (app), verified numerically.
6. Be validated against ground truth, not just by eye.

## Iteration 1: amplitude rejection
Throw away any 2 s window above 200 µV peak-to-peak.
**Test:** loses 11–15 % of data in the benchmark and 24 % of real Muse windows. Misses muscle activity (small amplitude, wrong frequency). **Fails criteria 3.**

## Iteration 2: Artifact Subspace Reconstruction (ASR)
Learn the statistics of the resting baseline, then find bursts in a rotated "component" space and rebuild them from the other channels (Mullen et al. 2015).
**Test (ground-truth benchmark):** repairs the median window, but on 8- and 32-channel layouts the *mean* error for eye and muscle artifacts was **up to 13× worse than doing nothing**.
**Root cause:** the baseline covariance was an element-wise median of window covariances, which is not guaranteed to be a valid (positive-definite) covariance. One eigenvalue was ~0 and the reconstruction blew up (matrix condition number 250,000).
**Fix:** geometric median (Weiszfeld), which is always a valid covariance; a small ridge for rank-deficient data (average reference); and an energy guard so a repaired block can never have more energy than the input.
**Retest:** artifact error now falls on every layout; average-referenced data is never amplified (test in `tests/test_cleaning.py`).

## Iteration 3: ASR erased real emotions
**Test:** the app's own unit test steered the simulated person to a *negative* state after the baseline, and the decoder reported **+1.00 / +1.00**, the opposite. ASR treated the genuine 28 % alpha increase as an artifact and "repaired" it away.
**Fix:** a component is never repaired unless it exceeds 2.5× its baseline amplitude. Real alpha can double; blinks and muscle bursts are far larger. The benchmark gained a genuine alpha-doubling state change so this failure can't come back unnoticed.
**Retest:** negative state decoded −1.00 / −1.00; the alpha-doubling segment's distortion is 0.0000.

## Iteration 4: choosing the threshold k
Tuned on one set of simulated recordings (seed 100), reported on fresh ones (seed 0) to avoid fooling ourselves.
k = 5 or 10 repaired more but distorted clean, state-changing windows (0.07–0.09 error vs 0.02). **k = 20** kept clean windows at the no-cleaning level while cutting artifact error 13–29 %. That is also the default the ASR authors recommend; we reached it independently.

## Iteration 5: an AI detector that failed on real hardware
Trained a neural network (EEGdenoiseNet) to label each channel clean / eye / muscle: **95.6 % accurate on lab data.**
**Test on real Muse recordings:** it called 46–79 % of channels "muscle", even at rest.
**Root cause:** domain shift. Dry consumer electrodes have a higher noise floor, so clean Muse EEG looks "high-frequency" to a model trained on lab EEG.
**Fix:** judge each window **relative to that channel's own resting baseline**, and train on 600 simulated headsets (random noise floor, analogue filter, mains hum, 128 or 256 Hz) so the network learns deviations rather than absolute spectra.
**Retest:** 93.8 % held-out accuracy, 0.8 % false alarms; on real Muse data, 6 % muscle at rest and 31 % when concentrating, which matches physiology.

## Iteration 6: the confound discovery
With full cleaning, real Muse decoding *dropped* from 0.71 to 0.55. Instead of turning cleaning off, we tested why: a classifier given **only** eye/muscle evidence scored 0.68. The original accuracy was mostly facial activity. The design decision was to keep cleaning on for emotion, and to use eye and jaw signals on purpose for robot-arm control, where they're reliable and honest.

## Iteration 7: the detector called real brain changes "artifacts"
**Test:** a simulated person whose alpha doubles (no artifacts at all) had 20 % of channel-windows masked as eye artifacts, against 9 % at baseline. On held-out data the detector flagged 5.4 % of clean windows that contained a genuine rhythm change, six times its resting rate (0.9 %).
**Root cause:** the training data only had resting EEG as "clean", so any change from baseline looked suspicious.
**Fix:** half of all training windows (clean and contaminated) now carry a genuine state change: one rhythm (theta, alpha or beta) scaled 0.4–3× and overall amplitude 0.7–1.5×, still labelled by artifact class.
**Retest:** 2.6 % false alarms during state changes, 0.8 % at rest; accuracy 93.4 %, sensitivity 89.5 % (unchanged).

## Iteration 8: the "face-only" control wasn't face-only
**Test:** the brain-or-face analysis used the detector's eye/muscle probabilities as the face evidence. A known-answer simulation where the label lived only in the brain (alpha) gave this "face-only" classifier 0.94–0.99. It was reading the brain too, because the detector's inputs include band-power shape.
**Fix:** three physiological face indices per electrode that a theta or alpha change cannot move: peak-to-peak of the 0.5–3 Hz signal (blinks, eye movements) and power at 20–30 and 30–45 Hz (muscle; Whitham et al. 2007 showed scalp power above 20 Hz is mostly muscle).
**Second failure:** the eye index still rose with alpha (log 2.30 → 2.69). Filtering each 2 s window separately leaked edge transients of the strong alpha rhythm into the 0.5–3 Hz band. Filtering whole recordings instead fixed it (2.066 → 2.065).
**Retest:** with the label in the brain, the face indices are at chance (0.48–0.52, never significant) in all 7 simulations run.

## Iteration 9: removing the face from the EEG
**Goal:** a test of whether cleaned EEG holds information the face can't explain. Regress the EEG features on the face indices (training people only) and decode from what is left.
**Test:** with a straight-line fit, simulated face-driven labels still leaked through (partialled accuracy 0.58–0.61).
**Fix:** add squared face terms (an artifact's effect on log band power is curved) and the 20–30 Hz muscle band.
**Retest:** face-driven simulations drop from 0.89–0.96 to 0.49–0.58; brain-driven ones stay at 0.97–0.99. Partialling also removes brain activity that moves together with the face, so it is a conservative test.
**Real data:** on the Muse dataset the partialled EEG scored 0.414 (p = 0.056). The leakage ratio was 1.01: the face alone explains the whole cross-person result.

## Iteration 10: a bug in our own test
The detector's "held-out" test rebuilt the train/validation/test split with a fresh random generator per signal type, while training used one generator for all three. For EOG and EMG the "test" epochs were therefore partly training epochs. Fixed by replaying the exact generator sequence; the held-out numbers above come from the corrected split.

## Iteration 11: honest confidence
The decoder said "90 % sure" far more often than it was right (expected calibration error 0.170). Temperature scaling fitted on out-of-fold predictions (T = 2.89) brought it to 0.078 on held-out people. The app weights its smoothing by this confidence, so an unsure decoder moves the world less.

## Iteration 12: a world that can't flash
Requirement: people with photosensitive epilepsy are excluded, but the system should be safe anyway. WCAG 2.3.1 allows at most 3 flashes per second. An adversarial test drives the decoder to flip the target the moment brightness has moved 10 %: smoothing alone allows 2 flashes/s; adding a slew limit (0.5 state units per second) brings it to 1. The limiter is now part of the world's update loop.

## Iteration 13: a world driven by the brain, not the prompt
Problem: the world was one fixed landscape with the robot arm in the middle, so the arm (a side feature) took over the view.
Design: the arm moved to its own scene in Arm mode. The Live view became an endless world (24 × 24 chunks streamed around an explorer, 2–6 ms each, seeded so the same settings always give the same world, chunk edges matching exactly). The decoded state sets colour grading, mist, wind, water, weather and the explorer's pace, through the photosensitive slew limit.
Change after review: a first version let a language model design the scenery from a typed prompt. It used the viewer's AI usage and pulled attention away from the brain. Now an optional short theme and an art style set only the scenery, interpreted on the laptop; the brain drives everything that moves. The world is locked during study sessions so mirror, guide and sham show the same world.

## Iteration 14: a brain-only decoder
Problem: the audit showed that on GAMEEMO the face explains most of a standard decoder's gain (leakage ratio 0.83 valence, 1.06 arousal), and on the Muse pilot all of it. Cleaning reduces this but doesn't remove it: 29–36 % of the cleaned decoder's output is still explained by face indices.
Requirement: decode from brain activity only, live, on any headset, with no extra hardware (no EOG/EMG electrodes).
Options tested on GAMEEMO (28 people, leave-one-person-out, same windows):

| Approach | Valence | Arousal | Face share of output | Runs live? |
|---|---|---|---|---|
| Standard, cleaned | 0.582 | 0.605 | 0.29 / 0.36 | yes |
| Erase every face-correlated direction (LEACE, Belrose et al. 2023) | 0.546 | 0.521 | 0.00 | yes |
| Erase only the top 10 / 40 directions | 0.585 / 0.566 | 0.584 / 0.549 | 0.24 / 0.05–0.08 | yes |
| Subtract what each window's measured face predicts | 0.569 | 0.564 | 0.06 / 0.09 | yes, chosen |

Full erasure removes too much brain signal; partial erasure leaves the face or matches the chosen method at best. (The erasure rows are scored with the same face indices they erased, so their face share flatters them; the chosen method is scored with indices it never used.) Subtracting what each window's measured face predicts keeps the most accuracy at a low face share and has a direct meaning: measure the face, remove what it explains. Iterations within it:
- The audit's face indices used a zero-phase filter over whole recordings, which a live system can't run. Replaced by a causal 3 Hz low-pass running continuously from the stream's start, so training and the app compute the same numbers (agreement 1.8e-15 in tests).
- Per-electrode indices don't transfer across headsets. They are now pooled into the 17 scalp regions with the same weights and channel mask as the EEG features.
- Squares of the indices are included, because an artifact's effect on log band power is curved.
- The subtraction is affine, so it folds into the first layer of the decoder: no run-time cost, and the exported model is still one matrix per layer.
- First browser test failed: after a simulated jaw clench the brain-only output moved more than the standard one (0.58 vs 0.17). Diagnosis: repeated clenches in the engine showed both decoders' changes were within the no-clench noise floor (the simulator's clench is mostly removed by ASR and the detector), and the test was measuring right after a recalibration, while the smoothing was still settling. The test was wrong, but it exposed a real risk: a face far beyond the training range makes the learned correction an extrapolation. Fix: a guard that holds the brain-only output when the largest standardized face index exceeds the 99.5th percentile of training (10.2 SD); it holds 0.7 % of a new person's windows and doesn't change accuracy.
Verification: the face share is measured with the audit's indices, which the decoder never used, so it can't pass by matching its own definition. A known-answer test (label in the brain vs only in the face) checks that brain information is kept and face information is not decoded. Python and the app agree to 1e-15.

## Iteration 15: more states, one headset (Emotiv Insight)
Requirement: decode stress, anxiety, depression, anger and poor sleep from the brain, on the Emotiv Insight (AF3, AF4, T7, T8, Pz), and show honestly what works.
- Data search (Mendeley; PhysioNet, Figshare, OpenNeuro unreachable from the build machine): stress recorded with a real Insight; PHQ-9 and GAD-7 screening EEG (EPOC+); resting EEG with sleepiness ratings. No reachable raw-EEG dataset was labelled for anger or for waking insomnia; both are reported as gaps, not guessed.
- Checking the data before trusting it found: one recording uploaded twice under two conditions (dropped); stress tasks recorded months apart with different people (only the horror/music pair is within-person); horror recorded second for 12 of 17 people (added an order-decoding control); every sleepiness volunteer had ESS ≤ 10 (switched the label to the state rating); unlabeled EPOC columns (order confirmed from occipital alpha).
- Within-person stress (17, leave-one-person-out): the standard cleaned decoder separates horror from music but also separates first from second recording just as well, and a quarter of its output is face. The brain-only decoder doesn't track order and is face-free, and its horror-minus-music difference is positive in 13 of 17 people, including all 5 who heard the music second. Per 2-second window it is near chance, so the stress output is a slow, within-person measure.
- Trait screens (one label per person, 5 min of rest): gain-free relative band power, person-level leave-one-out, 1000 permutations, markers fixed from the literature before looking. Brain-only: no screen beats chance on the Insight layout or the full 14/19-channel caps. Standard decoders reach AUC 0.60–0.68 for anxiety and sleepiness, but about half of their output is explained by the face covariates. With windows split at random (common in published work on these datasets) a random forest reaches 0.82–0.92; split by person it falls to 0.50–0.59. Conclusion: these are not brain-decodable traits with these data, and the app does not offer a depression, anxiety or insomnia readout.
- Built: Emotiv Cortex source in the browser (tested against a scripted fake service), named extra model heads (a stress head trained on the Insight data), OSC output for Unreal Engine 5, and a realistic world style whose vegetation is generated from the state at the moment each piece of land appears.

## Iteration 16: detecting every state, one person at a time
Requirement: detect calm, stress, anxiety, anger, low mood and drowsiness on the Insight, from the brain, honestly.
- Constraint from iteration 15: between-person decoding of traits fails; within-person decoding of stress works. Design: calibrate per person.
- Protocol: three rounds × six 1-minute inductions (order from the participant code), 15 s breaks, 0–10 rating of the target state after each. Three rounds, because with two blocks per state the block-permutation test can't go below p ≈ 0.09 even for a perfect decoder (two blocks of one state land on the same shuffled label about 1 time in 11); with three it can.
- Model: brain-only residual (ridge on the face indices and squares) → standardize → shrinkage LDA. Chosen over logistic regression because the covariance is label-free, so 199 permutations × 18 held-out blocks refit only class means: 3 s in the browser instead of minutes. Folds into one softmax layer over the brain-only head input, so the live engine runs it like any head.
- Gate: a state is shown only if its held-out AUC beats the permutations (p ≤ 0.05). Known-answer tests: states written into brain features pass 6/6; states written only into face activity pass 0/6 while the face-only reference reaches 100 %; no signal passes 0/6. Python and app agree to 2e-15.
- Depression and insomnia: not moment-to-moment states, and the public data showed no between-person signal. Daily 3-minute check-ins compare gain-free resting markers with the person's mood and sleep answers over ≥ 7 days (Spearman, day permutations, Holm). Test: a planted link is found; without one nothing survives correction.

## Iteration 17: a world that looks real, and that the brain can move
Requirement: no blocky look; real imagery; the wearer manipulates objects with the brain; every session's world unique.
- Image library (`scripts/build_assets.py`, `app/assets/`, 1.2 MB embedded): photographed grass, dirt, sand and rock with normal maps (Babylon.js Assets, CC BY 4.0), six photographed skies (Poly Haven, CC0, tone-mapped from HDR with a small RGBE reader; one Babylon.js skybox), and procedurally drawn leaves and bark. Checksums are pinned so a changed source is caught.
- Rendering: ground blends the photos by slope, height, wetness and noise at two rotated scales (no visible tiling) with normal maps; trees are bark trunks with crowns of alpha-tested leaf cards (alpha-to-coverage for soft edges); palms have drooping frond cards; the sky is the photograph, turned so its sun sits where the light comes from, with the photo's horizon colour feeding the fog so the land meets the sky.
- Brain control: alpha (8–13 Hz) relative to the wearer's baseline, posterior (Pz on the Insight). Below baseline for 1 s = focus = lift; above for 1 s = relax = place. Hysteresis, hold during facial activity. Tests: dwell, hold, hysteresis, 0 false commands in 2 min of baseline-level noise; browser test lifts and places a crystal.
- Uniqueness: when the resting baseline completes, its absolute spectrum (and the clock) seeds the world, so each session grows different land, plants, sky choice and crystals.

## Verification strategy
Each algorithm exists twice: Python for training and research, TypeScript for the live app. Automated tests run both on identical input and require agreement to 1e-8 or better (face indices and the brain-only decoder included). 7 test suites run with `./run_tests.sh`, including known-answer simulations for the audit, an axe-core accessibility audit, and an independent reader (MNE) for EDF files.
