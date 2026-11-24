# Results

Every number here is produced by a script in `research/` or `emotif/` and checked by a test in `tests/`. Accuracy is **balanced accuracy** (chance = 1 / number of classes). LOSO = leave-one-subject-out: the model is tested on a person it never saw.

## 1. Artifact repair against ground truth (`research/denoise_benchmark.py` → `results/denoise/`)

Simulated multichannel recordings built from EEGdenoiseNet test-split EEG, with real eye (EOG) and muscle (EMG) artifacts plus electrode pops placed where they occur on the head, at SNR −7 to +2 dB. In the second half of every recording alpha power genuinely doubles, a brain-state change that cleaning must keep. 20 recordings × 60 s per layout; threshold k chosen on separate recordings (seed 100), reported on fresh ones (seed 0).

| Layout | Artifact error, filter only | ASR (k = 20) | Change | Windows kept, ASR | Windows kept, amplitude rejection | Error on clean windows, filter → ASR |
|---|---|---|---|---|---|---|
| Muse (4 ch) | 0.887 | 0.660 | −26 % | 100 % | 89 % | 0.022 → 0.021 |
| OpenBCI (8 ch) | 0.688 | 0.488 | −29 % | 100 % | 86 % | 0.019 → 0.018 |
| Emotiv (14 ch) | 0.499 | 0.423 | −15 % | 100 % | 86 % | 0.016 → 0.015 |
| Lab cap (32 ch) | 0.380 | 0.332 | −13 % | 100 % | 85 % | 0.007 → 0.006 |

Error = RMS(estimate − truth) / RMS(truth), mean over 2 s windows.

## 2. Learned artifact detector (`research/train_artifact.py` → `results/artifact_detector/`)

A 64-32 neural network on 11 amplitude-free features per channel per 2 s window, each relative to that channel's resting baseline. Trained on 18,000 semi-synthetic windows from 600 simulated headsets; half of all windows also carry a genuine brain-state change (one rhythm scaled 0.4–3×, amplitude 0.7–1.5×). EEG, EOG and EMG source epochs were split 70/15/15 before mixing.

| Held-out test (9,000 windows) | Value |
|---|---|
| Accuracy, clean / eye / muscle | 0.934 (256 Hz 0.946, 128 Hz 0.924) |
| Contamination caught (sensitivity) | 0.895 |
| Clean windows wrongly flagged, at rest | 0.8 % |
| Clean windows wrongly flagged, during a genuine state change | 2.6 % (previous version: 5.4 %) |

## 3. Artifact-leakage audit on real Muse data (`python -m emotif.audit` → `results/audit/`)

Bird et al. (2018): 4 people, Muse, relaxed / neutral / concentrating, 24 recordings, 1,078 windows kept by every pipeline. Same windows, same LOSO split, logistic regression. p from 1,000 permutations that shuffle labels between whole recordings within each person; 95 % CI from resampling recordings.

| Decoder | Balanced accuracy | 95 % CI | p |
|---|---|---|---|
| EEG, filter only | 0.689 | 0.581–0.787 | 0.001 |
| EEG, cleaned (ASR + detector) | 0.592 | 0.520–0.670 | 0.001 |
| **Face only: blink + muscle indices** (3 numbers per electrode, no brain rhythms) | **0.693** | 0.588–0.794 | 0.001 |
| Face only: detector p(eye), p(muscle) | 0.694 | 0.598–0.788 | 0.001 |
| Cleaned EEG minus everything the face explains | 0.414 | 0.344–0.480 | 0.056 |
| Face indices + cleaned EEG | 0.686 | 0.583–0.783 | 0.001 |

- **Leakage ratio 1.01**: the face indices alone reach the filtered-EEG decoder's full gain over chance.
- After removing what the face explains, the cleaned EEG is not significantly above chance (p = 0.056). On this dataset, nothing the decoder learns *across people* can be separated from facial activity.
- The permutation null's 95th percentile is 0.42–0.47, not 0.33: with few recordings per person, "above chance" must be judged against the permutation null, not 1/3.
- Known-answer check (`tests/test_cleaning.py`): on simulated data where the label is in the brain (alpha), the audit finds face indices at chance and face-partialled EEG at 0.98 (p = 0.025); where the label is in the face (jaw muscle + blinks), face indices 0.96 and partialled EEG at chance. In 1 of 6 simulated face-driven datasets partialling left a small significant residual (0.58), so the partialled score is a conservative test, not a perfect one.

## 3b. The same audit on GAMEEMO (`python -m emotif.audit --data gameemo_raw:… --perms 200` → `results/audit_gameemo_*`)

GAMEEMO (Alakus et al. 2020): 28 people, 14-channel Emotiv, four 5-minute games each (boring, calm, horror, funny), 112 recordings, 16,456 windows kept by every pipeline. Labels are each game's intended emotion (valence: calm and funny positive; arousal: horror and funny high), not self-report. LOSO, logistic regression; p from 200 recording-level permutations (smallest possible p = 0.005).

| Decoder | Valence | p | Arousal | p |
|---|---|---|---|---|
| EEG, filter only | 0.591 (0.573–0.608) | 0.005 | 0.618 (0.599–0.637) | 0.005 |
| EEG, cleaned (ASR + detector) | 0.582 (0.563–0.600) | 0.005 | 0.605 (0.586–0.624) | 0.005 |
| **Face only: blink + muscle indices** | **0.575** (0.549–0.600) | 0.005 | **0.625** (0.603–0.647) | 0.005 |
| Face only: detector p(eye), p(muscle) | 0.563 (0.530–0.597) | 0.015 | 0.593 (0.571–0.615) | 0.005 |
| **Cleaned EEG minus everything the face explains** | **0.569** (0.554–0.585) | 0.005 | **0.557** (0.540–0.574) | 0.005 |
| Face indices + cleaned EEG | 0.605 (0.585–0.622) | 0.005 | 0.651 (0.633–0.668) | 0.005 |
| Leakage ratio | 0.83 | | 1.06 | |

- On a 14-channel headset with 28 people, the face still explains most of the decoder's gain (83 % for valence, all of it for arousal), but **face-partialled EEG stays above chance for both** (0.569 and 0.557). Unlike the 4-person Muse pilot, part of what is decoded here is not facial activity.
- Face and EEG together beat either alone, so they carry partly different information.
- Arousal separates the first two games from the last two (boring, calm vs horror, funny). If the games were played in a fixed order, time on task is a second confound for arousal; check the play order in the dataset paper before interpreting it.

## 3c. Brain-only decoder (`research/brain_decoder.py`, `research/brain_map.py` → `results/brain_gameemo_v2/`)

The decoder Emotif ships (`train.py --model brain`): every window's face indices are computed live (causal 0–3 Hz eye component, 20–30 and 30–45 Hz power, relative to the person's median, pooled into the 17 regions like the EEG); everything they and their squares predict is subtracted from the cleaned EEG features (ridge, fitted on the training people); logistic regression decodes the rest. Same GAMEEMO windows and LOSO split as 3b. **Face share** = R² of the decoder's continuous output explained by the audit's face indices (zero-phase, per electrode), which the brain-only decoder never uses.

| Decoder | Valence, 2 s window | Valence, 30 s | Arousal, 2 s | Arousal, 30 s | Face share (valence / arousal) |
|---|---|---|---|---|---|
| Standard, filter only | 0.591 | 0.685 | 0.618 | 0.703 | 0.52 / 0.52 |
| Standard, cleaned | 0.582 | 0.670 | 0.605 | 0.693 | 0.29 / 0.36 |
| **Emotif brain-only** | **0.569** | **0.655** | **0.564** | **0.642** | **0.06 / 0.09** |
| Audit reference (offline partialling; needs every test window's offline face indices) | 0.569 | 0.664 | 0.557 | 0.619 | 0.00 / 0.00 (same indices) |
| Erase every face-correlated direction | 0.546 | 0.620 | 0.521 | 0.567 | 0.00 / 0.00 (same indices) |

Chance 0.5. 30 s = class probabilities averaged over the last 15 windows of a recording (the app smooths with a Kalman filter instead).

- The brain-only decoder cuts the face's share of the output 4–5× compared with the cleaned standard decoder, measured with a face measure it never saw.
- Permutation test (100 recording-level shuffles, the whole LOSO rerun each time): brain-only valence p = 0.0099, the smallest p 100 shuffles can give. Arousal (50 shuffles): p = 0.0196, again the smallest possible. Both are above chance on new people after the face is removed.
- Guard: when the largest standardized face index exceeds the 99.5th percentile seen in training (10.2 SD), the app holds the brain-only output. On held-out people it holds 0.7 % of windows; accuracy on the rest is unchanged (0.570 / 0.564).
- Known answer (`tests/test_train.py`): synthetic people whose face activity leaks into every feature through a shared, person-varying mixing (linear and curved). Label in the brain features: standard 0.82, brain-only 0.86. Label only in the face: standard 0.96, brain-only 0.51 (chance 0.5), and the face-driven variance of its output falls from 0.20 to 0.0007.
- **Valence survives face removal**: 0.655 vs 0.670 over 30 s. **Arousal keeps a face component**: 0.642 vs 0.693.
- The trained model (`models/gameemo_brain.json`, with montage augmentation): LOSO valence 0.567, arousal 0.567 on the full 14-channel layout; on Muse positions 0.534 and 0.536; calibration error 0.033 → 0.018 and 0.039 → 0.018.
- **What it reads** (`fig_brain_map.png`): the largest stable weights (same sign in 100 % of 200 refits on resampled people) are, for pleasant states, *lower* alpha at F7 (left frontal) and higher theta at F4; for high arousal, *lower* alpha at F4 (right frontal). Lower alpha means more cortical activity; more left-frontal activity with positive, approach-related affect is the frontal-asymmetry pattern (Davidson 1992), and alpha falls as arousal rises. The decoder found these without being told; it was given 137 features and no hypothesis. Some gamma weights remain (F7, F8 for valence): above 30 Hz, residual muscle can't be ruled out.
- Muse pilot (4 people, 3 arousal classes, chance 0.333): brain-only 0.482 vs cleaned standard 0.592 and filter-only 0.689; face share of the output 0.10 vs 0.35 and 0.79. On 4 forehead/ear electrodes most of what a standard decoder reads is face; what remains after removing it is modest.

Same data, cleaning pipelines with montage augmentation (`research/confound_analysis.py` → `results/confound/`): filter 0.661; + amplitude rejection 0.706 (76 % of windows kept); + ASR 0.617; + ASR + detector 0.564 (88 % kept).

## 3d. More states on the Emotiv Insight layout (`research/states.py` → `results/states/`, `fig_states.png`)
Electrodes: AF3, AF4, T7, T8, Pz (the Insight). Datasets and their checks: `docs/DATASETS.md`. Everything is evaluated on people the model never saw.

**Acute stress, within a person** (17 people recorded with a real Insight; horror video vs relaxing music; 2 s windows; 200 recording-label permutations):

| Decoder | Per window (30 s) | Horror > music in | Wilcoxon p | Face R² of output | Decodes recording *order* (control) |
|---|---|---|---|---|---|
| Standard, filter only | 0.535 (0.505), perm p 0.18 | 12/17 | 0.098 | 0.47 | 0.542, perm p 0.10 |
| Standard, cleaned | 0.564 (0.520), perm p 0.020 | 14/17 | 0.002 | 0.24 | **0.548, perm p 0.030**; earlier recording rated lower in 13/17 (p 0.015) |
| Emotif brain-only | 0.522 (0.501), perm p 0.090 | **13/17**, all 5 who heard music second | **0.002** (dz 0.91) | 0.12 | 0.510, perm p 0.15; 10/17, p 0.52 |

Horror came second for 12 of 17 people, so a decoder that tracks time would also look like a stress decoder. The standard cleaned decoder does track order; the brain-only one does not, and its horror > music difference holds in every person who heard the music second. Per 2-second window it is near chance: stress shows as a person's slow shift, not window by window.

**Transfer across labs.** The GAMEEMO brain-only decoder (four shared electrodes AF3, AF4, T7, T8), applied without retraining: pre-registered direction horror < music on valence and horror > music on arousal. Valence went the *opposite* way (horror rated more pleasant in 13/17, Wilcoxon p 0.003); arousal in 11/17 (p 0.35). The horror recordings had lower left-frontal (AF3) relative alpha than the music recordings (median −0.52 log units per person), and lower left-frontal alpha is exactly what the GAMEEMO decoder reads as pleasant (`fig_brain_map.png`): in these data it marks engagement, not pleasantness. Valence/arousal labels from one stimulus set don't define a general emotion axis; the stress head is therefore trained on the Insight stress data directly (`models/insight_brain.json`).

**Trait screens from 5 minutes of rest** (one label per person; gain-free relative band power; leave-one-person-out logistic regression; 1000 person-label permutations):

| Screen | n (+/−) | Standard AUC (perm p) | its face R² | Brain-only AUC (perm p) | Pre-registered markers (Holm) | Windows split at random → by person (random forest) |
|---|---|---|---|---|---|---|
| Depression, PHQ-9 ≥ 20 vs ≤ 4 | 31 (18/13) | 0.31 (0.87) | 0.16 | 0.46 (0.58) | FAA, frontal alpha, frontal theta: all p ≥ 0.41 | 0.82 → 0.55 |
| Anxiety, GAD-7 ≥ 15 vs ≤ 4 | 38 (23/15) | 0.60 (0.20) | 0.49 | 0.54 (0.36) | frontal beta AUC 0.68, p 0.064 (Holm 0.19); FAA, alpha n.s. | 0.82 → 0.50 |
| Sleepiness, VAS ≥ 5 vs < 5 | 49 (27/22) | 0.68 (**0.042**) | **0.47** | 0.58 (0.21) | theta, alpha, delta: n.s.; Spearman with VAS |ρ| ≤ 0.13 | 0.84 → 0.59 |

No brain-only screen is above chance. The one significant standard result (sleepiness) is about half explained by eye and jaw activity. Splitting windows at random, so the same person is in training and test, makes all three look 82–84 % decodable; that is identification of the person, not of the condition (Saeb et al. 2017).

With every electrode of the original caps (14-channel EPOC+, 19-channel 10-20), the result is the same: brain-only AUC 0.32 / 0.56 / 0.49 (depression / anxiety / sleepiness; permutation p 0.89 / 0.28 / 0.48), standard 0.18 / 0.54 / 0.59 (p 0.98 / 0.36 / 0.20; the standard sleepiness output is 59 % face), and random window splits reach 0.92 / 0.90 / 0.86 against 0.53 / 0.50 / 0.56 split by person. More electrodes did not rescue the screens. Sample sizes of 31–49 give wide error bars (Varoquaux 2018); these results rule out large effects, not small ones.

## 3e. Why published accuracies on these datasets are so high (`research/leaderboard.py` → `results/leaderboard/`, `fig_leaderboard.png`)
Same windows and features as 3c–3d, evaluated the way many papers do (windows split at random) and with controls. Random forest (300 trees), balanced accuracy, 2 s windows. "New people" = 7-fold split with all windows of a person in one fold. Published numbers for these datasets: `docs/PUBLISHED_ACCURACIES.md`.

| Dataset (chance) | Random window split | **Same split, scrambled labels** | Same split, face only | Later 30 % of each recording | New people | New people, face only | New people, brain-only | Which recording? (chance) |
|---|---|---|---|---|---|---|---|---|
| GAMEEMO, 4 games (0.25) | 0.86 | 0.38 | 0.85 | 0.55 | 0.88 | **0.84** | 0.75 | 0.23 (0.009) |
| Stress, Insight (0.5) | 0.76 | **0.75** | 0.76 | 0.67 | 0.58 | **0.60** | 0.56 | 0.51 (0.029) |
| Depression PHQ-9 (0.5) | 0.83 | **0.83** | 0.60 | 0.80 | 0.55 | 0.50 | — | 0.68 (0.032) |
| Anxiety GAD-7 (0.5) | 0.82 | **0.81** | 0.58 | 0.79 | 0.49 | 0.47 | — | 0.66 (0.026) |
| Sleepiness VAS (0.5) | 0.84 | **0.82** | 0.59 | 0.80 | 0.54 | 0.48 | — | 0.69 (0.020) |

*Scrambled labels*: every recording gets a random class (balanced, independent of its true label; 3 draws averaged). The labels mean nothing, so accuracy above chance is recognition of the recording.
- **Stress and the three trait screens: the random-split accuracy is recording identity.** Scrambled labels score the same as the real ones (0.75 vs 0.76, 0.83 vs 0.83, 0.81 vs 0.82, 0.82 vs 0.84). One 2 s window identifies its recording 51–69 % of the time (chance 2–3 %). The depression paper on this dataset reports 98.43 % with 5-fold CV on 5 s samples.
- **GAMEEMO: the four games are separable for new people, but mostly by the face.** Eye and muscle indices alone (3 per electrode, no brain rhythms) name the game for new people 84 % of the time (chance 25 %), close to the full EEG's 88 %. Removing what the face indices predict (linearly, with squares) leaves 0.75 for a random forest, which can still exploit nonlinear face structure; the linear brain-only decoder is at 0.35 four-class. Training on the first 70 % of each game and testing on the last 30 % drops to 0.55, so the separating pattern changes over the course of each game and is shared across players: consistent with game-specific, time-locked eye and face behaviour (and possibly time on task, if the games were played in a fixed order; the dataset page doesn't say). Published GAMEEMO results of 94–99.99 % (random or unstated splits) therefore say little about emotion.
- **Trait screens: the face doesn't name the trait either.** Face indices alone score 0.58–0.60 on a random split (the face also fingerprints a recording) and 0.47–0.50 for new people (chance 0.5), so nothing on these three datasets, brain or face, carries to a new person.
- Logistic regression shows the same pattern at lower levels (stress: random 0.64, scrambled 0.57, new people 0.53, face only on new people 0.62).
- Tool: the same checks as a package, `facecheck/` (`facecheck audit data.npz`), with known-answer tests.

## 3f. Closed-loop audit: what the face does to the world (`research/closed_loop_audit.py` → `results/closed_loop/`, `fig_closed_loop.png`)
Sections 3–3e audit the classifier. Here each held-out person's real recording is replayed through the app's whole loop, ported from `app/src` and checked against the TypeScript (focus/relax control, `slewToward`, director, `moodWeather`: identical to ≤ 1e-15 on random input): 2 s windows every 0.25 s → 200 µV amplitude gate after ASR (app default) → decoder → Kalman smoothing (q 0.002) → director (0.25 per s) → world slew (τ 1.2 s, 0.5 per s, 60 frames/s) → cloud, storm, fog, saturation and the displayed weather (3 s dwell, fade). The focus/relax control (AlphaControl: posterior alpha vs baseline, threshold 0.7, dwell 1 s, τ 1.5 s, hysteresis, hold) runs on the same windows. GAMEEMO: LOSO models fitted on the other 27 people; the pipeline reproduces section 3c exactly (training rows identical to the 3c cache; balanced accuracy 0.591 / 0.582 / 0.569 valence, 0.618 / 0.605 / 0.564 arousal; face share 0.52 / 0.29 / 0.06 and 0.52 / 0.36 / 0.09). Insight stress data: the GAMEEMO models at AF3, AF4, T7, T8, fitted on all 28 GAMEEMO people; the control reads Pz. Normalization (z-scores, detector baseline, face centre, ASR) comes from each person's clean session and is held fixed when artifacts are added. 554 min of GAMEEMO and 97 min of Insight data replayed; seed 0.

**1. Face share of the world (GAMEEMO).** Out-of-person R² of each signal from an independent face measure (audit face indices per electrode, averaged causally over the last 2, 8 and 30 s, with squares; ridge fitted on the other people). p: 200 permutations that give each recording the face series of another person's recording. Per-person comparison: Wilcoxon, n = 28.

| Signal | Standard, filter only | Standard, cleaned | Brain-only | Cleaned > brain-only |
|---|---|---|---|---|
| Decoder output per window, valence / arousal | 0.33 / 0.35 | 0.20 / 0.26 | < 0 (−0.16 / −0.08) | |
| Valence the app shows | 0.42 (p 0.005) | 0.29 (p 0.005) | < 0 (p 0.93) | 28/28 people, p 7e-9 |
| Arousal the app shows | 0.44 (p 0.005) | 0.34 (p 0.005) | < 0 (p 0.46) | 26/28, p 3e-7 |
| Cloud cover / storm / fog | 0.29 / 0.26 / 0.29 | 0.22 / 0.17 / 0.22 | < 0 (p ≥ 0.23) | cloud 24/28, p 7e-5 |
| Colour saturation | 0.43 | 0.30 | < 0 (p 0.95) | 28/28, p 7e-9 |
| Displayed weather (5 kinds) | 0.06 | 0.03 | < 0 (−0.07) | 24/28, p 3e-5 |
| Focus/relax control level | 0.24 (no face hold) | | 0.24 (with face hold) | 12/28, p 0.10 |

- Smoothing does not wash the face out of the world; it concentrates it. The standard decoders' face share is *higher* in what the world shows (0.29–0.44) than per window (0.20–0.35), because the face differs between whole games and the loop integrates it. With the brain-only decoder no world parameter is predictable from the face history in new people (all R² < 0, none above the permutation null).
- The weather is sticky (3 s dwell, quadrant thresholds), so little of its variance is explained by anything (≤ 0.06). For the brain-only decoder the permutation p is 0.005 although R² is −0.07: swapping faces between recordings makes the regression worse still (null mean −0.15); it is not evidence that the face predicts its weather.
- **Unexpected:** the focus/relax control's level is 24 % face-explained (p 0.005), with or without the face hold. The hold is set for extreme face activity (z > 8.7 here) and fires on < 1 % of clean windows, so it does not remove ordinary co-variation between face indices and posterior alpha. Posterior alpha may genuinely co-vary with eye activity (alpha rises when the eyes are still or closed and drops with visual engagement), so this is not necessarily contamination, but the control is not face-independent in the way the affect decoder is.

**2. Counterfactual artifact injection.** Into real recordings, 30 s with / 30 s without, alternating: blinks (Gaussian, σ 60–100 ms, every 2.5 ± 0.5 s, peak 100/150/200 µV at AF3/AF4, scaled over the scalp), jaw clenches (20–45 Hz noise, 1.5 s on / 1 s off, RMS 10/20/40 µV at T7/T8; and 13–45 Hz at 20 µV), and genuine brain changes (8–13 Hz component of every channel ×2 or ×0.5). Compared with the same recording replayed without injection. GAMEEMO, app settings, mean over people; shown state = mean |change| of displayed valence and arousal; weather = % of injected time the displayed weather differs.

| Injection | Shown state moved: filter only / cleaned / brain-only | Weather differs | Windows decoded | Cleaned vs brain-only, valence (Wilcoxon) |
|---|---|---|---|---|
| Blinks 100 µV | 0.14 / 0.12 / 0.10 | 29 / 20 / 17 % | 66 % | p 0.025 |
| Blinks 200 µV | 0.15 / 0.13 / 0.16 | 26 / 22 / 22 % | 45 % | p 0.13; arousal: brain-only moves *more* (0.21 vs 0.14, p 0.018) |
| Jaw 10 µV | 0.27 / 0.19 / 0.13 | 55 / 35 / 22 % | 67 % | p 5e-4 |
| Jaw 20 µV | 0.46 / 0.21 / 0.16 | 67 / 38 / 25 % | 56 % | p 1.5e-8 |
| Jaw 40 µV | 0.40 / 0.22 / 0.17 | 54 / 36 / 25 % | 3 % | p 1.5e-8 |
| Jaw 13–45 Hz, 20 µV | 0.47 / 0.21 / 0.16 | 67 / 41 / 25 % | 60 % | p 1.5e-8 |
| **Alpha ×2 (genuine)** | 0.19 / 0.23 / 0.22 | 57 / 58 / 61 % | 66 % | brain-only moves the weather more, p 0.0015 |
| **Alpha ×0.5 (genuine)** | 0.17 / 0.19 / 0.16 | 53 / 52 / 52 % | 70 % | |

- Jaw clenches of 20 µV and more move the standard (filter-only) world about 3× more than the brain-only world, in all 28 people (cleaned > brain-only in 27/28); cleaning halves it and the brain-only decoder removes about another quarter. A genuine alpha change moves the brain-only world as much as the standard one (weather changes in 61 % of the time), so the brain-only decoder is not simply insensitive: per unit of world movement, jaw (20 µV) / alpha ×2 is 2.4 for the filter-only decoder, 0.9 cleaned, 0.7 brain-only.
- Blinks: small effects for every decoder, because ASR and the 200 µV gate already act on them (at 200 µV only 45 % of injected windows are decoded, against 69 % of clean windows; the world holds). Brain-only is not better than the cleaned standard decoder against large blinks, and for 200 µV blinks its arousal moves more (overcorrection: the ridge subtraction extrapolates beyond the blink sizes it was fitted on while the hold, z > 8.7, rarely fires). Known-answer reproduction: in `tests/test_train.py` the same overcorrection appeared when synthetic training blinks were all one size.
- The residual movement under jaw injection (0.16) is not zero: brain-only still lets about a third of the cleaned decoder's artifact-driven movement through.
- Without the amplitude gate (`rejectUv` 0) all windows are decoded and the filter-only world moves up to 0.68 under 40 µV clenches (weather differs 75 %); cleaned 0.21, brain-only 0.18 (all in `closed_loop.json`, `no_gate`).
- Insight stress data (transferred models; 17 people, short recordings): same ordering for clenches (jaw 20 µV: 0.19 / 0.11 / 0.10; weather 47 / 10 / 7 %), but cleaned vs brain-only differences are small and mostly not significant (n = 17; p 0.03–0.57). Genuine alpha ×2 moves the brain-only world more than the cleaned standard's valence (p 5e-4). Per-window face share replicates the ordering (valence 0.44 / 0.33 / 0.12, arousal 0.36 / 0.27 / 0.19); the out-of-person world regression does not generalize across these 17 short recordings (all R² < 0), so the world-level share is not measurable on this dataset.

**3. False commands of the focus/relax control.** Commands per minute during injected epochs minus the same epochs replayed clean (GAMEEMO, app settings).

| Injection | Added commands/min, with face hold / without | Time the control's state differs from clean | Held |
|---|---|---|---|
| Blinks 100 / 150 / 200 µV | −0.07 / −0.26 / −0.48 (same without hold) | 3 / 7 / 11 % | 34 / 45 / 56 % |
| Jaw 10 / 20 / 40 µV | 0.00 / −0.13 / −1.06 | 5 / 12 / 21 % | 33 / 44 / 97 % |
| Alpha ×2 (genuine relax) | +0.53 (relax 1.59/min vs 0.44 clean) | 70 % | 34 % |
| Alpha ×0.5 (genuine focus) | +0.74 (focus 1.79/min vs 0.66 clean) | 73 % | 31 % |

- Face artifacts did not create false commands: they removed commands, because the amplitude gate, the detector and ASR make the control hold or see no change in posterior alpha. The face hold added nothing measurable on top of the gate (identical rates; p ≥ 0.04 for one condition, ≥ 0.11 otherwise).
- Genuine alpha changes were detected: the right command appeared in 95 % (×2, relax) and 96 % (×0.5, focus) of injected 30 s epochs, against 29–30 % of the same epochs without injection; 85 % with 150 µV blinks on top. Insight (Pz only): 72–74 %, 51 % with blinks (the gate holds 61–83 % of windows there); 99 % without the gate.
- **Limitation found:** on clean recordings the control issues 1.2 commands per minute (Insight 1.3) and is in a command state 20 % (25 %) of the time without any instruction: natural alpha fluctuations cross the 0.7 threshold. These are false commands from the brain, not the face. Neither dataset has eyes-closed or instructed segments, so true and false commands can't be told apart on real data beyond the alpha-scaled controls.

Limits: synthetic artifacts are idealized (fixed spatial gains, Gaussian blinks, band-limited noise; real EMG reaches below 20 Hz, tested only by the 13–45 Hz variant); normalization is frozen at the clean session (the app's default "auto" mode keeps updating it, which would partly absorb sustained artifacts); weather is simulated from the state machine, not rendered; temperature calibration of the shipped models is not applied (it changes only the Kalman weight); one seed.

## 4. Models (`train.py`, `research/benchmark.py` → `results/muse_mental/`)

| Bird Muse, arousal, cleaned | Within-subject (held-out recording) | LOSO |
|---|---|---|
| Logistic regression | 0.444 | 0.564 |
| MLP (64) | 0.382 | 0.564 |
| Region-graph diffusion + logistic | 0.421 | 0.553 |

- Within-subject accuracy (train on one recording of each state, test on the other) is near chance while LOSO is not. What transfers across people here is a shared facial pattern; what changes between two recordings of the same person is mostly electrode contact.
- The region-graph model (Gaussian-kernel scalp graph, mask-aware diffusion that fills uncovered regions from covered ones) matches logistic regression on this 4-electrode data; it is built for transferring dense-cap models to sparse headsets, which needs GAMEEMO / DREAMER / DEAP to test.
- Montage augmentation: no significant effect on this pilot (Wilcoxon p ≥ 0.125 for every layout and model).
- **Calibration**: the raw model is over-confident. Temperature scaling (T = 2.89) cuts expected calibration error from 0.170 to 0.078 on held-out people. The app uses the calibrated confidence to weight its smoothing.

**GAMEEMO models** (`train.py --data gameemo_raw:…`, artifact-cleaned, LOSO, chance 0.5; bundled in the app as *GAMEEMO (Emotiv, 28 people)*):

| Layout the model is tested on | Valence | Arousal |
|---|---|---|
| Full 14-channel Emotiv | 0.575 | 0.603 |
| Muse positions only | 0.563 | 0.566 |
| OpenBCI Cyton positions | 0.560 | 0.576 |
| Neurosity Crown | 0.542 | 0.587 |
| Unicorn | 0.549 | 0.578 |
| BrainBit | 0.540 | 0.568 |
| 2 frontal electrodes | 0.522 | 0.524 |

- Trained once on the 14-channel data with montage augmentation, the same model keeps most of its above-chance accuracy on consumer layouts it never saw (Muse: 0.563 vs 0.575 valence).
- Calibration (temperature scaling): expected calibration error 0.045 → 0.005 (valence), 0.040 → 0.012 (arousal) on held-out people.
- Within-subject scores aren't reported for GAMEEMO: each recording has one label, so holding out a recording leaves its class with a single training recording and biases predictions toward the other class (scores fall below chance by construction).

## 5. Live system on real recordings (`tests/test_replay.mjs`)

| Person | Speed | Per decoded window | Decoded arousal, relaxed → concentrating |
|---|---|---|---|
| a | 194× real time | 0.64 ms (p99 4.9 ms) | −0.41 → +0.32 |
| b | 254× | 0.64 ms (p99 4.4 ms) | −0.41 → +0.22 |
| c | 315× | 0.59 ms (p99 1.2 ms) | −0.25 → +0.10 |
| d | 348× | 0.58 ms (p99 1.2 ms) | −0.14 → +0.06 |

Latency: ASR look-ahead 313 ms, then a 2 s window updated every 0.25 s. These four people were in the training data, so this shows the pipeline runs on real data in real time, not its accuracy (section 3).

## 6. Safety and accessibility

| Check | Result |
|---|---|
| Photosensitive safety: worst-case adversarial decoder driving the world (`tests/test_app.mjs`) | ≤ 1 flash per second (WCAG 2.3.1 limit: 3) |
| axe-core WCAG 2.1 AA audit, 7 views × light and dark themes (`tests/test_browser.py`) | 0 serious or critical violations |
| Keyboard | skip link first in tab order; every control reachable; Esc stops the arm |
| Screen reader | state announced when it holds 5 s, at most every 10 s |
| Sound display | pitch and chord follow valence, pulse follows arousal |

## 7. Hardware and software verification

| Check | Result |
|---|---|
| Python ↔ app agreement: filter, features, region embedding, graph transform, calibrated models, ASR, detector, arm kinematics | ≤ 1e-8 |
| EDF / BDF reader vs MNE-Python (16- and 24-bit, µV / mV / V, extra channels) | identical to 3e-14 µV |
| OpenBCI Cyton decoder (random chunking, garbage bytes, lost packet) | exact |
| Real EDF played end to end in the browser | decoded |
| Supported boards via the server (BrainFlow) | 40, plus any LSL stream |
| World settings with wrong types or impossible values | always clamped to a renderable world (`tests/test_app.mjs`) |
| World: same settings → same world; chunk edges | identical; seamless |
| Bundled models (Muse, GAMEEMO standard, graph and brain-only): Python vs app probabilities | ≤ 1e-9 |
| Face indices (eye, jaw) and regional pooling: Python vs app at 128 / 250 / 256 Hz | ≤ 2e-15 |
| Brain vs face, live: simulated jaw clench in the browser | brain-only output moves less than the standard decoder's (`tests/test_browser.py`) |
| Test suites (`./run_tests.sh`) | 7, all pass |
