# Study design notes

These are working notes, not your ISEF research plan. ISEF 2027 does not allow AI-written research plans: write yours yourself from these facts (see `docs/ISEF_CHECKLIST.md`).

**Working title.** Brain, not face: artifact-robust, montage-agnostic EEG affect decoding for consumer headsets, with a sham-controlled closed-loop environment and a brain-driven robot arm.

**Category.** Biomedical Engineering (OCSEF and ISEF BMED): a biomedical signal-processing system, designed, iterated and validated against ground truth (see `ENGINEERING_NOTEBOOK.md`).

## Problem
Most EEG emotion models are trained and tested on one lab's electrode cap. Consumer headsets have 2–16 electrodes in different places, so a published model can't run on them, and every new device needs new data. Separately, emotion-responsive environments are usually shown working but rarely compared with a control that ignores the brain, so it's unclear whether the feedback itself does anything.

## Questions and hypotheses
- **RQ0 (cleaning, done).** Can artifacts be repaired live on consumer headsets without discarding data or erasing genuine brain-state changes, and how much of consumer-EEG "decoding" is really facial activity?
  - Result: ASR (redesigned) cuts ground-truth artifact error 13–29 % with 100 % of data kept and no distortion of clean data; the detector reaches 93.4 % held-out accuracy; on real Muse data three face indices per electrode match the EEG decoder (0.693 vs 0.689) and face-partialled EEG is not above chance (0.414, p = 0.056). See `RESULTS.md`.
- **RQ0b (brain-only decoding, done on GAMEEMO).** Can emotion be decoded on consumer-style EEG after removing everything facial activity explains, and how much accuracy remains?
  - Result: brain-only valence 0.569 per window, 0.655 over 30 s (standard 0.670); arousal 0.564 / 0.642 (standard 0.693); face-explained share of the output 6–9 % vs 29–36 %. See `RESULTS.md`.
  - H0b (Phase 2): during a short block of deliberate blinking and jaw clenching, the standard decoder's output changes more than the brain-only decoder's (paired Wilcoxon across participants); during the emotion blocks, both follow the SAM ratings (Spearman correlation per participant).
- **RQ0c (more states, Emotiv Insight layout, done on public data).** Which states can a 5-electrode headset decode from the brain alone: acute stress, and trait depression, anxiety, sleepiness?
  - Result: stress within a person, yes but slowly: brain-only horror − music difference positive in 13/17 new people (Wilcoxon p = 0.002, dz = 0.91; all 5 who heard the music second), with no ability to decode recording order. Traits, no: brain-only person-level AUC at chance for PHQ-9, GAD-7 and sleepiness on the Insight layout and full caps; standard decoders' signal is about half face; random window splits inflate the same data to 0.82–0.92. See `RESULTS.md` 3d.
  - H0c (Phase 2): within each participant, the brain-only stress output is higher during the arithmetic stressor than during relaxing music (Wilcoxon across participants), and it follows the 0–10 stress rating across blocks (Spearman per participant, median above 0).
- **RQ0d (personal multi-state decoding, the Phase 2 primary question).** After a 28-minute calibration, can a 5-electrode headset tell one person's calm, stress, anxiety, anger, low mood and drowsiness apart from the brain alone, on rounds it never saw?
  - H0d (pre-register): for each state, held-out one-vs-rest AUC > 0.5 across participants (one-sided Wilcoxon, Holm over six states); more participants pass their personal permutation check (p ≤ 0.05) than the 5 % expected by chance (binomial); brain-only accuracy is unrelated to how well the face alone separates the states (Spearman). Analysis: `research/personal_states.py`.
  - Manipulation check: each block's 0–10 rating of the target state is higher than the same rating after other blocks.
  - Secondary (exploratory, 14 days of check-ins): within-person Spearman of resting markers with mood and sleep, Holm-corrected.
- **RQ1 (transfer across headsets).** Does a region-pooled feature space with montage augmentation keep decoding accuracy when a model is tested on electrode layouts it wasn't trained on?
  - H1: With augmentation, balanced accuracy on consumer-headset layouts stays above chance and drops less from the full montage than without augmentation (Wilcoxon signed-rank across held-out subjects, α = 0.05).
- **RQ2 (cross-dataset).** Does pooling datasets recorded on different headsets (GAMEEMO/DREAMER on Emotiv, DEAP on BioSemi) into one model help new subjects?
  - H2: Leave-one-subject-out accuracy on each dataset is at least as high for the pooled model as for the single-dataset model.
- **RQ3 (closed loop).** Does an environment that guides the decoded state toward calm speed recovery after a stressor compared with mirroring and with a sham environment?
  - H3: After the arithmetic stressor, self-rated arousal (SAM) and decoded arousal fall more in the *guide* condition than in *sham* (within-subject, repeated-measures; Friedman test across the three conditions, then pairwise Wilcoxon with Holm correction).
- **Engineering goal.** Users drive the RoArm-M2-S to targets with the double-blink / jaw-clench interface; report task completion time, false activations per minute and information transfer rate.

## Phase 1: offline benchmark (no human subjects needed)
0. Done: denoising benchmark, artifact detector, confound analysis on Bird et al. Muse data. Repeat the confound analysis on GAMEEMO and DREAMER (Emotiv, 14 channels) to test whether the finding generalises.
1. Get GAMEEMO (open), then DREAMER and DEAP.
2. For each dataset: `python research/benchmark.py --data <name>:<path> --out results/<name> --models logreg,mlp`.
3. Pooled model: `train.py --data gameemo:… --data dreamer:… --data deap:… --cv loso`.
4. Report: per-layout balanced accuracy (mean ± 95 % CI over held-out subjects), the augmentation effect with p-values, the region-weight scalp maps, chance levels.
5. Ablations worth a table: no region pooling (raw channels; only possible within one headset), augmentation copies 0/1/2/4, logreg vs MLP, with and without asymmetry features.

The included Muse pilot (`results/muse_mental/`) already shows the pattern on real data: with only the two temporal electrodes, accuracy falls to chance without augmentation and stays well above chance with it. Four subjects is too few to claim anything, which is why Phase 1 uses the larger datasets.

## Phase 2: human study (needs IRB/SRC approval first)
- **Participants.** 16–24 healthy volunteers, no neurological or psychiatric diagnosis, normal or corrected vision. Exclude anyone with photosensitive epilepsy (the world uses moving light).
- **Equipment.** Emotiv Insight (with an Emotiv licence for raw EEG; fallback Muse 2), laptop with Chrome and the EMOTIV Launcher, Emotif server, RoArm-M2-S for the arm task.
- **Session (about 45 min).** Consent and fitting (10 min) → Emotion induction protocol (14 min) → Closed-loop regulation protocol (20 min; conditions counterbalanced by participant code, participant blind to condition) → arm task (optional, 10 min) → debrief.
- **World.** Before the first block each participant describes (or picks) one inner world and art style. It stays the same for every condition (the app locks world generation during a session), so conditions differ only in how the decoded state drives the world. The world description is saved in the session events.
- **Measures.** EEG, SAM ratings and a 0–10 stress rating after every block, Karolinska sleepiness and hours slept at the start (no clinical questionnaires), decoded valence/arousal, condition, sham seed, arm events (all saved per session by the server).
- **Analysis.** Personal models via `train.py --data session:data/sessions` (leave-one-block-out); RQ3 on SAM change from stressor to recovery and on mean decoded arousal in the last 60 s of recovery.
- **Power.** The public Insight stress data give dz = 0.91 for the brain-only stress difference (12 people for 80 % power), but pilot effects are optimistic; plan for 20, which detects dz ≈ 0.65 at 80 % power, α = 0.05, two-sided (paired t; Wilcoxon needs about 5 % more). State this as a limitation if you recruit fewer.

## Risks and how the design handles them
- *Muscle and eye artefacts mimic emotion.* Channel-quality rejection, the arousal marker uses beta relative to alpha, and the SAM labels come from the participant, not the stimulus.
- *Decoder learns the task, not the emotion.* Within-subject CV splits by block; the closed-loop analysis compares conditions with the same task.
- *Expectation effects in the closed loop.* Sham control, counterbalanced order, participant blind to condition.
- *Different worlds for different people.* Worlds differ between participants but not between a participant's conditions; every comparison is within-subject.
- *Small sample.* Pre-register hypotheses (osf.io) before collecting data; report effect sizes and CIs, not just p-values.

## Paper outline (target: *Journal of Emerging Investigators* or *Frontiers in Human Neuroscience* student submission; arXiv preprint first)
1. Introduction: the headset-transfer problem; closed-loop affective environments.
2. Methods: pipeline (filtering, DE, region embedding with equations), augmentation, datasets, CV design, closed-loop protocol, arm interface and safety governor.
3. Results: Phase 1 tables/figures; Phase 2 RQ3; arm performance.
4. Discussion: what transfers and what doesn't, the sham comparison, limitations (sample size, EOG/EMG in the arm interface, consumer-headset signal quality).
5. Ethics and data availability: code and models released; participant data de-identified.

## Region embedding (for the methods section)
For channel unit position $p_c$ and region anchor $a_r$, with angular distance $d_{rc} = \arccos(a_r \cdot p_c)$:

$$w_{rc} = \frac{\exp(-d_{rc}^2 / 2\sigma^2)\,[d_{rc} \le d_{max}]}{\sum_{c'} \exp(-d_{rc'}^2 / 2\sigma^2)\,[d_{rc'} \le d_{max}]}, \qquad \sigma = 18^\circ,\ d_{max} = 30^\circ$$

Region feature $R_{rb} = \sum_c w_{rc} z_{cb}$ where $z_{cb}$ is the per-session z-scored differential entropy of channel $c$ in band $b$. Seven left−right region differences and a 17-element mask complete the 137-dimensional vector.
