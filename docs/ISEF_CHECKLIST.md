# OCSEF / ISEF checklist

**Category:** Biomedical Engineering. OCSEF (Orange County Science & Engineering Fair) is an ISEF-affiliated fair and follows the ISEF rules and forms; check the OCSEF website for this year's registration and SRC deadlines, which come weeks before the fair.

Check the current year's *International Rules for Pre-College Science Research* (societyforscience.org/isef/international-rules) and your affiliated fair's deadlines; the forms below are the ones these rules normally require for this kind of project.

## AI rule (new for 2027)
- [ ] AI may be used as a cited resource; you may **not** use it to write the research plan, abstract or poster, or to create citations. Write those yourself and fill in `docs/AI_USE.md`.
- [ ] Be able to explain every pipeline stage and number on your board without notes.

## Before collecting any human data
- [ ] **Research Plan / Project Summary** written (use `docs/RESEARCH_PLAN.md`), with the risk assessment and how you'll handle data.
- [ ] **Human Participants** review: EEG recording plus mood induction is human-subjects research. An **IRB** (at least: a science teacher, a school administrator, and a medical or mental-health professional) must approve the protocol **before** the first participant. Expect the IRB to ask about the negative-memory and stress tasks; the protocols keep them mild and short, and participants can stop at any time.
- [ ] **Forms (2027 numbering)**: Checklist for Adult Sponsor (1), Student Checklist (1A), Research Plan / Project Summary, Approval Form (1B), Student Support Disclosure (2A, new, all projects), Qualified Scientist (2B) if one supervises, Regulated Research Institution (2C) if done at a lab, Risk Assessment (3) for the robot arm and if the IRB rates the study more than minimal risk, Human Participants (4) with three IRB signatures dated before the first participant, and blank consent / assent / parental-permission forms plus the SAM sheet.
- [ ] **Consent**: written informed consent from adults; for minors, parent/guardian permission **and** the participant's assent.
- [ ] **Data plan**: participant codes only (the app enforces letters/numbers), consent forms stored separately from data, data kept on one computer, deleted or de-identified after the fair.
- [ ] **Exclusions**: photosensitive epilepsy, recent head injury, current psychiatric treatment (ask the IRB's medical member to confirm the list).
- [ ] **Personal calibration tasks** (anxiety imagery, anger recall, low-mood recall, drowsiness) are in the consent/assent forms; each is 1 minute, mild, followed by a break; the session ends with breathing. Daily check-ins are optional and ask only mood 0–10, hours slept, sleep quality 0–10.
- [ ] **No clinical questionnaires with minors** unless the IRB asks: depression, anxiety and sleep are analysed in public adult datasets only (the public-data results show a 5-electrode headset can't screen for them). If added: PHQ-8 (not PHQ-9), GAD-7, a counselor as referral contact, and a Qualified Scientist (Form 2B).
- [ ] **Pre-register** the hypotheses on OSF (strong signal of rigour to judges).

Phase 1 (public datasets only) needs no IRB, because the data are de-identified and public; you can start that today.

## Robot arm safety
- [ ] Adult supervision whenever the arm can move; participants seated outside its reach (radius ≈ 0.5 m).
- [ ] Speed limit ≤ 120 mm/s, workspace limits set for your table, pen-only end effector for drawing.
- [ ] Demonstrate the emergency stop (button, Esc key, and the 1-second watchdog) in the video and at the booth.

## Engineering-category display
- [ ] Problem, design criteria, and the iterations with their numbers (`docs/ENGINEERING_NOTEBOOK.md`): this is what BMED judges score.
- [ ] Core figures: brain-only vs standard decoder (`results/brain_gameemo_v2/fig_brain_decoder.png`), what the decoder reads (`fig_brain_map.png`), stress / trait screens / split inflation on the Insight layout (`results/states/fig_states.png`), the leakage audit (`results/audit/fig_audit.png`), cleaning vs ground truth (`results/denoise`).
- [ ] A dated lab notebook (paper or digital) with entries for each test.

## At the fair
- [ ] **Booth demo** button runs a narrated simulator tour with no headset, so a dead battery doesn't end the demo; bring the headset for live demos. The **Method** view shows the pipeline with live numbers and the result figures.
- [ ] Display rules usually ban human-subject photos without consent and live hazards; check what's allowed for a moving arm at your fair.
- [ ] Have ready: the benchmark figure (accuracy by headset layout, with and without augmentation), the closed-loop result with the sham control, and the arm task numbers.
- [ ] Be ready to explain: why region pooling works, why CV splits by recording, why a sham control, what the blink/clench interface actually measures (eye and muscle signals, not "thoughts"), and the limits.

## Judges will probe
- Data leakage (answer: trial-wise splits, LOSO, chance reported; fig_states panel C shows the same data scoring 0.82–0.92 with random window splits and ~0.5 split by person).
- "Can it detect depression?" (answer: tested on public PHQ-9 / GAD-7 data with person-level evaluation and permutation tests: no, not from the brain alone; standard decoders' apparent anxiety signal is about half face. This is a finding, and it's why the app shows no clinical readout.)
- Whether "emotion" is really decoded (answer: SAM labels, the sham comparison, effect sizes, and honest accuracy).
- What's original (answer: a brain-only decoder that measures facial activity in every window and removes what it explains, verified with an independent face measure and shown live next to a standard decoder; the artifact-leakage audit with known-answer validation; stability-corrected ASR and a state-aware artifact detector checked against ground truth; one model for any headset; a sham-controlled, photosafe closed-loop world).
