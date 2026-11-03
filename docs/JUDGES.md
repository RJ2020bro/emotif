# Judge questions: the facts behind each answer

Practise answering out loud in your own words. Numbers are in `docs/RESULTS.md`.

**What problem does this solve?**
Consumer EEG headsets put a few dry electrodes on the forehead and temples, where eye and jaw-muscle signals are far larger than brain signals and change with emotion. A decoder can score well while reading the face. Emotif decodes emotion from the brain part only, on any headset, and measures how well that worked.

**What do you mean by "the face"?**
Electrical activity from the face, not a camera. Blinks and eye movements make slow voltage swings of 100+ µV at the forehead; clenching or tensing the jaw and forehead makes fast muscle activity above 20 Hz. Brain signals at the scalp are about 10–50 µV. Both change when people feel things, so a decoder can learn the face instead of the brain.

**What's the innovation?**
A brain-only decoder. In every 2-second window it measures blink size and jaw-muscle power at each electrode, and subtracts from the brain features everything those measurements predict (learned on the training people). The subtraction folds into the decoder's weights, so it runs live at no extra cost, on any headset. Then I check it with a face measure it never saw: the share of its output the face explains drops from 29–36 % (standard decoder) to 6–9 %.

**What's your main finding?**
On 28 new people (GAMEEMO), the brain-only decoder reads valence at 0.569 per window and 0.655 over 30 s (chance 0.5); the standard decoder, which also uses the face, gets 0.670. So valence is decoded from the brain. Arousal keeps a face component: 0.642 brain-only vs 0.693 standard. The brain-only decoder rediscovered a known pattern: pleasant states go with less left-frontal alpha, high arousal with less right-frontal alpha, both stable in every resampled refit.

**And the Muse pilot?**
On the Bird et al. Muse dataset, 3 face numbers per electrode (blink size, 20–30 and 30–45 Hz power) decode relaxed / neutral / concentrating at 0.693, the same as the EEG decoder (0.689; chance 0.333). With the face partialled out, cleaned EEG is 0.414 (p = 0.056): not significant. That's why the brain-only decoder exists.

**How do you know it removes the face and keeps the brain?**
Three checks. (1) Known answer: synthetic data where the label is in the brain features, or only in the face. Brain in: the brain-only decoder keeps it (0.86 vs 0.82 standard). Face only: the standard decoder reads it (0.96), the brain-only one falls to chance (0.51) and its face-driven output variance drops about 300×. (2) Real data: on 28 new people the face-explained share drops from 29–36 % to 6–9 %, measured with face indices the decoder never used. (3) Permutation tests, shuffling labels between whole recordings: valence p = 0.0099 (100 shuffles), arousal p = 0.0196 (50 shuffles); both the smallest possible.

**What if someone pulls a big face?**
If eye or jaw activity goes beyond anything in the training data, the correction would be a guess, so the brain-only decoder holds its last value and the status shows "Face activity · holding". That happens in 0.7 % of a new person's windows on GAMEEMO. The *Brain vs face* panel shows both decoders and the eye/jaw activity live.

**So EEG doesn't work?**
Not what it shows. Cleaned EEG still decodes above chance (0.592, p = 0.001). Partialling also removes brain activity that moves with the face, so 0.414 is a lower bound. What it shows: on this dataset you can't claim neural decoding across people without a face control. 4 people is a pilot. On GAMEEMO (28 people, 14 channels) the face still explains most of the gain, but face-partialled EEG stays above chance for valence (0.569) and arousal (0.557), so with more electrodes and people part of the decoding is not the face.

**How do you know the audit itself works?**
Known-answer simulations. Label put in the brain (alpha): face indices at chance in all 7 runs, face-partialled EEG 0.97–0.99. Label put in the face (jaw muscle + blinks): face indices 0.94–0.96, partialled EEG 0.49–0.58. One of six face runs left a small residual, which I report.

**Why not compare to 1/3 chance?**
With 24 recordings, a model can beat 1/3 by luck. Shuffling labels between whole recordings and rerunning everything 1,000 times gives the real chance band: 95th percentile 0.42–0.47.

**How do you know the cleaning works?**
Ground truth: real eye and muscle artifacts added to clean EEG, so the right answer is known. ASR cuts error 13–29 % on four headset layouts, keeps all data, and leaves clean windows and a real alpha doubling unchanged. Threshold chosen on one set of simulations, reported on another.

**What did you change in ASR?**
It blew up (13× worse) because the baseline covariance wasn't valid; geometric median + ridge + energy guard fixed it. It erased real emotional changes; a 2.5× floor fixed it.

**Is the detector really AI, and does it work on real hardware?**
A 64-32 neural network on 11 features per channel, 93.4 % on held-out data. Version 1 called most real Muse channels "muscle"; judging each channel against its own baseline across 600 simulated headsets fixed it. Version 2 also flagged real alpha changes as artifacts (5.4 %); training with genuine state changes brought that to 2.6 %.

**How does it work on different headsets?**
Every electrode is placed on the 10-05 system and pooled into 17 scalp regions, so every model sees the same 137 features. An optional region graph fills regions the headset doesn't cover from the nearest covered ones. Browser: Muse (Bluetooth), OpenBCI Cyton (USB). Server: 40 BrainFlow boards, any LSL stream. Files: EDF, BDF, CSV.

**Why logistic regression and not a deep network?**
4 people. I tested an MLP and a region-graph model: no better (0.564, 0.553 vs 0.564), and the graph survey I read warns that learned graph edges overfit small data. The graph's job is transferring dense-cap models to sparse headsets, which needs the bigger datasets.

**Within-subject accuracy is near chance but across-people isn't. Why?**
What generalises across people here is a shared facial pattern; between two recordings of the same person, electrode contact changes more than the brain state. The audit confirms it.

**What does "calibrated" mean?**
When the model says 80 % sure it should be right about 80 % of the time. Temperature scaling took the error from 0.170 to 0.078 on held-out people. The app moves the world less when it's unsure.

**What's the sham condition for?**
Without it you can't tell whether the feedback helps or people just relax over time. The sham world follows a random path that ignores the brain; order is counterbalanced and hidden; everyone is debriefed.

**What does the world do?**
It shows the decoded state: light, colour, haze, clouds, wind, weather and how fast the explorer travels follow valence and arousal. It also grows from the brain: every piece of land is generated when the explorer first approaches it, and what grows there (flowers and lush grass, or bare rock and wilted foliage) is set by the state at that moment and kept, so the landscape becomes a map of the session. The ground itself never depends on the state, so neighbouring pieces always meet. The person can pick an art style and a short theme; that runs on the laptop, no AI service. For showcases the same state goes to Unreal Engine 5 over OSC; the study uses the tested browser world.

**Can it detect stress, anxiety, depression, anger or insomnia?**
Stress: within one person, partly. On 17 people recorded with the same Emotiv Insight I use, the brain-only decoder rated the horror video above the relaxing music in 13 of 17 new people, including all 5 who heard the music second, while it could not tell which recording came first (so it isn't tracking time). Per 2-second window it is near chance, so it's a slow, personal measure. Depression, anxiety, sleepiness: no. With person-level testing and permutation tests on public PHQ-9, GAD-7 and sleepiness data, brain-only screening is at chance on the Insight layout and on full 14/19-electrode caps; standard decoders look better for anxiety and sleepiness, but about half of what they output is explained by the face. Random window splits, common in published work on these datasets, push the same data to 82–92 %. Anger and waking insomnia: no reachable dataset had raw EEG with those labels, so I make no claim. That's why the app has no clinical readout.

**So how does it detect all of those states?**
Personally, not universally. Public data showed the brain differences between states are consistent within a person but not across people. So each person does a 28-minute calibration: three rounds of six one-minute tasks (breathing for calm, timed arithmetic for stress, imagining a test for anxiety, recalling something unfair for anger, recalling a disappointment for low mood, trying to feel sleepy for drowsiness). Emotif removes whatever the eye and jaw activity explains, learns that person's signature of each state, and tests each one on a round it never saw against 199 label shuffles. Only states that pass are shown live, and a face-only reference shows how much the face alone could have done. Depression and insomnia aren't moments, so they're handled as change over days: a 3-minute daily check-in compares resting brain markers with the person's own mood and sleep answers. It tracks change in one person and never diagnoses.

**Can the wearer actually move things with their brain?**
Yes, with alpha rhythm, which is a brain signal: focusing lowers posterior alpha (Pz on the Insight) below your own resting level, relaxing raises it. Hold focus for a second and the nearest crystal lifts and follows you; relax and it's set down and blooms. It's slow (one command every few seconds), which is honest for a 5-electrode headset, and it pauses whenever the face is active, so blinks and clenches can't fake it. Each lift and placement is logged with a timestamp, so its reliability can be measured.

**Why the Emotiv Insight?**
Five semi-dry sensors with no gel, a two-minute setup, and Pz: the one consumer-headset electrode over parietal alpha. A public stress dataset was recorded with this exact headset, so I could test on the real device before buying it.

**Is the robot arm controlled by thoughts?**
It is a separate mode. Partly, and I say so. Embody and Draw modes follow the decoded state. Deliberate control uses double blinks and jaw clenches: eye and muscle signals, used on purpose because they're reliable. A hybrid EEG/EOG/EMG interface.

**Safety and accessibility?**
Arm: workspace clamp, 120 mm/s limit, 1 s watchdog, Esc and STOP. World: can't flash more than once per second even if the decoder is adversarial (limit 3). App: WCAG 2.1 AA (axe-core, 0 serious issues), screen-reader announcements, keyboard, sound display, reduced motion. Data stays on the laptop under participant codes.

**Did you use AI?**
Yes, cited as a resource (see `docs/AI_USE.md`). Say exactly what it did and what you designed, ran, checked and decided. Be ready to explain any file.

**Which headset should someone use?**
Emotiv Insight ($499) for this study (needs an Emotiv licence for raw EEG). A Muse 2 ($250) works straight in the browser with no licence; its electrodes are where the face is loudest, which is what the brain-only decoder is for.

**What would you do next?**
The IRB-approved Insight study: within-person stress vs relaxation, pre-registered, with the closed-loop world against a sham; the brain-only decoder on DREAMER and DEAP (self-reported labels); a paper on the audit and on how window-level splits inflate EEG screening results.
