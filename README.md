# Emotif

Emotif is research software that decodes emotion from brain activity alone, on any EEG headset, and lets the decoded state shape a 3-D world.

Consumer headsets sit on the forehead and behind the ears, where blinks and jaw-muscle activity are many times larger than brain signals and change with emotion too. A decoder can score well by reading the face. Emotif measures that facial activity in every 2-second window and removes everything it explains before decoding, so what remains comes from the brain. It also measures how well that worked.

```
headset ─┬─ Emotiv Insight / EPOC (Cortex) · Muse (Web Bluetooth) · OpenBCI Cyton (Web Serial), in the browser
         ├─ 40 BrainFlow boards / any LSL stream ──▶ emotif.server ───┐
         └─ simulator / EDF · BDF · CSV replay                         │ WebSocket
                                                                       ▼
 app (browser)
  filter 0.5 Hz HP + 50/60 Hz notch ─┬─▶ face indices per channel: blink size (0-3 Hz), jaw power (20-30, 30-45 Hz)
                                     └─▶ ASR artifact repair ─▶ AI artifact detector ─▶ differential entropy, 5 bands
  ─▶ 17-region scalp embedding (any headset) ─▶ brain-only decoder: remove what the face explains, then decode
  ─▶ calibrated valence + arousal (+ stress) ─▶ Kalman smoothing ─▶ 3-D world (mirror / guide / sham)  ·  OSC ─▶ Unreal Engine 5
                                                                                                 ·  Arm mode: RoArm-M2-S
```

## Results at a glance (details in `docs/RESULTS.md`)
| | |
|---|---|
| **Brain-only decoder**, GAMEEMO (28 new people, 14-ch Emotiv, chance 0.5) | valence 0.569, arousal 0.564 per 2 s window (permutation p = 0.010, 0.020); 0.655 and 0.642 over 30 s |
| Share of the decoder's output the face explains (independent face measure) | standard decoder 29–36 % (52 % without cleaning) → brain-only 6–9 % |
| Valence without the face vs with it (30 s) | 0.655 vs 0.670: valence is decoded from the brain, not the face |
| What the brain-only decoder reads | pleasant: less left-frontal alpha (F7); high arousal: less right-frontal alpha (F4); both stable in 100 % of 200 resampled refits |
| Same decoder on consumer layouts | Muse positions 0.534 / 0.536 (valence / arousal) |
| **Stress on a real Emotiv Insight** (17 new people, horror video vs relaxing music) | brain-only output higher for horror in 13/17 (Wilcoxon p = 0.002), all 5 who heard music second; does not decode recording order (the standard decoder does) |
| Depression (PHQ-9), anxiety (GAD-7), sleepiness from 5 min of rest, Insight layout | brain-only at chance (AUC 0.46–0.58, all p ≥ 0.21); standard decoders' best result (sleepiness, p = 0.042) is half face; random window splits inflate the same data to 0.82–0.84 |
| Artifact-leakage audit, real Muse data (4 people) | 3 face numbers per electrode decode the state as well as the EEG (0.693 vs 0.689, chance 0.333): small-headset pilots need a face control |
| Artifact repair, ground truth, 4 headset layouts | error −13 to −29 %, 100 % of data kept, clean data and real alpha changes unchanged |
| AI artifact detector, held-out | 93.4 % accuracy (clean / eye / muscle), 0.8 % false alarms at rest |
| Live engine | 0.6–0.8 ms per decoded window in a browser |
| Python ↔ app agreement | ≤ 1e-8 for every algorithm, including the face indices and the brain-only decoder |

## What's new
1. **Brain-only decoding.** For every window the app measures blink size and jaw-muscle power per electrode, pools them into the same 17 scalp regions as the EEG, and subtracts from the EEG features everything those numbers (and their squares) predict; the coefficients are learned on the training people. The subtraction folds into the decoder's weights, so it costs nothing at run time. Its success is checked with a face measure it never saw (the audit's): the face-explained share of the output falls from 29–36 % to 6–9 %.
2. **An artifact-leakage audit** (`python -m emotif.audit`). Six decoders on the same windows and split, specific face indices, face-partialled EEG and recording-level permutation tests: any consumer-EEG study can report how much of its result the face explains. Validated with known-answer simulations.
3. **Brain vs face, live.** The Live view runs the standard decoder next to the brain-only one, with eye and jaw activity drawn underneath. When facial activity goes beyond anything in the training data (0.7 % of a new person's windows), the brain-only decoder holds its last value instead of guessing.
4. **Cleaning built for consumer headsets.** Artifact Subspace Reconstruction (ASR), redesigned to stay stable on rank-deficient data and to never erase genuine state changes, plus a neural-network artifact detector that judges each channel against its own resting baseline.
5. **Montage-agnostic.** Any electrode layout is pooled into 17 fixed scalp regions, so one model runs on Muse, Emotiv, OpenBCI, Neurosity, Unicorn, BrainBit or a lab cap. Confidence is calibrated.
6. **A sham-controlled closed loop.** The world can mirror the decoded state, guide it toward calm, or follow a random path that ignores the brain. Conditions are counterbalanced and hidden from the participant.
7. **An inner world driven by the brain.** The decoded state sets the light, colour, mist, wind, weather and the explorer's pace in an endless world; every second leaves a beacon on its path (time as the fourth dimension). An optional short theme and an art style (voxel, low-poly, dreamscape, neon) set only the scenery, on your computer, with no AI service.
8. **Accessible.** Screen-reader announcements, keyboard navigation, a sound display of the state, reduced motion, high contrast, and a world that cannot flash.
9. **Personal state calibration: calm, stress, anxiety, anger, low mood, drowsiness.** A 28-minute session (Study → *Personal state calibration*) induces each state three times; Emotif learns that person's brain signature of each with facial activity removed, tests every state on rounds it never saw against 199 label shuffles, and shows a state live only if it passes. A face-only reference reports how much the face alone could have done. Depression- and insomnia-related change is tracked over days with 3-minute *Daily check-ins* (resting markers vs the person's own mood and sleep answers, within-person, Holm-corrected), never as a diagnosis. Why personal: on public data the same features separate states within a person (stress 13/17) but not traits between people.
10. **More states, tested honestly.** Stress, depression, anxiety and sleepiness on the Emotiv Insight layout with person-level evaluation, permutation tests, an order control and the face audit (`research/states.py`). Stress works slowly within a person; the trait screens don't work from the brain, and the app makes no clinical claim.
11. **Emotiv Insight, Unreal Engine 5.** A Cortex source in the browser; the decoded state (and the experimental stress head) goes out over OSC to an Unreal Engine 5 scene (`ue5/`).
12. **Arm mode.** A separate mode for the RoArm-M2-S: embody the state, draw it, or drive it deliberately with double blinks and jaw clenches (eye and muscle signals, used on purpose). Workspace limits, a speed limit, a watchdog and Esc stop apply to every command.

## Which headset
**Emotiv Insight** ($499): 5 semi-dry sensors (AF3, AF4, T7, T8, Pz), no gel, and Pz covers parietal alpha, which forehead bands miss. Emotif's stress head is trained on real Insight recordings. It connects through the EMOTIV Launcher (Setup → *Emotiv Insight / EPOC X*); raw EEG needs an Emotiv licence that includes the EEG stream, so check that before buying. Cheaper, no licence: **Muse 2** ($249.99, Web Bluetooth). Details in `docs/HARDWARE.md`.

## Run it
```bash
pip install -r requirements.txt
python -m emotif.server            # open http://localhost:8000
```
- **No headset:** the simulator starts on its own. **Booth demo** (top bar) runs a narrated tour for an exhibit.
- **Emotiv Insight / EPOC X:** EMOTIV Launcher running and signed in → Setup → *Emotiv Insight / EPOC X* → paste your Cortex app's client ID and secret → approve Emotif in the Launcher.
- **Muse 2 / S:** Setup → *Muse over Bluetooth* (Chrome or Edge), or via the server with BrainFlow.
- **OpenBCI Cyton:** Setup → *OpenBCI Cyton over USB* (Chrome or Edge), or via the server.
- **40 BrainFlow boards** (OpenBCI, Muse, Neurosity, Unicorn, ANT Neuro, Mentalab, BrainBit, FreeEEG…): Setup → *Headset through the Emotif server*.
- **Emotiv and others:** publish an LSL stream from the vendor app, then Setup → *Lab Streaming Layer*; or export EDF and play the file.
- Full list: `docs/HARDWARE.md`.
- **Fit:** Setup → *Electrode fit* tells you, per electrode, what to fix (no contact, mains hum, muscle, drift).
- **Headset check:** Study → *Headset check* (4 min, with an eyes-closed minute) or *Blink and jaw check* (1 min), then `python research/session_qa.py data/sessions/<folder>`: alpha eyes closed/open ≥ 1.5 at Pz/T7/T8 (when run), blink index ≥ 2 at AF3/AF4, jaw power ≥ 3 at T7/T8, every electrode ≤ 20 % unusable windows.
- **Every session:** sit still for the 30 s resting baseline. It calibrates the cleaning and the decoder to you.
- **RoArm-M2-S:** Robot arm → USB or Wi-Fi (192.168.4.1 on the arm's own network). Tick the safety box, then Enable motion.
- **Your states:** Study → participant code → *Personal state calibration*. When it ends, the Study view shows each state's held-out AUC, p and whether it's shown; the Live view then lists the passing states and the world's weather follows the dominant one (calm: fireflies, stress: rain, anxiety: haze, anger: storm, low mood: snow and cloud, drowsiness: dimmer light). *Daily check-in* builds the over-days record.
- **Brain-only decoder:** selected by default (Models). The *Brain vs face* panel (Live) shows it next to the standard decoder, with eye and jaw activity underneath.
- **Move things with your brain:** in the world, a ring marks the nearest glowing crystal. Focus (posterior alpha drops below your resting level for a second) lifts it and it floats beside you; relax (alpha rises) sets it down and a ring of flowers grows around it. The control holds whenever the face is active, so blinking or clenching can't do it. The bar at the bottom shows your focus/relax level.
- **World:** Live → *World* panel: art style (default *Natural*: photographed ground and sky from an open image library, leaf-card trees, palms, water, wind-blown grass) and an optional theme (e.g. "forest with a lake"). The decoded state sets light, haze, clouds, wind and weather, and decides what grows on each piece of land as it appears, so the landscape is a map of the session. Click the world, then arrow keys or W A S D steer.
- **Unreal Engine 5:** `python -m emotif.server --osc 127.0.0.1:8001` sends the decoded state as OSC; `ue5/README.md` wires it into a UE5 scene.
- **Method** (sidebar) shows every pipeline stage with live numbers and the result figures, for judges.

Without the server the app still runs the simulator, CSV replay, models and the arm twin. Recording and the physical arm need the server.

## Models, data and research
```bash
./scripts/get_bird_muse.sh            # real Muse recordings (Bird et al. 2018)
./scripts/get_eegdenoisenet.sh        # EEGdenoiseNet clean/EOG/EMG epochs (MIT)

python research/train_artifact.py     # AI artifact detector -> models/artifact_detector.json
python research/denoise_benchmark.py  # ground-truth cleaning benchmark -> results/denoise
python -m emotif.audit --data muse_mental:data/external/bird_muse --perms 1000      # brain or face? -> results/audit
python research/brain_decoder.py --data gameemo_raw:data/external/gameemo --hop 2 --out results/brain_gameemo_v2   # brain-only decoder vs standard
python research/brain_map.py --cache results/brain_gameemo_v2/cache.npz --out results/brain_gameemo_v2           # what it reads
python scripts/get_mendeley.py wnshbvdxs2 --out data/external/stress_insight   # + ct5vrc4k2t, rh3fy75zdv, wnp3jnymvn (docs/DATASETS.md)
python research/states.py stress --perms 200 --out results/states     # stress within a person, real Insight data
python research/states.py transfer --out results/states                # GAMEEMO decoder on the Insight stress data
python research/states.py traits --perms 1000 --out results/states     # depression, anxiety, sleepiness (add --full for all electrodes)
python research/states.py figure --out results/states
python research/personal_states.py --sessions data/sessions --out results/personal_states   # your calibration sessions: per-person and group tests
python research/confound_analysis.py --data muse_mental:data/external/bird_muse   # cleaning pipelines compared
python research/benchmark.py --data muse_mental:data/external/bird_muse --out results/muse_mental --models logreg,mlp,graph

python train.py --data muse_mental:data/external/bird_muse --out models/muse_mental_logreg.json
python train.py --data gameemo_raw:data/external/gameemo --hop 2 --model brain --out models/gameemo_brain.json   # brain-only
python train.py --data gameemo_raw:data/external/gameemo --data stress_insight:data/external/stress_insight --hop 2 --model brain --targets valence,arousal,stress --cv none --out models/insight_brain.json
python train.py --data gameemo:./GAMEEMO --data dreamer:./DREAMER.mat --model mlp --out models/emotiv.json
python train.py --data session:data/sessions --out models/my_study.json     # your own rated sessions
```
`train.py` cleans exactly as the app does (`--clean full|asr|detector|none`, `--asr-k 20`). Cross-validation always splits by recording, never by window.

## Tests
```bash
./run_tests.sh      # 7 suites: parity (incl. face indices + brain-only decoder), cleaning, training, server + arm, app modules + world, real-data replay, full browser run (incl. a simulated jaw clench)
```

## Layout
```
emotif/      features · face · asr · artifact · audit · erasure · montage · models · datasets · edf · server · osc · devices · recorder · roarm
app/src/     TypeScript app (engine, asr, artifact, dsp, montage, model, edf, sources, cortex, safety, sonify, intent, director, protocol, arm, face, worldspec, terrain, world, charts, main)
web/         built single-file app (served by the server; also works as a static page)
research/    brain_decoder · brain_map · states · train_artifact · denoise_benchmark · confound_analysis · benchmark
docs/        RESULTS · ENGINEERING_NOTEBOOK · LITERATURE · HARDWARE · RESEARCH_PLAN · JUDGES · ISEF_CHECKLIST · DATASETS · AI_USE
ue5/         Unreal Engine 5 showcase recipe (OSC)
models/      gameemo_brain.json (default) · insight_brain.json (+ stress) · gameemo_logreg.json · gameemo_graph.json · muse_mental_logreg.json · artifact_detector.json
```
Rebuild the app after editing TypeScript: `cd app && npm install && npm run build`.

## Limits
- EEG emotion estimates are noisy and person-specific; treat them as research measurements, never as a diagnosis.
- The bundled models are trained on GAMEEMO, whose labels are each game's intended emotion, not self-report; accuracy on a new person is modest (0.56–0.66, chance 0.5). Train on your own rated sessions (Study → Models → Fit to my ratings) for more.
- The brain-only decoder removes what the measured face activity predicts *linearly and quadratically*; it cuts the face's share (to 6–9 %), it does not prove zero. In the simulator, a jaw clench is mostly removed by the cleaning before it reaches either decoder, so the live panel isn't a test of the decoder; the evidence is the GAMEEMO comparison and a known-answer test. Brain activity that moves together with facial activity is removed too, so accuracy is a conservative estimate.
- Blink and jaw control uses eye and muscle signals on purpose; describe it as a hybrid EEG/EOG/EMG interface.
- Web Bluetooth and Web Serial aren't available in Safari or on iOS; use the server there.

## Acknowledgements
Built with substantial help from an AI assistant (Claude, Anthropic); see `docs/AI_USE.md`. Data: GAMEEMO (Alakus et al. 2020), Bird et al. (2018) and EEGdenoiseNet (Zhang et al. 2021).
