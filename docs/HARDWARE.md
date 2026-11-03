# Supported hardware

Emotif works with any EEG device that gives it channel names and microvolts. Electrodes are placed by name on the 10-05 system and pooled into 17 scalp regions, so the same model runs on 2 electrodes or 64. Rename channels under **Setup → Electrodes** if a device reports generic names (the setting is remembered per device).

## Which headset to buy

| Pick | Price (Oct 2026) | Why | Trade-off |
|---|---|---|---|
| **Emotiv Insight** (the study headset) | $499 ([emotiv.com price guide](https://www.emotiv.com/blog/emotiv-headset-price-guide)) | 5 semi-dry sensors (AF3, AF4, T7, T8, Pz), no gel, 2-minute setup; Pz is the only consumer-headset electrode over parietal alpha; a public stress dataset was recorded with this exact headset, so Emotif's stress head is trained on Insight data | Raw EEG through Cortex needs an Emotiv licence that includes the EEG stream (EmotivPRO; Emotiv offers academic pricing on request; ask before buying). Without it, export EDF from EmotivPRO and play the file, or use LSL |
| Muse 2 (cheapest) | $249.99 ([choosemuse.com](https://choosemuse.com/products/muse-2)) | Connects straight to the browser over Bluetooth, no licence; Emotif is validated on real Muse recordings | 4 electrodes only at the forehead and behind the ears, where eye and jaw signals are strongest |
| OpenBCI Ganglion | $624.99 board + gold-cup electrodes and paste | 4 channels you can place anywhere (e.g. F3, F4, Pz, Oz); open hardware; through the server (BrainFlow) | Wet electrodes, ~10 min to apply |

Not recommended for human-subjects work: kits marketed as uncertified for use on the body; an IRB will ask.

### Emotiv Insight setup
1. Install the EMOTIV Launcher, sign in, pair the Insight (Bluetooth).
2. emotiv.com → My account → Cortex apps → create an app; copy its client ID and secret.
3. Emotif → Setup → *Emotiv Insight / EPOC X*: paste them, *Use this source*, approve Emotif in the Launcher.
4. If the status says the EEG stream was refused, your plan doesn't include raw EEG: upgrade, or record in EmotivPRO and replay the EDF (Setup → Recording file).

## Connections

| Connection | Needs | Devices |
|---|---|---|
| Web Bluetooth, in the browser | Chrome or Edge | Muse 2, Muse S, Muse 2016 |
| EMOTIV Cortex (wss://localhost:6868), in the browser | EMOTIV Launcher, a Cortex app ID, a licence with raw EEG | Emotiv Insight (5 ch, 128 Hz), EPOC X / EPOC+ (14 ch), MN8, any headset Cortex lists |
| Web Serial, in the browser | Chrome or Edge, USB dongle | OpenBCI Cyton (8 ch, 250 Hz) |
| Emotif server + BrainFlow | `python -m emotif.server` | 40 boards: OpenBCI Cyton / Daisy / Ganglion (USB, Bluetooth, WiFi), Galea; Muse 2 / S / S Athena / 2016 (native or BLED112); Neurosity Crown, Notion 1 / 2; g.tec Unicorn; ANT Neuro eego; Mentalab Explore 4 / 8 / Pro 8 / Pro 32; BrainBit; Callibri; Enophone; FreeEEG32 / 128; PiEEG; NeuroPawn Knight; Synchroni; BioListener; AAVAA |
| Emotif server + Lab Streaming Layer | an LSL outlet | Emotiv (EmotivPRO LSL), Brain Products (LSL connector), Wearable Sensing DSI, Neuroelectrics (NIC), BioSemi, g.tec g.Hiamp, OpenViBE, and anything else that publishes an EEG stream |
| File playback, in the browser | the file | EDF, EDF+, BDF (Emotiv, OpenBCI GUI, BioSemi, g.tec, Neuroelectrics, most clinical systems), CSV (`time` column + one µV column per channel) |

Requirements: at least 100 Hz sampling (for the 30–45 Hz band) and at least two electrodes with known 10-05 names. EDF/BDF files keep the voltage signals at the file's main sampling rate; accelerometer, status and annotation channels are left out automatically.

## Verified in tests
- Muse: real recordings (Bird et al. 2018) streamed through the live engine (`tests/test_replay.mjs`).
- OpenBCI Cyton: packet decoder on synthetic byte streams with garbage, split packets and a lost packet (`tests/test_app.mjs`); 1 count = 4.5 V / 24 / (2²³ − 1) = 0.02235 µV.
- EDF/BDF: files written with 16-bit and 24-bit samples, µV / mV / V units and an accelerometer channel are read identically by the app, `emotif/edf.py` and MNE-Python (`tests/test_parity.py`), and play end to end in the browser (`tests/test_browser.py`).
- Emotiv Cortex: the full JSON-RPC handshake (approval wait, authorize, headset connect, session, eeg subscription, licence refusal, close on stop) against a scripted fake Cortex service (`tests/test_app.mjs`). Not yet run against a real Insight.
- Emotiv Insight data: 17 people's real Insight recordings run through the whole pipeline (`research/states.py stress`).
- BrainFlow and LSL: server tests with BrainFlow's synthetic board and a live LSL outlet (`tests/test_server.py`).
- Region embedding: every headset layout above produces the same 137 features in Python and in the app (`tests/test_parity.py`).

Hardware not yet run with a person: OpenBCI over Web Serial and the BrainFlow boards other than the synthetic one. The first session on any new device should start with the Electrode fit panel (Setup) and a 60-second eyes-closed check: alpha (8–13 Hz) should rise at the back of the head.
