# Datasets

| Name | Loader | Headset / channels | Subjects | Labels used | Access |
|---|---|---|---|---|---|
| Bird et al. 2018, mental state | `muse_mental` | Muse, 4 ch, 256 Hz | 4 | arousal proxy: relaxed −1, neutral 0, concentrating +1 | `scripts/get_bird_muse.sh` (github.com/jordan-bird/eeg-feature-generation; cite the paper below) |
| GAMEEMO | `gameemo` | Emotiv EPOC+, 14 ch, 128 Hz | 28 | game quadrant: boring (−,−), calm (+,−), horror (−,+), funny (+,+) | Open, CC BY 4.0: data.mendeley.com/datasets/b3pn4kwpmn |
| DREAMER | `dreamer` | Emotiv EPOC, 14 ch, 128 Hz | 23 | self-reported valence / arousal 1–5, > 3 → +1 | Free with a short request form (Zenodo) |
| DEAP | `deap` | BioSemi, 32 ch (preprocessed 128 Hz) | 32 | self-reported valence / arousal 1–9, > 5 → +1 | Free after signing the EULA (eecs.qmul.ac.uk/mmv/datasets/deap) |
| SEED | `seed` | ESI NeuroScan, 62 ch, 200 Hz | 15 × 3 sessions | film valence −1 / 0 / +1 | Free after an academic application (bcmi.sjtu.edu.cn/home/seed) |
| EEGdenoiseNet | (research/train_artifact.py) | single-channel 2 s epochs, 256 Hz | 4514 EEG / 3400 EOG / 5598 EMG | clean / eye / muscle | MIT, `scripts/get_eegdenoisenet.sh` |
| Stress (Mane 2023) | `stress_insight` | **Emotiv Insight**, 5 ch (AF3 T7 Pz T8 AF4), 128 Hz | 17 paired | horror video stress +1, relaxing music −1, same person same day; `later` = recorded second (order control) | Open, CC BY 4.0, Mendeley wnshbvdxs2: `python scripts/get_mendeley.py wnshbvdxs2 --out data/external/stress_insight` |
| Depression screening (Sakib et al.) | `depression_phq9` | Emotiv EPOC+, 14 ch, 128 Hz, 5 min rest | 31 | PHQ-9 ≥ 20 → +1 (18), ≤ 4 → −1 (13) | Open, CC BY 4.0, Mendeley ct5vrc4k2t |
| Anxiety screening (Sakib et al.) | `anxiety_gad7` | Emotiv EPOC+, 14 ch, 128 Hz, 5 min rest | 38 | GAD-7 ≥ 15 → +1 (23), ≤ 4 → −1 (15) | Open, CC BY 4.0, Mendeley rh3fy75zdv |
| Resting EEG and sleepiness (Martynova & Portnova) | `sleepiness` | 19-ch 10-20, 250 Hz, 4–11 min rest | 49 | visual-analogue sleepiness ≥ 5 → +1 (27), else −1 (22); ESS, SSS, VASS kept | Open, CC BY 4.0, Mendeley wnp3jnymvn |
| Your sessions | `session` | any | — | each participant's SAM ratings (or `labels="design"`) | `data/sessions/` |

Notes on the state datasets (checked on the files, October 2026):
- Stress: only the horror and relaxing-music folders were recorded with the same people on the same days; the arithmetic, Stroop and Trier folders were recorded months later with other people, so they can't give a within-person contrast and are not used. Recordings were paired from the EDF header (names are not stored in the code); one recording was uploaded twice (horror `002 (5)` = music `001 (3)`) and is left out; two pairs are matched by date, time and initial only. Horror came second for 12 of 17 people, hence the order control.
- Depression and anxiety: columns are in Emotiv's export order (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4); confirmed by relative alpha peaking in columns 7–8 (O1, O2). Raw values carry the headset's ~4200 µV offset, removed by the 0.5 Hz filter. No recording appears in both sets.
- Sleepiness: every volunteer scored ESS ≤ 10, so the label is the state rating at the recording, not daytime sleepiness. This set is not insomnia; no public waking-EEG insomnia set with raw data was reachable (the PSG insomnia set, Mendeley 3hx58k232n, is overnight sleep).
- Not used: ECSMP (Mendeley vn5nknh3mn; frontal-only 7-channel cap, event table stored as a MATLAB table object), anger-in-VR (Mendeley y76vbw92y9; spreadsheet only, no raw EEG). No reachable dataset had raw EEG with anger as a label.

`scripts/get_mendeley.py ID --out FOLDER` downloads any public Mendeley dataset with SHA-256 checks (`--list` shows the files first).

Point the loader at the folder you unpacked; it searches sub-folders. GAMEEMO: use the folder that contains `(S01)`, `(S02)`… DREAMER: the path to `DREAMER.mat`. DEAP: `data_preprocessed_python/`. SEED: the folder containing `Preprocessed_EEG/`.

Order to get them: GAMEEMO today (no form, same headset family as DREAMER), DREAMER and DEAP next (forms take a few days), SEED last (needs a supervisor's signature).

## Citations
- Mane, M. (2023). An EEG recordings dataset for mental stress detection (v1). Mendeley Data. https://doi.org/10.17632/wnshbvdxs2.1
- Sakib, N., et al. (2025). Multi-channel wireless EEG recordings of young adults for depression screening based on PHQ-9 (v1). Mendeley Data. https://doi.org/10.17632/ct5vrc4k2t.1 ; analysis paper: Sakib, N., Islam, M. K., & Faruk, T. (2023). Machine learning model for computer-aided depression screening among young adults using wireless EEG headset. *Computational Intelligence and Neuroscience*, 2023, 1701429.
- Sakib, N., et al. (2025). Multi-channel wireless EEG recordings of young adults for anxiety screening based on GAD-7 (v2). Mendeley Data. https://doi.org/10.17632/rh3fy75zdv.2
- Martynova, O., & Portnova, G. (2020). Resting-state EEG and ARSQ (v1). Mendeley Data. https://doi.org/10.17632/wnp3jnymvn.1
- Zhang, H., Zhao, M., Wei, C., Mantini, D., Li, Z., & Liu, Q. (2021). EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG denoising. *Journal of Neural Engineering*, 18(5), 056057.
- Mullen, T. R., et al. (2015). Real-time neuroimaging and cognitive monitoring using wearable dry EEG. *IEEE Trans. Biomedical Engineering*, 62(11), 2553–2567.
- Kothe, C. A. E., & Jung, T.-P. (2016). Artifact removal techniques with signal reconstruction. U.S. patent application (the ASR method; cite the published patent number you look up).
- Bird, J. J., Manso, L. J., Ribeiro, E. P., Ekárt, A., & Faria, D. R. (2018). A study on mental state classification using EEG-based brain-machine interface. *International Conference on Intelligent Systems*.
- Alakus, T. B., Gonen, M., & Turkoglu, I. (2020). Database for an emotion recognition system based on EEG signals and various computer games – GAMEEMO. *Biomedical Signal Processing and Control*, 60, 101951.
- Katsigiannis, S., & Ramzan, N. (2018). DREAMER: A database for emotion recognition through EEG and ECG signals from wireless low-cost off-the-shelf devices. *IEEE J. Biomedical and Health Informatics*, 22(1), 98–107.
- Koelstra, S., et al. (2012). DEAP: A database for emotion analysis using physiological signals. *IEEE Trans. Affective Computing*, 3(1), 18–31.
- Zheng, W.-L., & Lu, B.-L. (2015). Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks. *IEEE Trans. Autonomous Mental Development*, 7(3), 162–175.
- Duan, R.-N., Zhu, J.-Y., & Lu, B.-L. (2013). Differential entropy feature for EEG-based emotion classification. *IEEE/EMBS NER*.
