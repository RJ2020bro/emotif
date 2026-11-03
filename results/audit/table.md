| Decoder (LOSO) | Balanced accuracy | 95 % CI | Permutation p |
|---|---|---|---|
| EEG, filter only | 0.689 | 0.581–0.787 | 0.0010 |
| EEG, cleaned (ASR + detector) | 0.592 | 0.520–0.670 | 0.0010 |
| Face only: blink + muscle indices | 0.693 | 0.588–0.794 | 0.0010 |
| Face only: detector p(eye), p(muscle) | 0.694 | 0.598–0.788 | 0.0010 |
| Cleaned EEG minus what the face explains | 0.414 | 0.344–0.480 | 0.0559 |
| Face indices + cleaned EEG | 0.686 | 0.583–0.783 | 0.0010 |

Chance 0.333; 1078 windows, 24 recordings, 4 subjects; 1000 recording-level permutations.
Leakage ratio (face-index gain / filtered-EEG gain over chance): 1.01.
