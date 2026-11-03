| Decoder (LOSO) | Balanced accuracy | 95 % CI | Permutation p |
|---|---|---|---|
| EEG, filter only | 0.591 | 0.573–0.608 | 0.0050 |
| EEG, cleaned (ASR + detector) | 0.582 | 0.563–0.600 | 0.0050 |
| Face only: blink + muscle indices | 0.575 | 0.549–0.600 | 0.0050 |
| Face only: detector p(eye), p(muscle) | 0.563 | 0.530–0.597 | 0.0149 |
| Cleaned EEG minus what the face explains | 0.569 | 0.554–0.585 | 0.0050 |
| Face indices + cleaned EEG | 0.605 | 0.585–0.622 | 0.0050 |

Chance 0.500; 16456 windows, 112 recordings, 28 subjects; 200 recording-level permutations.
Leakage ratio (face-index gain / filtered-EEG gain over chance): 0.83.
