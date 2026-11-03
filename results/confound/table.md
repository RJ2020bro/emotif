| Pipeline | LOSO balanced accuracy | Windows kept | Channel-windows masked |
|---|---|---|---|
| Filter only | 0.661 | 100% | 0% |
| Filter + amplitude rejection | 0.706 | 76% | 5% |
| Filter + ASR | 0.617 | 100% | 0% |
| Filter + ASR + artifact detector | 0.564 | 88% | 43% |
| Eye/muscle detector probabilities only (no EEG spectra) | 0.676 | 100% | |

Chance = 0.333. Target: arousal.
