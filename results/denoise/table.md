| Layout | Method | Artifact windows: RRMSE | corr. | alpha error (dB) | beta error (dB) | Clean windows: RRMSE | Windows kept |
|---|---|---|---|---|---|---|---|
| muse (4 ch) | filter only | 0.887 | 0.844 | 0.18 | 0.85 | 0.022 | 100% |
| muse (4 ch) | reject >200 uV | 0.639 | 0.879 | 0.12 | 0.45 | 0.022 | 89% |
| muse (4 ch) | ASR k=3 | 0.599 | 0.806 | 2.23 | 1.26 | 0.306 | 100% |
| muse (4 ch) | ASR k=5 | 0.599 | 0.807 | 2.21 | 1.25 | 0.303 | 100% |
| muse (4 ch) | ASR k=10 | 0.519 | 0.870 | 0.83 | 0.72 | 0.074 | 100% |
| muse (4 ch) | ASR k=20 | 0.660 | 0.862 | 0.28 | 0.67 | 0.021 | 100% |
| openbci (8 ch) | filter only | 0.688 | 0.916 | 0.11 | 0.40 | 0.019 | 100% |
| openbci (8 ch) | reject >200 uV | 0.488 | 0.934 | 0.07 | 0.26 | 0.019 | 86% |
| openbci (8 ch) | ASR k=3 | 0.560 | 0.831 | 1.94 | 0.83 | 0.390 | 100% |
| openbci (8 ch) | ASR k=5 | 0.559 | 0.832 | 1.92 | 0.83 | 0.388 | 100% |
| openbci (8 ch) | ASR k=10 | 0.424 | 0.911 | 0.63 | 0.44 | 0.100 | 100% |
| openbci (8 ch) | ASR k=20 | 0.488 | 0.928 | 0.15 | 0.35 | 0.018 | 100% |
| emotiv (14 ch) | filter only | 0.499 | 0.935 | 0.07 | 0.35 | 0.016 | 100% |
| emotiv (14 ch) | reject >200 uV | 0.374 | 0.949 | 0.05 | 0.17 | 0.016 | 86% |
| emotiv (14 ch) | ASR k=3 | 0.458 | 0.894 | 0.86 | 0.48 | 0.237 | 100% |
| emotiv (14 ch) | ASR k=5 | 0.458 | 0.894 | 0.86 | 0.48 | 0.237 | 100% |
| emotiv (14 ch) | ASR k=10 | 0.365 | 0.939 | 0.26 | 0.32 | 0.088 | 100% |
| emotiv (14 ch) | ASR k=20 | 0.423 | 0.942 | 0.07 | 0.31 | 0.015 | 100% |
| lab cap (32 ch) | filter only | 0.380 | 0.970 | 0.04 | 0.14 | 0.007 | 100% |
| lab cap (32 ch) | reject >200 uV | 0.271 | 0.978 | 0.02 | 0.07 | 0.007 | 85% |
| lab cap (32 ch) | ASR k=3 | 0.360 | 0.942 | 0.46 | 0.29 | 0.193 | 100% |
| lab cap (32 ch) | ASR k=5 | 0.360 | 0.942 | 0.46 | 0.29 | 0.193 | 100% |
| lab cap (32 ch) | ASR k=10 | 0.307 | 0.966 | 0.16 | 0.19 | 0.054 | 100% |
| lab cap (32 ch) | ASR k=20 | 0.332 | 0.972 | 0.04 | 0.14 | 0.006 | 100% |

Means over 2 s windows from 20 simulated 60 s recordings per layout (seed 0). RRMSE = RMS(error) / RMS(clean signal).
