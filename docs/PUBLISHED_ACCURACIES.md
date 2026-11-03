# Published accuracies on the datasets Emotif audits

Collected November 2026 from the pages linked. Check each paper yourself before citing it: several statements below
come from abstracts or from other papers' comparison tables, and some papers do not say how windows were split.

| Dataset | Paper (page read) | Reported accuracy | Split as described |
|---|---|---|---|
| GAMEEMO, 4 games | DEEPHER, LSTM (Engineering Proceedings 10(1):32, [mdpi.com/2673-4591/10/1/32](https://mdpi.com/2673-4591/10/1/32)) | 99.99 % | 70 % train / 30 % test; unit of the split not stated |
| GAMEEMO | Alakus et al. 2020 (dataset paper), as cited by the DEEPHER paper | 80 % | not stated in the citing paper |
| GAMEEMO | GoogLeNet on CWT images, cited as [15] in Pattern Anal. Appl. 2025 ([link](https://link.springer.com/article/10.1007/s10044-025-01501-1)) | 98.78 % | 10-fold CV |
| GAMEEMO | BiLSTM with spectral entropy, cited as [18] there | 76.93 % | 10-fold CV |
| GAMEEMO | MFIF + Hjorth/entropy + KNN, cited as [31] there | 97.857 % | not stated |
| GAMEEMO | Wavelet scattering, multi-channel voting (Pattern Anal. Appl. 2025, link above) | 79.0 % per channel; 100 % with voting | leave-one-subject-out |
| GAMEEMO | GA-PDPL ([PMC10377713](https://pmc.ncbi.nlm.nih.gov/articles/PMC10377713/)) | 64.34 % (2-class), 49.01 % (4-class) | leave-one-subject-out |
| GAMEEMO | LSTM-GRU ([arXiv 2508.21103](https://arxiv.org/pdf/2508.21103)) | 93.3 % binary valence, 94.5 % multi-class (the paper also states 86 % / 99 % elsewhere) | "80:20 subject-independent"; window assignment not explained |
| GAMEEMO | Shahzad et al., cited as [2] in arXiv 2508.21103 | 98.21 % (random forest) | not stated in the citing paper |
| Depression, PHQ-9 (Mendeley ct5vrc4k2t) | Sakib, Islam & Faruk 2023, Comput. Intell. Neurosci. ([repository abstract](https://ar.iub.edu.bd/handle/123456789/558)) | 98.43 % (KNN, 5-s samples) | 5-fold CV; a 70/30 split also reported (98.10 %) |
| Other lab's stress data (Emotiv EPOC X) | CubicPat, Diagnostics 2025 ([PDF](https://research.usq.edu.au/download/b89c9bac55c74a11f730cb9f871193094258cb2e0532ec10f30817daba0f3318/2432308/diagnostics-15-00363-v2.pdf)) | 96.29 % 10-fold; 76.17 % leave-one-subject-out | both reported: shows the same gap on another dataset |

What Emotif measures on the same data (`results/leaderboard/`, `research/leaderboard.py`): see RESULTS.md section 3e.

Pattern: papers that split windows (or segments) at random, or do not say, report 94–99.99 %; papers that hold out
whole subjects report 49–79 % per window. A claim of 100 % with per-subject voting deserves the same scrutiny.
