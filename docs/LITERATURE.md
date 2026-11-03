# What the five papers say, and what Emotif does with it

| Paper | Key findings | Used in Emotif | Gap Emotif addresses |
|---|---|---|---|
| **Fernandes et al. 2024**, *Bioengineering* 11:782. LR, KNN, SVM, DNN and DGCNN on SEED (62 ch). | Differential entropy (DE) beat PSD for every classifier (LR 82.5 %, SVM 84.4 %, DNN 85.2 %, DGCNN 90.0 %). Asymmetry features (DASM, RASM) did nearly as well with fewer dimensions. β and γ were the most informative bands. Evaluation was subject-dependent only; the authors list subject-independent testing and adaptive filtering as future work. | DE in 5 bands; 7 left–right asymmetry pairs; a Gaussian-kernel graph prior as in DGCNN. | Every result is subject-dependent on one 62-channel lab system. Emotif reports leave-one-subject-out results on a consumer headset. On consumer EEG, β and γ are exactly the bands muscle dominates (Whitham et al. 2007), so their high accuracy needs a face control there. |
| **Liu et al. 2024**, arXiv 2408.06027. Survey of graph methods for EEG emotion recognition. | Taxonomy: node features (DE is common because SEED ships it), edges (prior/physical-distance vs learned), aggregation (Chebyshev / GCN filters), graph manipulation (brain-region graphs; IAG groups electrodes into 17 regions). Learned edges overfit when data are small. | Region-level graph over 17 scalp regions with prior edges (great-circle Gaussian kernel), 2-hop linear diffusion; no learned edges because consumer datasets are small. | All surveyed methods assume a fixed electrode set. Emotif's graph is over regions, not electrodes, and its diffusion only averages *observed* regions, so it runs on any headset and fills uncovered regions from the nearest covered ones. |
| **Gkintoni et al. 2025**, *Brain Sciences* 15:220. PRISMA review of 64 studies. | Multimodal methods exceed 90 % in some studies; obstacles are real-time cost, inconsistent labels, non-standard protocols and datasets, motion artifacts in wearables, and ethics (privacy, bias). | Real-time design (0.6 ms per window), local-only data, SAM self-report labels, published protocols, sham control. | Multimodal "EEG + face" fusion can look like better EEG decoding while the gain comes from the face. Emotif measures the face's share instead of adding it. |
| **Li et al. 2022**, *ACM Computing Surveys* 55. Tutorial and review. | Recommends ASR plus common-average reference for artifacts; warns that random segment-level splits leak between train and test; baseline calibration and alignment for cross-subject work; domain shift is the central problem. | ASR (with stability fixes), recording-level cross-validation, per-person baseline normalisation. | Extends their leakage warning to significance testing: permutations must shuffle whole recordings. With 4 people × 6 recordings, the chance level is 0.42–0.47, not 0.33. |
| **Li, Yan et al. 2025**, *Biomed. Signal Process. Control* 109188 (abstract and section headings only). | Reviews CNN, CapsNet, RNN and hybrids; describes EEG as "resistant to artifacts" and an objective indicator of emotion. | — | Tests that claim on consumer hardware: on the Muse dataset, 3 face numbers per electrode predict the label as well as the EEG does (leakage ratio 1.01). |

## Design decisions traced to the literature
- **Features:** DE in δ θ α β γ (all five papers).
- **Artifact repair:** ASR (Mullen et al. 2015; recommended by Li et al. 2022), redesigned for stability (geometric-median covariance, energy guard, 2.5× floor).
- **Artifact detection:** EEGdenoiseNet (Zhang et al. 2021) for clean / eye / muscle examples.
- **Spatial model:** brain-region graph with prior edges (Liu et al. 2024: prior edges suit small data; IAG's 17 regions).
- **Evaluation:** subject-independent (LOSO) and recording-wise within-subject; balanced accuracy; recording-level permutation tests.
- **Calibration:** temperature scaling (Guo et al. 2017).
- **Face control:** muscle above 20 Hz (Whitham et al. 2007; Goncharova et al. 2003), blinks below 3 Hz.
- **Brain-only decoder:** face indices measured live and their predicted contribution subtracted (regression partialling, as in the audit), compared against linear concept erasure (LEACE; Belrose et al. 2023). Interpretation of the result against frontal asymmetry (Davidson 1992).

## Full references
- Fernandes, J. V. M. R., et al. (2024). Emotion detection from EEG signals using machine deep learning models. *Bioengineering*, 11(8), 782. https://doi.org/10.3390/bioengineering11080782
- Liu, C., Zhou, X., Wu, Y., Ding, Y., Zhai, L., Wang, K., Jia, Z., & Liu, Y. (2024). A comprehensive survey on EEG-based emotion recognition: A graph-based perspective. arXiv:2408.06027.
- Gkintoni, E., Aroutzidis, A., Antonopoulou, H., & Halkiopoulos, C. (2025). From neural networks to emotional networks: A systematic review of EEG-based emotion recognition in cognitive neuroscience and real-world applications. *Brain Sciences*, 15(3), 220. https://doi.org/10.3390/brainsci15030220
- Li, X., Zhang, Y., Tiwari, P., Song, D., Hu, B., Yang, M., Zhao, Z., Kumar, N., & Marttinen, P. (2022). EEG based emotion recognition: A tutorial and review. *ACM Computing Surveys*, 55(4), 1–57. https://doi.org/10.1145/3524499
- Li, X., Yan, Z., Gong, P., Lin, D., & Chen, F. (2025). Research progress on emotion recognition based on electroencephalogram signals. *Biomedical Signal Processing and Control*, 109188. https://doi.org/10.1016/j.bspc.2025.109188
- Whitham, E. M., et al. (2007). Scalp electrical recording during paralysis: Quantitative evidence that EEG frequencies above 20 Hz are contaminated by EMG. *Clinical Neurophysiology*, 118(8), 1877–1888.
- Goncharova, I. I., McFarland, D. J., Vaughan, T. M., & Wolpaw, J. R. (2003). EMG contamination of EEG: spectral and topographical characteristics. *Clinical Neurophysiology*, 114(9), 1580–1593.
- Mullen, T. R., et al. (2015). Real-time neuroimaging and cognitive monitoring using wearable dry EEG. *IEEE Transactions on Biomedical Engineering*, 62(11), 2553–2567.
- Zhang, H., et al. (2021). EEGdenoiseNet: A benchmark dataset for deep learning solutions of EEG denoising. *Journal of Neural Engineering*, 18(5), 056057.
- Song, T., Zheng, W., Song, P., & Cui, Z. (2018). EEG emotion recognition using dynamical graph convolutional neural networks. *IEEE Transactions on Affective Computing*, 11(3), 532–541.
- Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023). LEACE: Perfect linear concept erasure in closed form. *Advances in Neural Information Processing Systems 36* (NeurIPS 2023).
- Davidson, R. J. (1992). Anterior cerebral asymmetry and the nature of emotion. *Brain and Cognition*, 20(1), 125–151.
- Alakus, T. B., Gonen, M., & Turkoglu, I. (2020). Database for an emotion recognition system based on EEG signals and various computer games – GAMEEMO. *Biomedical Signal Processing and Control*, 60, 101951.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Bird, J. J., Manso, L. J., Ribeiro, E. P., Ekárt, A., & Faria, D. R. (2018). A study on mental state classification using EEG-based brain-machine interface. *International Conference on Intelligent Systems*.

- Henriques, J. B., & Davidson, R. J. (1991). Left frontal hypoactivation in depression. *Journal of Abnormal Psychology*, 100(4), 535–545. (source of the frontal-alpha-asymmetry marker tested for depression)
- van der Vinne, N., Vollebregt, M. A., van Putten, M. J. A. M., & Arns, M. (2017). Frontal alpha asymmetry as a diagnostic marker in depression: Fact or fiction? A meta-analysis. *NeuroImage: Clinical*, 16, 79–87. (the marker does not separate depressed from healthy people reliably; Emotif's null result agrees)
- Newson, J. J., & Thiagarajan, T. C. (2019). EEG frequency bands in psychiatric disorders: A review of resting state studies. *Frontiers in Human Neuroscience*, 12, 521. (source of the theta, alpha and beta markers for depression and anxiety)
- Strijkstra, A. M., Beersma, D. G. M., Drayer, B., Halbesma, N., & Daan, S. (2003). Subjective sleepiness correlates negatively with global alpha (8–12 Hz) and positively with central frontal theta (4–8 Hz) frequencies in the human resting awake electroencephalogram. *Neuroscience Letters*, 340(1), 17–20. (source of the sleepiness markers)
- Saeb, S., Lonini, L., Jayaraman, A., Mohr, D. C., & Kording, K. P. (2017). The need to approximate the use-case in clinical machine learning. *GigaScience*, 6(5), gix019. (record-wise vs subject-wise cross-validation; what panel C of fig_states shows)
- Varoquaux, G. (2018). Cross-validation failure: Small sample sizes lead to large error bars. *NeuroImage*, 180, 68–77. (why n = 31–49 results need permutation tests and wide intervals)

Check every reference against the original before citing it yourself; ISEF does not allow AI-generated citations.
