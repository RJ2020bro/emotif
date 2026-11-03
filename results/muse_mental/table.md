| Target | Model | Layout | No aug. bal. acc | Aug. bal. acc | Wilcoxon p |
|---|---|---|---|---|---|
| arousal | logreg | full | 0.601 ± 0.165 | 0.580 ± 0.181 | 0.250 |
| arousal | logreg | brainbit | 0.523 ± 0.070 | 0.558 ± 0.193 | 0.625 |
| arousal | logreg | frontal_2ch | 0.488 ± 0.094 | 0.401 ± 0.066 | 0.125 |
| arousal | mlp | full | 0.578 ± 0.154 | 0.574 ± 0.242 | 1.000 |
| arousal | mlp | brainbit | 0.504 ± 0.130 | 0.568 ± 0.204 | 0.250 |
| arousal | mlp | frontal_2ch | 0.477 ± 0.106 | 0.422 ± 0.140 | 0.250 |
| arousal | graph | full | 0.602 ± 0.141 | 0.572 ± 0.158 | 0.250 |
| arousal | graph | brainbit | 0.480 ± 0.121 | 0.546 ± 0.225 | 0.250 |
| arousal | graph | frontal_2ch | 0.476 ± 0.095 | 0.401 ± 0.103 | 0.125 |

Mean ± 95 % CI across 4 held-out subjects (leave-one-subject-out). Chance balanced accuracy = 1 / number of classes.
