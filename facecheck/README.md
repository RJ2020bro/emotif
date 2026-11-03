# facecheck

Audit an EEG decoder for the two leaks that most often inflate reported accuracy:

1. **Split leakage.** If every recording has one label and windows are split at random, windows of the same
   recording land in training and test, and a model can score well by recognising the recording. facecheck gives
   every recording a random label and reruns the same split: if accuracy stays high, the protocol measures
   recording identity, not the state.
2. **Face leakage.** Blinks, eye movements and jaw or forehead muscle activity are large in scalp EEG and change
   with emotion and effort. facecheck decodes from face indices alone and measures how much of the EEG decoder's
   output they explain.

```bash
pip install ./facecheck
facecheck audit data.npz            # arrays X, y, recording, subject (+ optional face)
```

```python
import facecheck as fc
F, starts = fc.face_indices(eeg_uV, fs=128)        # (windows, channels*3): eye, jaw 20-30 Hz, jaw 30-45 Hz
r = fc.audit(X, y, recording, subject, face=F, model="logreg")   # windows in time order within each recording
print(fc.report(r))
```

| Output | Meaning |
| --- | --- |
| `random_split` | accuracy with windows split at random (the common protocol) |
| `scrambled_labels` | same split with a random label per recording; above chance = recording identification |
| `temporal_split` | train on the first 70 % of each recording, test on the last 30 % |
| `subject_split` | all windows of a subject in one fold: accuracy on new people |
| `face_only_random`, `face_only_subject` | the same decoder given only the face indices |
| `face_share` | cross-validated R² of the subject-wise decoder's output explained by the face indices |
| `leakage_ratio` | face-only gain over chance / EEG gain over chance, on new subjects |

Known-answer tests (`python facecheck/tests/test_facecheck.py`): with labels carried only by a recording
fingerprint, the random split scores 0.81 and so do scrambled labels, while new subjects are at chance; with labels
in brain features the face checks stay at chance; with labels in the face, the face-only decoder matches the EEG
decoder and the leakage ratio exceeds 0.8.

Part of [Emotif](../README.md). MIT licence.
