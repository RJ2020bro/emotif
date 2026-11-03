#!/usr/bin/env bash
# Downloads EEGdenoiseNet (Zhang et al. 2021, MIT licence): 4514 clean EEG, 3400 EOG and 5598 EMG 2-s epochs at 256 Hz.
# Used to train the artifact detector and for the ground-truth denoising benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/ncclabsustech/EEGdenoiseNet "$tmp/edn"
(cd "$tmp/edn" && git checkout HEAD -- LICENSE data/EEG_all_epochs.npy data/EOG_all_epochs.npy data/EMG_all_epochs.npy)
mkdir -p data/external/eegdenoisenet
cp "$tmp"/edn/data/*_all_epochs.npy "$tmp/edn/LICENSE" data/external/eegdenoisenet/
rm -rf "$tmp"
echo "saved to data/external/eegdenoisenet"
