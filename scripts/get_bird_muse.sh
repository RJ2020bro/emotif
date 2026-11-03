#!/usr/bin/env bash
# Downloads the Bird et al. (2018) Muse recordings used for the bundled arousal model and the pilot benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
git clone --depth 1 https://github.com/jordan-bird/eeg-feature-generation "$tmp/bird"
mkdir -p data/external
rm -rf data/external/bird_muse
cp -r "$tmp/bird/dataset/original_data" data/external/bird_muse
rm -rf "$tmp"
echo "saved to data/external/bird_muse ($(ls data/external/bird_muse | wc -l) files)"
