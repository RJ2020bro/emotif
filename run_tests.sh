#!/usr/bin/env bash
# Runs every test. Needs: python deps (requirements.txt), node + npm, and playwright with Chromium for the browser test.
set -euo pipefail
cd "$(dirname "$0")"
(cd app && npm install --silent && npx tsc --noEmit -p . && node build.mjs --test-bundles)
python3 tests/test_parity.py
python3 tests/test_cleaning.py
python3 tests/test_train.py
python3 tests/test_server.py
node tests/test_app.mjs
node tests/test_replay.mjs
python3 tests/test_browser.py "${1:-}"
echo "ALL TESTS PASSED"
