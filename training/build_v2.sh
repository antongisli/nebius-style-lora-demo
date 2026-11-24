#!/usr/bin/env bash
# Build dataset v2 (v1 plus generated images), upload it, and submit a sweep on it.
#
#   1. Put generated images in dataset/gen/raw/, named by subject (battle-axe.png).
#   2. training/build_v2.sh --ranks 16 --lrs 2e-4,4e-4   # sweep.py options; default is all 8 runs
#
# Stops after the snapped images and their contact sheet unless CONFIRM=1, so the
# images can be reviewed before anything is uploaded or submitted. DRY_RUN=1 prints
# the job commands instead of submitting them. Use a new VERSION (v3, ...) when re-running
# with a different image set: uploads add files but never delete old ones from the bucket.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VERSION="${VERSION:-v2}"
PREFIX="${PREFIX:-rpg}"

uv run -q training/snap_palette.py dataset/gen/raw dataset/gen/processed
if [[ -z "${CONFIRM:-}" ]]; then
  echo "Review dataset/gen/processed/sheet.jpg and the captions, then re-run with CONFIRM=1."
  exit 0
fi

rm -rf "dataset/$VERSION"
mkdir -p "dataset/$VERSION"
cp dataset/v1/*.png dataset/v1/*.txt "dataset/$VERSION/"
cp dataset/gen/processed/*.png dataset/gen/processed/*.txt "dataset/$VERSION/"
{ cat dataset/v1-manifest.csv; tail -n +2 dataset/gen/processed/manifest.csv; } >"dataset/$VERSION-manifest.csv"
echo "dataset/$VERSION: $(find "dataset/$VERSION" -name "*.png" | wc -l | tr -d ' ') images"

if [[ -z "${DRY_RUN:-}" ]]; then
  uv run -q infra/bucket.py put "dataset/$VERSION" "dataset/$VERSION"
fi
python3 training/sweep.py --prefix "$PREFIX" "$@"
DATASET="dataset/$VERSION" training/submit_sweep.sh
