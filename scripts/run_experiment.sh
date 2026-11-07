#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${1:-42}"
PY=.venv/bin/python
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
$PY -m coursework.download
for MODEL in fastvit repvit; do
  $PY -m coursework.train --model "$MODEL" --seed "$SEED"
  $PY -m coursework.evaluate "runs/full/$MODEL-seed$SEED/best.pt"
  $PY -m coursework.export "runs/full/$MODEL-seed$SEED/best.pt"
done
$PY -m coursework.bundle --seed "$SEED"
$PY -m coursework.report
