#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
# Usage: bash scripts/visualization/text_saliency.sh CHECKPOINT [visualization arguments...]
CHECKPOINT="${1:?Provide a trained checkpoint path}"
shift
OUT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/text_saliency}"
mkdir -p "$OUT"
exec "$PYTHON_BIN" tools/visualization/text_saliency.py \
  --config configs/evisam3/rrsisd-evf-sam3.yaml \
  --evf_ckpt "$CHECKPOINT" --output_dir "$OUT" --split val \
  --num_samples 6 --pool_size 120 --min_iou 0.5 --seed 42 --filename text_saliency.png "$@"
