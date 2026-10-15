#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
# Usage: bash scripts/visualization/strength_map.sh CHECKPOINT [visualization arguments...]
CHECKPOINT="${1:?Provide a trained checkpoint path}"
shift
OUT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/strength_map}"
mkdir -p "$OUT"
exec "$PYTHON_BIN" tools/visualization/strength_map.py \
  --config configs/evisam3/rrsisd-evf-sam3.yaml \
  --evf_ckpt "$CHECKPOINT" --output_dir "$OUT" --split val \
  --num_samples 400 --seed 42 --filename strength_map.png "$@"
