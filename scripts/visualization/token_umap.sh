#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
# Usage: bash scripts/visualization/token_umap.sh CHECKPOINT [visualization arguments...]
CHECKPOINT="${1:?Provide a trained checkpoint path}"
shift
OUT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/token_umap}"
mkdir -p "$OUT"
exec "$PYTHON_BIN" tools/visualization/token_umap.py \
  --config configs/evisam3/rrsisd-evf-sam3.yaml \
  --evf_ckpt "$CHECKPOINT" --output_dir "$OUT" --split val \
  --num_samples 60 --per_class_per_sample 120 --block_index -1 --method umap --seed 42 --filename token_umap.png "$@"
