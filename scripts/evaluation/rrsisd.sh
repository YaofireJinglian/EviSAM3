#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
# Usage: bash scripts/evaluation/rrsisd.sh CHECKPOINT [SPLIT] [evaluation arguments...]
CHECKPOINT="${1:?Provide a trained checkpoint path}"
SPLIT="${2:-test}"
shift
if (($#)); then shift; fi
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 \
  tools/evaluate.py --config configs/evisam3/rrsisd-evf-sam3.yaml \
  --checkpoint "$CHECKPOINT" --split "$SPLIT" "$@"
