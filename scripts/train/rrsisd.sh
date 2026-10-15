#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
# Usage: bash scripts/train/rrsisd.sh [NUM_GPUS] [training arguments...]
NUM_GPUS="${1:-1}"
if (($#)); then shift; fi
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  tools/train.py --config configs/evisam3/rrsisd-evf-sam3.yaml \
  --name rrsisd_evfsam3 "$@"
