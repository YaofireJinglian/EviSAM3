#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../common.sh"
# Evaluate the pretrained SAM3 baseline without EviSAM3 adaptation.
exec "$PYTHON_BIN" tools/evaluation/sam3_baseline.py \
  --config configs/evisam3/rrsisd-evf-sam3.yaml \
  --splits val,test --confidence_threshold 0.5 --mask_threshold 0.5 "$@"
