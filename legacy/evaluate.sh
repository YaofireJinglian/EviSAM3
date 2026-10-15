#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
# Usage: bash legacy/evaluate.sh --model CHECKPOINT [legacy evaluation arguments...]
exec "$PYTHON_BIN" legacy/evaluate.py --config configs/legacy/cod-sam-vit-l_all.yaml "$@"
