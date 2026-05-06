#!/bin/bash
# =============================================================
# EviSAM3 Test Script for RRSIS-D Dataset
# =============================================================
# Usage:
#   bash scripts/test_rrsisd.sh [CHECKPOINT_PATH] [SPLIT]
#
# Example:
#   bash scripts/test_rrsisd.sh save/rrsisd_evfsam3_run1/model_epoch_best.pth test

CKPT=${1:-"save/rrsisd_evfsam3_run1/model_epoch_best.pth"}
SPLIT=${2:-"test"}
CONFIG="configs/rrsisd-evf-sam3.yaml"
PYTHON="${PYTHON_BIN}"

echo "==========================================="
echo " EviSAM3 Testing on RRSIS-D"
echo " Checkpoint: ${CKPT}"
echo " Split: ${SPLIT}"
echo "==========================================="

cd "$(dirname "$0")/.."

${PYTHON} -m torch.distributed.run --nproc_per_node=1 \
    test_evf.py \
    --config ${CONFIG} \
    --checkpoint ${CKPT} \
    --split ${SPLIT}
