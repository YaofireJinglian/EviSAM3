#!/bin/bash
# =============================================================
# EviSAM3 Training Script for RRSIS-D Dataset
# =============================================================
# Usage:
#   bash scripts/train_rrsisd.sh [NUM_GPUS]
#
# Example:
#   bash scripts/train_rrsisd.sh 2

NUM_GPUS=${1:-1}
CONFIG="configs/rrsisd-evf-sam3.yaml"
PYTHON="${PYTHON_BIN}"

echo "==========================================="
echo " EviSAM3 Training on RRSIS-D"
echo " GPUs: ${NUM_GPUS}"
echo " Config: ${CONFIG}"
echo "==========================================="

cd "$(dirname "$0")/.."

${PYTHON} -m torch.distributed.run --nproc_per_node=${NUM_GPUS} \
    train_evf.py \
    --config ${CONFIG} \
    --name rrsisd_evfsam3 \
    --tag run1
