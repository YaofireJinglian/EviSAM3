#!/usr/bin/env bash
# Pure SAM3 baseline on RRSIS-D val+test, single GPU (card 5), no EVF, no fine-tuning.
set -e

cd ${PROJECT_ROOT}/test-s

LOG_DIR=${LOG_ROOT}
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=5 \
${PYTHON_BIN} \
    scripts/eval_sam3_baseline.py \
    --config ${PROJECT_ROOT}/test-s/configs/rrsisd-evf-sam3.yaml \
    --splits val,test \
    --confidence_threshold 0.5 \
    --mask_threshold 0.5 \
    2>&1 | tee -a "${LOG_DIR}/eval-sam3-baseline.log"
