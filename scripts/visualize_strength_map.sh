#!/usr/bin/env bash
# Strength Map (D1): 2-D heatmap of mean delta-IoU = IoU(Full) - IoU(SAM3)
# binned over (object area / image area, prompt length in #words). Companion
# heatmap shows mean IoU(Full). 400 random val samples, seed 42.
#
# Single GPU 6.
set -e

cd ${PROJECT_ROOT}/test-s

OUT=${OUTPUT_ROOT}
mkdir -p "${OUT}"

CONFIG=${PROJECT_ROOT}/test-s/configs/rrsisd-evf-sam3.yaml
EVF_CKPT=${LOG_ROOT}/_rrsisd-evf-sam3/model_epoch_ep50.pth

PY=${PYTHON_BIN}

CUDA_VISIBLE_DEVICES=6 \
${PY} scripts/visualize_strength_map.py \
    --config "${CONFIG}" \
    --evf_ckpt "${EVF_CKPT}" \
    --output_dir "${OUT}" \
    --split val \
    --num_samples 400 \
    --seed 42 \
    --filename strength_map.png \
    2>&1 | tee -a "${OUT}/strength_map.log"

echo "Done. Strength map at: ${OUT}/strength_map.png"
