#!/usr/bin/env bash
# C1: Text-token saliency on EvfSam3. Per sample we compute the
# foreground-targeted gradient of the predicted mask w.r.t. the BEiT-3
# text-token embeddings (gradient * input, L2 over the channel axis)
# and render a coloured-bar strip above the rendered prompt.
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
${PY} scripts/visualize_text_saliency.py \
    --config "${CONFIG}" \
    --evf_ckpt "${EVF_CKPT}" \
    --output_dir "${OUT}" \
    --split val \
    --num_samples 6 \
    --pool_size 120 \
    --min_iou 0.5 \
    --seed 42 \
    --filename text_saliency.png \
    2>&1 | tee -a "${OUT}/text_saliency.log"

echo "Done. Text saliency at: ${OUT}/text_saliency.png"
