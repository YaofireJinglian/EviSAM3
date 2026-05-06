#!/usr/bin/env bash
# B1: Image-token UMAP atlas before vs after the EvidenceRectifiedAdapter
# of the last ViT block, coloured by GT foreground/background. Reports
# silhouette score on the high-dim (pre vs post) embeddings.
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
${PY} scripts/visualize_token_umap.py \
    --config "${CONFIG}" \
    --evf_ckpt "${EVF_CKPT}" \
    --output_dir "${OUT}" \
    --split val \
    --num_samples 60 \
    --per_class_per_sample 120 \
    --block_index -1 \
    --method umap \
    --seed 42 \
    --filename token_umap.png \
    2>&1 | tee -a "${OUT}/token_umap.log"

echo "Done. Token UMAP at: ${OUT}/token_umap.png"
