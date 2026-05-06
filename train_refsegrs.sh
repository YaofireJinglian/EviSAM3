cd ${PROJECT_ROOT}/test-s
mkdir -p ${LOG_ROOT}
CUDA_VISIBLE_DEVICES=1 ${PYTHON_BIN} -m torch.distributed.run --nnodes 1 --nproc_per_node 1 train_evf.py --config ${PROJECT_ROOT}/test-s/configs/refsegrs-evf-sam3.yaml --resume ${LOG_ROOT}/_refsegrs-evf-sam3/model_epoch_last.pth 2>&1 | tee -a ${LOG_ROOT}/train-refsegrs.log
