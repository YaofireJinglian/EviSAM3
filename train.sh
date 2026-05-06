cd ${PROJECT_ROOT}/test-s
mkdir -p ${LOG_ROOT}
ln -sfn ${LOG_ROOT}/_rrsisd-evf-sam3 ${LOG_ROOT}/_rrsisd-evf-sam3
CUDA_VISIBLE_DEVICES=0,1,2,3,4 ${PYTHON_BIN} -m torch.distributed.run --nnodes 1 --nproc_per_node 5 --master_port=29501 train_evf.py --config ${PROJECT_ROOT}/test-s/configs/rrsisd-evf-sam3.yaml --resume ${LOG_ROOT}/_rrsisd-evf-sam3/model_epoch_last.pth 2>&1 | tee -a ${LOG_ROOT}/train-rrsisd.log
