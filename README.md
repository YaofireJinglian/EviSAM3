## Anonymous Submission — Code Release

This repository contains the code for our anonymous submission. Author information, citations, and any acknowledgements have been intentionally removed for double-blind review. The full citation and author list will be added upon acceptance.

## Environment
This code was implemented with Python 3.11 and PyTorch 2.3.0. You can install all the requirements via:
```bash
pip install -r requirements.txt
```


## Quick Start
1. Download the dataset and put it in ./load.
2. Download the pre-trained backbone weights and put them in ./pretrained.
3. Training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 loadddptrain.py --config configs/demo.yaml
```
Note: training is memory-intensive; we use 4 x A100 GPUs.

4. Evaluation:
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Train
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run train.py --nnodes 1 --nproc_per_node 4 --config [CONFIG_PATH]
```

## Test
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Acknowledgement of Third-Party Code

Parts of this codebase are adapted from publicly released third-party projects. Original copyright notices are preserved in the corresponding source files. We do not claim authorship over those components.
