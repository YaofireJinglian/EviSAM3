# Preparation and reproduction notes

[Back to the project README](../README.md)

These notes describe the supplied source code. The paper result tables are taken from the manuscript; no training or benchmark run was performed while preparing this documentation.

## Environment

The entrypoints initialize NCCL at import time, so use Linux with NVIDIA GPUs and launch both training and evaluation through `torchrun`, including single-GPU evaluation.

The bundled [SAM3 instructions](../models/README.md#installation) specify Python 3.12+, PyTorch 2.7+, and CUDA 12.6+. A starting environment consistent with those requirements is:

```bash
conda create -n evisam3 python=3.12 -y
conda activate evisam3

python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu126

python -m pip install \
  "numpy>=1.26,<2" scipy matplotlib Pillow PyYAML tqdm \
  imageio opencv-python scikit-learn tensorboardX attrs \
  "typing_extensions>=4.11" "timm==0.4.12" \
  einops sentencepiece fairscale huggingface_hub iopath \
  ftfy regex pycocotools psutil

export PYTHONPATH="$PWD/third_party/unilm:$PWD/models:$PWD:${PYTHONPATH:-}"
```

The CUDA wheel pairing follows the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v270). The supplemental packages above are derived from repository imports. This setup has not been verified by an end-to-end training run. Additional dependencies may be needed by optional analysis tools.

The old root `requirements.txt` references Python 3.11 / PyTorch 2.3-era dependencies, including NumPy 1.24 and Pillow 9.4. It should be reconciled with the SAM3 requirements before being used in a Python 3.12 environment.

`third_party/unilm/` exposes the bundled TorchScale package. Keep it on `PYTHONPATH` so BEiT3 can resolve imports such as `torchscale.model.BEiT3`. The project already includes SAM3 code under `models/`; preserve those local components when using pretrained SAM3 weights.

## Dataset source module

The original supplementary code package includes:

```text
datasets/
├── __init__.py
├── paired_image_folders.py
├── refsegrs_refer.py
├── rrsisd_refer.py
├── rrsisd_wrapper.py
└── wrappers.py
```

These files define the local `datasets.make(...)` registry used by `tools/train.py` and `tools/evaluate.py`. This is the project's own module; installing Hugging Face's similarly named `datasets` package does not supply this registry.

The Git checkout currently lacks this source package. Restore the original six
Python files to `datasets/` before running training or evaluation. The root
`.gitignore` now excludes only `/data/` and checkpoint/output directories, so
restored dataset source files can be committed normally. The cleanup does not
substitute an unverified loader implementation for the missing source.

## Data and weights

### RRSIS-D

Download the data and annotations from the [official RMSIN repository](https://github.com/Lsan2401/RMSIN). The supplied loader delegates annotation handling to [`refer/refer.py`](../refer/refer.py):

```text
/path/to/data/RRSIS-D/
├── rrsisd/
│   ├── refs(unc).p
│   └── instances.json
└── images/
    └── rrsisd/
        └── JPEGImages/
            └── <image filename from instances.json>
```

Set `refer_data_root` to `/path/to/data/RRSIS-D`, `dataset` to `rrsisd`, and `split_by` to `unc`. Binary ground-truth caches are created under `mask_cache/<split>/`; the source dataset directory must be writable unless a separate cache location is supplied.

### RefSegRS

Download RefSegRS from the [authors' repository](https://github.com/zhu-xlab/rrsis) or [dataset page](https://huggingface.co/datasets/JessicaYuan/RefSegRS). The original `refsegrs_refer.py` expects:

```text
/path/to/data/RefSegRS/
├── images/
│   └── <id>.tif
├── masks/
│   └── <id>.tif
├── output_phrase_train.txt
├── output_phrase_val.txt
└── output_phrase_test.txt
```

Each phrase-file line is `<id> <referring expression>`. Image and mask basenames must match that ID. Masks are treated as binary foreground wherever the pixel value is greater than zero. The loader also writes a `mask_cache/<split>/` directory.

Both loaders return an image for SAM3 at 1008 × 1008, an image for BEiT3 at 224 × 224, a binary target mask, a caption, and sample metadata.

### Pretrained components

```text
/path/to/pretrained/
├── sam3.pt
├── beit3_large_patch16_224.pth
└── beit3.spm
```

- SAM3 weights: follow the access and download steps in the [official SAM3 documentation](https://github.com/facebookresearch/sam3#getting-started).
- BEiT3 Large and its SentencePiece tokenizer: use the [official BEiT3 resources](https://github.com/microsoft/unilm/tree/master/beit3).

Use the SAM3 checkpoint expected by this code. A pretrained backbone checkpoint is distinct from a trained EviSAM3 checkpoint.

### Configuration paths

For each selected YAML file, replace all `${DATA_ROOT}` and `${CKPT_ROOT}` strings with actual directories. The code uses `yaml.load(...)` without environment expansion, so setting shell variables alone does not resolve these YAML values.

Check these locations:

- Top-level `refer_data_root`, `sam_checkpoint`, `encoder_pretrained`, and `beit3_spm`.
- The `refer_data_root` inside every train/validation/test dataset specification.
- `model.args.encoder_pretrained`.

The provided YAML files currently use a base learning rate of `2e-5`, adapter/expert rates of `1e-4`, and a cosine schedule. The manuscript describes dataset-specific base rates and a final ten-epoch decay. Review the configuration and scheduler when targeting the paper's exact experimental settings.

## Output paths

Training uses the `LOG_ROOT` environment variable and defaults to `outputs/`.
With `--name rrsisd_evfsam3`, the default checkpoint paths are:

```text
outputs/rrsisd_evfsam3/
├── config.yaml
├── model_epoch_last.pth
└── model_epoch_best.pth
```

The launchers in `scripts/` resolve the repository root automatically and set
`PYTHONPATH` for local modules. They use `python` unless `PYTHON_BIN` is set,
respect `CUDA_VISIBLE_DEVICES`, and forward additional command-line arguments.
Direct Python commands should be run from the repository root after setting
`PYTHONPATH` as shown in the environment section.

## Code organization

Use `tools/train.py` and `tools/evaluate.py` for EviSAM3. Task-specific analysis
lives in `tools/evaluation/` and `tools/visualization/`. The integrated SAM3 implementation retains its existing
`models/sam3/` import path. Upstream evaluation utilities are preserved in
`third_party/sam3_eval/`; BEiT3 and TorchScale live in `third_party/unilm/`.

## Evaluation protocol

For the usual benchmark protocol, binarize each predicted probability map at a fixed mask threshold, calculate each sample's IoU, then compute:

- **Pr@t:** fraction of sample IoUs greater than t.
- **mIoU:** mean sample IoU.
- **oIoU:** total intersection divided by total union.

The [training evaluator](../tools/train.py) uses a fixed probability threshold of 0.5 before applying the IoU thresholds.

The current [standalone test evaluator](../tools/evaluate.py) instead changes the mask probability threshold across 0.5–0.9, computes metrics at each threshold, and averages those mIoU/oIoU values. It also uses the same threshold for the corresponding precision calculation. Its aggregate outputs are therefore not directly comparable to the paper tables under the fixed-threshold protocol.

When reproducing the reported benchmark numbers, align the standalone evaluation with the fixed mask threshold, the intended caption policy, and the dataset split. Both entrypoints currently select the first available caption during evaluation. Use an explicitly supplied, existing trained EviSAM3 checkpoint; the standalone evaluator otherwise continues without loading one.
