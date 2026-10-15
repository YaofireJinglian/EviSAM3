# Legacy segmentation code

These files retain the earlier generic segmentation training/evaluation path.
The EviSAM3 paper entrypoints are `tools/train.py` and `tools/evaluate.py`.

- `train.py`: earlier distributed trainer.
- `evaluate.py` and `evaluate.sh`: earlier prediction and metric exporter.
- `utils.py` and `sod_metric.py`: utilities and salient-object detection metrics.
- `mmseg/`: retained MMSegmentation components; not used by the EviSAM3 entrypoints.
- `../configs/legacy/`: configurations for this older workflow.

Run entrypoints from the repository root. These workflows need their original
datasets and checkpoints and are not validated as reproductions of EviSAM3.

To import the retained MMSegmentation package directly, add `legacy/` to
`PYTHONPATH`. Its optional original dependencies are still required.
