"""
Pure SAM3 baseline evaluation on RRSIS-D val/test (referring expression segmentation).

- Loads ONLY the original (un-finetuned) SAM3 checkpoint via build_sam3_image_model.
- Uses SAM3's native text prompt API (Sam3Processor.set_text_prompt) — no EVF/BEiT-3.
- Computes the same metrics as test_evf.py: Pr@t, mIoU, oIoU.
- Single-GPU only. Run with `CUDA_VISIBLE_DEVICES=5 python scripts/eval_sam3_baseline.py ...`.

This script does NOT modify any existing files in the repo.
"""

import argparse
import os
import sys
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

# Make repo root importable (so `import datasets`, `from models...` work)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS_DIR)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import datasets  # registers rrsisd-refer
from models.model_builder import build_sam3_image_model
from models.sam3.model.sam3_image_processor import Sam3Processor


_BPE_FALLBACKS = [
    os.path.join(_REPO, "assets", "bpe_simple_vocab_16e6.txt.gz"),
    os.path.join(_REPO, "models", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
    "/data/yz_data/sen_data/sen_env/lib/python3.10/site-packages/clip/bpe_simple_vocab_16e6.txt.gz",
]


def _resolve_bpe(user_path):
    if user_path and os.path.exists(user_path):
        return user_path
    for p in _BPE_FALLBACKS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Cannot find bpe_simple_vocab_16e6.txt.gz. Tried: {_BPE_FALLBACKS}. "
        f"Pass --bpe_path explicitly."
    )


def _load_gt_full_res(meta):
    """Load GT mask at original image resolution (cached PNG, 0/255)."""
    arr = np.array(Image.open(meta["mask_path"]).convert("L"))
    return arr > 127  # bool [H, W]


def _combine_pred_masks(masks_logits, threshold=0.5):
    """
    masks_logits: tensor [N, 1, H, W] in [0, 1] (sigmoid).
    Returns binary numpy mask [H, W]. Union (max over detections) — matches PCS semantics.
    """
    if masks_logits is None or masks_logits.numel() == 0 or masks_logits.shape[0] == 0:
        return None
    combined = masks_logits.squeeze(1).max(dim=0).values  # [H, W] in [0, 1]
    return (combined > threshold).cpu().numpy()


@torch.inference_mode()
def evaluate_split(processor, dataset, thresholds, mask_threshold=0.5, max_samples=None, log_every=200):
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    per_iou = np.zeros(n, dtype=np.float64)
    total_inter = 0.0
    total_union = 0.0
    empty_pred_count = 0

    pbar = tqdm(range(n), desc="eval", leave=True)
    for i in pbar:
        _img_sam3, _img_evf, _target, cap, meta = dataset[i]

        # Match test_evf.py: first caption when list (eval_mode)
        text = cap[0] if isinstance(cap, list) else cap

        pil = Image.open(meta["img_path"]).convert("RGB")
        gt = _load_gt_full_res(meta)  # bool HxW at original res

        state = processor.set_image(pil)
        state = processor.set_text_prompt(text, state)

        pred = _combine_pred_masks(state.get("masks_logits"), threshold=mask_threshold)
        if pred is None:
            pred = np.zeros_like(gt, dtype=bool)
            empty_pred_count += 1
        elif pred.shape != gt.shape:
            # Should not happen (Sam3Processor resizes to original_h/w), but be safe.
            from PIL import Image as _Image

            pred_img = _Image.fromarray(pred.astype(np.uint8) * 255).resize(
                (gt.shape[1], gt.shape[0]), _Image.NEAREST
            )
            pred = np.array(pred_img) > 127

        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = float(inter) / (float(union) + 1e-8) if union > 0 else 0.0
        per_iou[i] = iou
        total_inter += inter
        total_union += union

        if (i + 1) % log_every == 0:
            pbar.set_postfix(mIoU=f"{per_iou[: i + 1].mean():.4f}", empty=empty_pred_count)

    results = {}
    for t in thresholds:
        results[f"Pr@{t}"] = float((per_iou > t).mean())
    results["mIoU"] = float(per_iou.mean())
    results["oIoU"] = float(total_inter / max(total_union, 1e-8))
    results["_empty_pred"] = int(empty_pred_count)
    results["_n"] = int(n)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(_REPO, "configs", "rrsisd-evf-sam3.yaml"))
    parser.add_argument("--splits", default="val,test", help="Comma list of splits (val,test).")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="SAM3 detection confidence threshold (filter low-score boxes).")
    parser.add_argument("--mask_threshold", type=float, default=0.5,
                        help="Sigmoid threshold to binarize the union mask.")
    parser.add_argument("--bpe_path", default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap for quick smoke-test.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[!] CUDA not available, falling back to CPU (will be very slow).")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    sam_ckpt = config.get("sam_checkpoint")
    if not sam_ckpt or not os.path.exists(sam_ckpt):
        raise FileNotFoundError(f"sam_checkpoint not found: {sam_ckpt}")

    bpe_path = _resolve_bpe(args.bpe_path)
    print(f"[*] config       : {args.config}")
    print(f"[*] sam_ckpt     : {sam_ckpt}")
    print(f"[*] bpe_path     : {bpe_path}")
    print(f"[*] device       : {device}")
    print(f"[*] conf_thr     : {args.confidence_threshold}")
    print(f"[*] mask_thr     : {args.mask_threshold}")

    print("[*] Building SAM3 image model (text-prompted, no EVF) ...")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        checkpoint_path=sam_ckpt,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    print("[*] SAM3 model ready.")

    processor = Sam3Processor(
        model=model,
        resolution=config.get("img_size", 1008),
        device=device,
        confidence_threshold=args.confidence_threshold,
    )

    thresholds = config.get("eval_thresholds", [0.5, 0.6, 0.7, 0.8, 0.9])

    summary = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        ds_key = f"{split}_dataset"
        ds_config = config.get(ds_key)
        if ds_config is None:
            print(f"[!] {ds_key} missing in config; skip.")
            continue

        # Build the raw dataset (skip wrapper — we want raw img_path / mask_path).
        dataset = datasets.make(ds_config["dataset"])
        print(f"\n===== Evaluating split={split}, num samples={len(dataset)} =====")

        results = evaluate_split(
            processor,
            dataset,
            thresholds,
            mask_threshold=args.mask_threshold,
            max_samples=args.max_samples,
        )
        summary[split] = results

        print(f"\n--- Pure SAM3 baseline on {split} (n={results['_n']}, empty_pred={results['_empty_pred']}) ---")
        for k in [f"Pr@{t}" for t in thresholds] + ["mIoU", "oIoU"]:
            print(f"  {k:<8}: {results[k]:.4f}")

    print("\n========== Final Summary ==========")
    for split, results in summary.items():
        line = " | ".join([f"{k}={results[k]:.4f}" for k in [f"Pr@{t}" for t in thresholds] + ["mIoU", "oIoU"]])
        print(f"  [{split}] {line}  (n={results['_n']}, empty={results['_empty_pred']})")


if __name__ == "__main__":
    main()
