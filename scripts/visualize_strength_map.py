"""Strength Map: 2-D heatmap of IoU(Full) - IoU(SAM3) over a binning of
(object area, prompt-length) for RRSIS-D val. Scores `--num_samples`
random samples and writes the figure plus a per-sample CSV.
"""
import argparse
import os
import sys
import random

import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS_DIR)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import datasets  # noqa: E402
import models   # noqa: E402
from models.model_builder import build_sam3_image_model  # noqa: E402
from models.sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

_BPE_FALLBACKS = [
    os.path.join(_REPO, "assets", "bpe_simple_vocab_16e6.txt.gz"),
    os.path.join(_REPO, "models", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
    "/data/yz_data/sen_data/sen_env/lib/python3.10/site-packages/clip/bpe_simple_vocab_16e6.txt.gz",
]


def _resolve_bpe(p):
    if p and os.path.exists(p):
        return p
    for q in _BPE_FALLBACKS:
        if os.path.exists(q):
            return q
    raise FileNotFoundError("bpe_simple_vocab_16e6.txt.gz not found")


class BEiT3Tokenizer:
    def __init__(self, spm_path, max_length=64):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)
        self.max_length = max_length

    def __call__(self, text):
        ids = self.sp.EncodeAsIds(text)[: self.max_length]
        attn = [1] * len(ids) + [0] * (self.max_length - len(ids))
        ids = ids + [0] * (self.max_length - len(ids))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.bool),
        }


def _gt(meta):
    arr = np.array(Image.open(meta["mask_path"]).convert("L"))
    return (arr > 127).astype(np.uint8)


def _resize(arr, hw):
    x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    x = torch.nn.functional.interpolate(x, size=hw, mode="bilinear", align_corners=False)
    return x.squeeze().numpy()


def _iou(p, g):
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter) / (float(union) + 1e-8) if union > 0 else 0.0


@torch.inference_mode()
def _run_sam3(proc, pil, txt):
    s = proc.set_image(pil); s = proc.set_text_prompt(txt, s)
    ml = s.get("masks_logits")
    if ml is None or ml.numel() == 0 or ml.shape[0] == 0:
        H, W = s["original_height"], s["original_width"]
        return np.zeros((H, W), np.float32)
    return ml.squeeze(1).max(dim=0).values.cpu().numpy().astype(np.float32)


@torch.inference_mode()
def _run_evf(model, tok, img_sam3, img_evf, txt, dev):
    t = tok(txt)
    inp = t["input_ids"].unsqueeze(0).to(dev)
    am = t["attention_mask"].unsqueeze(0).to(dev)
    off = torch.tensor([0, 1], dtype=torch.long, device=dev)
    out = model(img_sam3.unsqueeze(0).to(dev), img_evf.unsqueeze(0).to(dev),
                inp, am, offset=off, gt_mask=None)
    if isinstance(out, tuple):
        out = out[0]
    return out.sigmoid().squeeze().cpu().numpy().astype(np.float32)


def _bin_idx(value, edges):
    # Returns index in [0, len(edges)-2]; values >= last edge clipped to last bin.
    i = int(np.searchsorted(edges, value, side="right") - 1)
    return max(0, min(i, len(edges) - 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_REPO, "configs", "rrsisd-evf-sam3.yaml"))
    ap.add_argument("--evf_ckpt",
                    default="${LOG_ROOT}/_rrsisd-evf-sam3/model_epoch_ep50.pth")
    ap.add_argument("--output_dir", default="${OUTPUT_ROOT}")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--num_samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mask_threshold", type=float, default=0.5)
    ap.add_argument("--filename", default="strength_map.png")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = "cuda"

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    ds = datasets.make(cfg[f"{args.split}_dataset"]["dataset"])
    print(f"[*] {args.split} size: {len(ds)}")

    bpe = _resolve_bpe(None)
    print("[*] Building SAM3 ...")
    sam3 = build_sam3_image_model(bpe_path=bpe, device=device, eval_mode=True,
                                  checkpoint_path=cfg["sam_checkpoint"], load_from_HF=False,
                                  enable_segmentation=True, enable_inst_interactivity=False,
                                  compile=False)
    proc = Sam3Processor(sam3, resolution=cfg.get("img_size", 1008),
                         device=device, confidence_threshold=0.5)

    print("[*] Building EvfSam3 ...")
    evf = models.make(cfg["model"]).cuda()
    ck = torch.load(args.evf_ckpt, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    miss = evf.load_state_dict(sd, strict=False)
    print(f"[*] loaded ({len(miss.missing_keys)} miss, {len(miss.unexpected_keys)} unexpected)")
    evf.eval()

    spm_path = cfg.get("beit3_spm", "${CKPT_ROOT}/beit3.spm")
    tokenizer = BEiT3Tokenizer(spm_path, max_length=64)

    n_total = len(ds)
    pool = random.sample(range(n_total), min(args.num_samples, n_total))

    rows = []
    pbar = tqdm(pool, desc="scoring")
    for idx in pbar:
        try:
            img_sam3, img_evf, _t, cap, meta = ds[idx]
        except Exception as e:
            print(f"  skip {idx}: {e}"); continue
        text = cap[0] if isinstance(cap, list) else cap
        n_words = len(text.split())
        pil = Image.open(meta["img_path"]).convert("RGB")
        gt = _gt(meta)
        H, W = gt.shape
        area_ratio = float(gt.sum()) / (H * W + 1e-8)

        sig_sam3 = _run_sam3(proc, pil, text)
        pred_sam3 = (sig_sam3 > args.mask_threshold).astype(np.uint8)
        iou_sam3 = _iou(pred_sam3.astype(bool), gt.astype(bool))

        sig_evf_low = _run_evf(evf, tokenizer, img_sam3, img_evf, text, device)
        sig_evf = _resize(sig_evf_low, (H, W))
        pred_evf = (sig_evf > args.mask_threshold).astype(np.uint8)
        iou_full = _iou(pred_evf.astype(bool), gt.astype(bool))

        rows.append({"idx": idx, "n_words": n_words, "area_ratio": area_ratio,
                     "iou_sam3": iou_sam3, "iou_full": iou_full,
                     "delta": iou_full - iou_sam3, "text": text,
                     "ref_id": meta.get("ref_id", "-")})
        pbar.set_postfix(d=f"{iou_full - iou_sam3:+.2f}")

    if not rows:
        raise SystemExit("No rows scored.")

    # Bin edges (object area in log10 space, prompt length linear).
    area_edges = np.array([0.0, 0.001, 0.005, 0.02, 0.10, 1.0])
    area_labels = ["<0.1%", "0.1-0.5%", "0.5-2%", "2-10%", ">10%"]
    word_edges = np.array([0, 4, 6, 8, 10, 999])
    word_labels = ["<=3", "4-5", "6-7", "8-9", ">=10"]

    H, W = len(area_edges) - 1, len(word_edges) - 1
    sum_d = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.int64)
    for r in rows:
        i = _bin_idx(r["area_ratio"], area_edges)
        j = _bin_idx(r["n_words"], word_edges)
        sum_d[i, j] += r["delta"]; cnt[i, j] += 1
    mean_d = np.where(cnt > 0, sum_d / np.maximum(cnt, 1), np.nan)

    # Mean IoU per cell (Full).
    sum_iou = np.zeros((H, W), dtype=np.float64)
    for r in rows:
        i = _bin_idx(r["area_ratio"], area_edges)
        j = _bin_idx(r["n_words"], word_edges)
        sum_iou[i, j] += r["iou_full"]
    mean_iou = np.where(cnt > 0, sum_iou / np.maximum(cnt, 1), np.nan)

    # Plot 1x2: left = delta heatmap, right = count grid.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [1.3, 1.0]})

    ax = axes[0]
    vmax = float(np.nanmax(np.abs(mean_d))) if np.any(~np.isnan(mean_d)) else 0.5
    vmax = max(vmax, 0.05)
    im = ax.imshow(mean_d, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto", origin="upper")
    ax.set_xticks(range(W)); ax.set_xticklabels(word_labels, fontsize=10)
    ax.set_yticks(range(H)); ax.set_yticklabels(area_labels, fontsize=10)
    ax.set_xlabel("prompt length (#words)", fontsize=11)
    ax.set_ylabel("object area / image area", fontsize=11)
    ax.set_title("Strength Map: $\\Delta$IoU = IoU(Full) - IoU(SAM3)", fontsize=12)
    for i in range(H):
        for j in range(W):
            if cnt[i, j] > 0:
                v = mean_d[i, j]
                ax.text(j, i, f"{v:+.2f}\nn={cnt[i, j]}",
                        ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.55 * vmax else "black")
            else:
                ax.text(j, i, "-", ha="center", va="center",
                        fontsize=10, color="#888")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("mean $\\Delta$IoU", fontsize=10)

    ax2 = axes[1]
    im2 = ax2.imshow(mean_iou, cmap="viridis", vmin=0.0, vmax=1.0,
                     aspect="auto", origin="upper")
    ax2.set_xticks(range(W)); ax2.set_xticklabels(word_labels, fontsize=10)
    ax2.set_yticks(range(H)); ax2.set_yticklabels(area_labels, fontsize=10)
    ax2.set_xlabel("prompt length (#words)", fontsize=11)
    ax2.set_ylabel("object area / image area", fontsize=11)
    ax2.set_title("Mean IoU (Full)", fontsize=12)
    for i in range(H):
        for j in range(W):
            if cnt[i, j] > 0:
                ax2.text(j, i, f"{mean_iou[i, j]:.2f}\nn={cnt[i, j]}",
                         ha="center", va="center", fontsize=9,
                         color="white" if mean_iou[i, j] < 0.5 else "black")
            else:
                ax2.text(j, i, "-", ha="center", va="center",
                         fontsize=10, color="#bbb")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.03)
    cbar2.set_label("mean IoU(Full)", fontsize=10)

    fig.suptitle(f"Strength of EvfSam3 over SAM3 baseline -- {args.split} "
                 f"(n={len(rows)} samples)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(args.output_dir, args.filename)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] saved -> {save_path}")

    # CSV with per-sample data.
    csv_path = os.path.join(args.output_dir, "strength_map_per_sample.csv")
    with open(csv_path, "w") as f:
        f.write("idx,ref_id,n_words,area_ratio,iou_sam3,iou_full,delta,caption\n")
        for r in rows:
            f.write(f'{r["idx"]},{r["ref_id"]},{r["n_words"]},{r["area_ratio"]:.6f},'
                    f'{r["iou_sam3"]:.4f},{r["iou_full"]:.4f},{r["delta"]:+.4f},'
                    f'"{r["text"]}"\n')
    print(f"[*] wrote {csv_path}")


if __name__ == "__main__":
    main()
