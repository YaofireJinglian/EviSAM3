"""C1: Text-token saliency on EvfSam3. For each sampled (image, prompt)
pair we compute the gradient of a foreground-targeted mask objective with
respect to the BEiT-3 text-token embeddings, then aggregate to a per-token
saliency map by Saliency_l = ||grad_l * emb_l||_2. The figure renders, for
each sample, the per-token saliency as a horizontal coloured-bar strip
above the rendered prompt, plus a thumbnail of the predicted mask.
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
import matplotlib.patches as mpatches

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS_DIR)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import datasets  # noqa: E402
import models   # noqa: E402


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
        }, ids[: sum(attn)]

    def decode_ids(self, id_list):
        # decode each id individually to recover sub-word piece text
        pieces = []
        for i in id_list:
            try:
                pieces.append(self.sp.IdToPiece(int(i)))
            except Exception:
                pieces.append("?")
        return pieces


def _gt(meta):
    arr = np.array(Image.open(meta["mask_path"]).convert("L"))
    return (arr > 127).astype(np.uint8)


def _resize(arr, hw):
    x = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    x = torch.nn.functional.interpolate(x, size=hw, mode="bilinear", align_corners=False)
    return x.squeeze().numpy()


def _find_text_embedding(beit3):
    """Heuristic: the BEiT-3 text path has a word-embedding nn.Embedding
    whose num_embeddings matches the SentencePiece vocab (~64k). Locate
    it by name or by the largest nn.Embedding under beit3."""
    candidates = []
    for name, m in beit3.named_modules():
        if isinstance(m, torch.nn.Embedding):
            candidates.append((name, m))
    if not candidates:
        return None, None
    # prefer name containing 'text' or 'word'
    for name, m in candidates:
        ln = name.lower()
        if "text" in ln or "word" in ln or "token" in ln:
            return name, m
    # else pick the one with largest num_embeddings
    candidates.sort(key=lambda nm: -nm[1].num_embeddings)
    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_REPO, "configs", "rrsisd-evf-sam3.yaml"))
    ap.add_argument("--evf_ckpt",
                    default="${LOG_ROOT}/_rrsisd-evf-sam3/model_epoch_ep50.pth")
    ap.add_argument("--output_dir", default="${OUTPUT_ROOT}")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--num_samples", type=int, default=6)
    ap.add_argument("--pool_size", type=int, default=120)
    ap.add_argument("--min_iou", type=float, default=0.5,
                    help="select among samples on which Full reaches at least this IoU")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--filename", default="text_saliency.png")
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

    print("[*] Building EvfSam3 ...")
    evf = models.make(cfg["model"]).cuda()
    ck = torch.load(args.evf_ckpt, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    miss = evf.load_state_dict(sd, strict=False)
    print(f"[*] loaded ({len(miss.missing_keys)} miss, {len(miss.unexpected_keys)} unexpected)")
    evf.eval()
    for p in evf.parameters():
        p.requires_grad_(False)

    spm_path = cfg.get("beit3_spm", "${CKPT_ROOT}/beit3.spm")
    tokenizer = BEiT3Tokenizer(spm_path, max_length=64)

    name_emb, mod_emb = _find_text_embedding(evf.mm_extractor.beit3)
    if mod_emb is None:
        raise SystemExit("Could not locate a text embedding module in beit3.")
    print(f"[*] hooking text embedding at: {name_emb} "
          f"(num_embeddings={mod_emb.num_embeddings}, dim={mod_emb.embedding_dim})")

    captured = {}

    def emb_hook(module, inputs, outputs):
        ids = inputs[0]
        # Heuristic: only intercept the *text* embedding call, not e.g.
        # any vision-side nn.Embedding. We require last-dim length to be
        # the text sequence length (max_length).
        if ids.dim() != 2 or ids.shape[-1] != tokenizer.max_length:
            return
        new = outputs.detach().clone().requires_grad_(True)
        new.retain_grad()
        captured["emb"] = new
        return new

    h = mod_emb.register_forward_hook(emb_hook)

    pool = random.sample(range(len(ds)), min(args.pool_size, len(ds)))
    selected = []

    for idx in tqdm(pool, desc="scanning"):
        try:
            img_sam3, img_evf, _t, cap, meta = ds[idx]
        except Exception as e:
            print(f"  skip {idx}: {e}"); continue
        text = cap[0] if isinstance(cap, list) else cap
        gt = _gt(meta)
        H, W = gt.shape

        t, ids_keep = tokenizer(text)
        inp = t["input_ids"].unsqueeze(0).to(device)
        am = t["attention_mask"].unsqueeze(0).to(device)
        off = torch.tensor([0, 1], dtype=torch.long, device=device)

        captured.clear()
        with torch.enable_grad():
            out = evf(img_sam3.unsqueeze(0).to(device),
                      img_evf.unsqueeze(0).to(device),
                      inp, am, offset=off, gt_mask=None)
            if isinstance(out, tuple):
                out = out[0]
            sig = out.sigmoid().squeeze()

            # IoU(Full) for selection (no grad needed for this metric).
            with torch.no_grad():
                pred_low = sig.detach()
                pred_low_np = pred_low.cpu().numpy()
            pred_full_np = _resize(pred_low_np, (H, W))
            pred_bin = (pred_full_np > 0.5).astype(np.uint8)
            inter = np.logical_and(pred_bin, gt).sum()
            union = np.logical_or(pred_bin, gt).sum()
            iou_full = float(inter) / (float(union) + 1e-8) if union > 0 else 0.0
            if iou_full < args.min_iou:
                continue

            # Saliency target: encourage FG sigmoid -> 1.
            gt_low = torch.from_numpy(_resize(gt.astype(np.float32), sig.shape)).to(device)
            target = (sig * gt_low).sum() - (sig * (1.0 - gt_low)).sum()

            evf.zero_grad(set_to_none=True)
            target.backward()

        emb = captured.get("emb")
        if emb is None or emb.grad is None:
            continue
        # Saliency: per-token L2 norm of (emb * grad), keep only valid tokens.
        sal = (emb * emb.grad).norm(dim=-1).squeeze(0).detach().cpu().numpy()
        L = len(ids_keep)
        sal = sal[:L]
        pieces = tokenizer.decode_ids(ids_keep)

        selected.append({
            "idx": idx, "text": text, "iou_full": iou_full,
            "pieces": pieces, "saliency": sal,
            "pred_low": pred_low_np, "gt": gt,
            "img_path": meta["img_path"],
        })
        if len(selected) >= args.num_samples:
            break

    h.remove()

    if not selected:
        raise SystemExit("No qualifying samples (try lowering --min_iou or "
                         "raising --pool_size).")

    # ---- Plot ----
    n = len(selected)
    fig = plt.figure(figsize=(14, 1.4 * n + 1.2))
    cmap = plt.get_cmap("YlOrRd")

    grid = fig.add_gridspec(n, 6, width_ratios=[1.2, 1.2, 1.2, 4.0, 4.0, 0.05],
                            hspace=0.55, wspace=0.10)

    for r, s in enumerate(selected):
        # Image
        ax0 = fig.add_subplot(grid[r, 0])
        pil = Image.open(s["img_path"]).convert("RGB")
        ax0.imshow(pil); ax0.axis("off")
        if r == 0:
            ax0.set_title("input", fontsize=10)

        # GT
        ax1 = fig.add_subplot(grid[r, 1])
        ax1.imshow(pil); ax1.imshow(s["gt"], alpha=0.45, cmap="Greens")
        ax1.axis("off")
        if r == 0:
            ax1.set_title("GT", fontsize=10)

        # Pred (Full)
        ax2 = fig.add_subplot(grid[r, 2])
        H, W = s["gt"].shape
        pred_hw = _resize(s["pred_low"], (H, W))
        ax2.imshow(pil); ax2.imshow(pred_hw, alpha=0.45, cmap="Blues")
        ax2.axis("off")
        if r == 0:
            ax2.set_title("Full pred", fontsize=10)
        ax2.text(0.02, 0.96, f"IoU={s['iou_full']:.2f}", transform=ax2.transAxes,
                 fontsize=8, color="w", va="top",
                 bbox=dict(facecolor="black", alpha=0.55, pad=1, edgecolor="none"))

        # Saliency strip + tokens
        ax3 = fig.add_subplot(grid[r, 3:5])
        sal = s["saliency"].astype(np.float32)
        if sal.max() > 0:
            sal_n = sal / sal.max()
        else:
            sal_n = sal
        L = len(sal_n)
        for k in range(L):
            ax3.add_patch(mpatches.Rectangle((k, 0.6), 1, 0.4,
                                             facecolor=cmap(sal_n[k]),
                                             edgecolor="white", linewidth=0.5))
            piece = s["pieces"][k].lstrip("\u2581")
            if not piece:
                piece = "_"
            ax3.text(k + 0.5, 0.30, piece, ha="center", va="center",
                     fontsize=8, rotation=0)
        ax3.set_xlim(0, max(L, 1)); ax3.set_ylim(0, 1.0)
        ax3.set_xticks([]); ax3.set_yticks([])
        for spine in ax3.spines.values():
            spine.set_visible(False)
        if r == 0:
            ax3.set_title("text-token saliency  $\\|\\nabla_{\\mathrm{emb}}\\mathcal{L}\\odot \\mathrm{emb}\\|_2$",
                          fontsize=10)
        ax3.text(-0.5, 0.8, f"#{r + 1}", fontsize=9, ha="right", va="center")

    # Colourbar legend
    ax_cb = fig.add_subplot(grid[:, 5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("normalised saliency", fontsize=9)

    fig.suptitle(f"Text-Token Saliency on EvfSam3 (val, n={n})  -- "
                 f"gradient $\\times$ input attribution at the BEiT-3 word "
                 f"embedding", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(args.output_dir, args.filename)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] saved -> {save_path}")

    # CSV
    csv_path = os.path.join(args.output_dir, "text_saliency_per_sample.csv")
    with open(csv_path, "w") as f:
        f.write("idx,iou_full,piece,saliency,token_pos\n")
        for s in selected:
            for k, (p, v) in enumerate(zip(s["pieces"], s["saliency"])):
                p_safe = p.replace(",", "_").replace('"', "'")
                f.write(f'{s["idx"]},{s["iou_full"]:.4f},"{p_safe}",{v:.6f},{k}\n')
    print(f"[*] wrote {csv_path}")


if __name__ == "__main__":
    main()
