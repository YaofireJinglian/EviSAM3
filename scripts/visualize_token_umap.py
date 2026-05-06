"""B1: UMAP/t-SNE atlas of image-token features before vs. after the
EvidenceRectifiedAdapter, coloured by GT foreground/background. For each
sampled image we capture the last block's pre-adapter and post-adapter
token tensors via forward hooks, downsample the GT mask to the token
grid, aggregate FG/BG tokens across samples, then project both clouds
to 2-D and report silhouette scores.
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


def _downsample_mask(gt_hw, h, w):
    """gt_hw: (H, W) numpy uint8 -> (h, w) uint8 by max-pool."""
    x = torch.from_numpy(gt_hw).float().unsqueeze(0).unsqueeze(0)
    x = torch.nn.functional.adaptive_max_pool2d(x, output_size=(h, w))
    return (x.squeeze().numpy() > 0.5).astype(np.uint8)


class AdapterIO:
    def __init__(self):
        self.pre = None
        self.post = None

    def pre_hook(self, module, inputs):
        # inputs: (x_4d, text_for_adapter, evidence)
        x = inputs[0]
        # x is (B, H, W, C) per evf_sam3 contract
        self.pre = x.detach().float().cpu().numpy()

    def post_hook(self, module, inputs, outputs):
        # outputs: (x_4d, text_for_adapter, evidence)
        if isinstance(outputs, tuple):
            y = outputs[0]
        else:
            y = outputs
        self.post = y.detach().float().cpu().numpy()


def _extract_tokens(arr_bhwc, gt_hw, max_per_class):
    """arr_bhwc: (1, h, w, C). Returns (X_fg, X_bg) numpy arrays."""
    a = arr_bhwc[0]  # (h, w, C)
    h, w, C = a.shape
    m = _downsample_mask(gt_hw, h, w)
    fg_idx = np.argwhere(m > 0)
    bg_idx = np.argwhere(m == 0)
    if len(fg_idx) == 0:
        return None, None
    if len(fg_idx) > max_per_class:
        sel = np.random.choice(len(fg_idx), max_per_class, replace=False)
        fg_idx = fg_idx[sel]
    if len(bg_idx) > max_per_class:
        sel = np.random.choice(len(bg_idx), max_per_class, replace=False)
        bg_idx = bg_idx[sel]
    fg = a[fg_idx[:, 0], fg_idx[:, 1]]
    bg = a[bg_idx[:, 0], bg_idx[:, 1]]
    return fg, bg


def _silhouette(X, y, sample=2000):
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return float("nan")
    if len(X) > sample:
        idx = np.random.choice(len(X), sample, replace=False)
        X = X[idx]; y = y[idx]
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(silhouette_score(X, y, metric="euclidean", sample_size=min(sample, len(X))))
    except Exception:
        return float("nan")


def _project_2d(X, method="umap", seed=42):
    """Try UMAP -> t-SNE -> PCA fallback."""
    if method == "umap":
        try:
            import umap
            return umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                             random_state=seed).fit_transform(X), "UMAP"
        except Exception as e:
            print(f"  [umap unavailable: {e}] falling back to t-SNE")
    try:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, init="pca",
                    learning_rate="auto", random_state=seed).fit_transform(X), "t-SNE"
    except Exception as e:
        print(f"  [tsne unavailable: {e}] falling back to PCA")
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(X), "PCA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_REPO, "configs", "rrsisd-evf-sam3.yaml"))
    ap.add_argument("--evf_ckpt",
                    default="${LOG_ROOT}/_rrsisd-evf-sam3/model_epoch_ep50.pth")
    ap.add_argument("--output_dir", default="${OUTPUT_ROOT}")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--num_samples", type=int, default=60)
    ap.add_argument("--per_class_per_sample", type=int, default=120)
    ap.add_argument("--block_index", type=int, default=-1, help="space_adapter index (-1 = last)")
    ap.add_argument("--method", default="umap", choices=["umap", "tsne", "pca"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--filename", default="token_umap.png")
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

    spm_path = cfg.get("beit3_spm", "${CKPT_ROOT}/beit3.spm")
    tokenizer = BEiT3Tokenizer(spm_path, max_length=64)

    n_blocks = len(evf.space_adapters)
    bi = args.block_index if args.block_index >= 0 else n_blocks + args.block_index
    print(f"[*] hooking space_adapters[{bi}] of {n_blocks} blocks")
    io = AdapterIO()
    h_pre = evf.space_adapters[bi].register_forward_pre_hook(io.pre_hook)
    h_post = evf.space_adapters[bi].register_forward_hook(io.post_hook)

    pool = random.sample(range(len(ds)), min(args.num_samples, len(ds)))
    pre_fg, pre_bg, post_fg, post_bg = [], [], [], []

    for idx in tqdm(pool, desc="hooking"):
        try:
            img_sam3, img_evf, _t, cap, meta = ds[idx]
        except Exception as e:
            print(f"  skip {idx}: {e}"); continue
        text = cap[0] if isinstance(cap, list) else cap
        gt = _gt(meta)
        t = tokenizer(text)
        inp = t["input_ids"].unsqueeze(0).to(device)
        am = t["attention_mask"].unsqueeze(0).to(device)
        off = torch.tensor([0, 1], dtype=torch.long, device=device)
        with torch.inference_mode():
            _ = evf(img_sam3.unsqueeze(0).to(device),
                    img_evf.unsqueeze(0).to(device),
                    inp, am, offset=off, gt_mask=None)
        if io.pre is None or io.post is None:
            continue
        a_fg, a_bg = _extract_tokens(io.pre, gt, args.per_class_per_sample)
        b_fg, b_bg = _extract_tokens(io.post, gt, args.per_class_per_sample)
        if a_fg is None or b_fg is None:
            continue
        pre_fg.append(a_fg); pre_bg.append(a_bg)
        post_fg.append(b_fg); post_bg.append(b_bg)

    h_pre.remove(); h_post.remove()

    if not pre_fg:
        raise SystemExit("No tokens collected.")

    pre_fg = np.concatenate(pre_fg, axis=0).astype(np.float32)
    pre_bg = np.concatenate(pre_bg, axis=0).astype(np.float32)
    post_fg = np.concatenate(post_fg, axis=0).astype(np.float32)
    post_bg = np.concatenate(post_bg, axis=0).astype(np.float32)

    # Subsample BG to balance.
    def _balance(fg, bg):
        n = min(len(fg), len(bg))
        rs = np.random.RandomState(args.seed)
        return fg[rs.choice(len(fg), n, replace=False)], bg[rs.choice(len(bg), n, replace=False)]
    pre_fg, pre_bg = _balance(pre_fg, pre_bg)
    post_fg, post_bg = _balance(post_fg, post_bg)
    print(f"[*] pre  : FG={len(pre_fg)}  BG={len(pre_bg)}  C={pre_fg.shape[1]}")
    print(f"[*] post : FG={len(post_fg)} BG={len(post_bg)} C={post_fg.shape[1]}")

    X_pre = np.concatenate([pre_fg, pre_bg], axis=0)
    y_pre = np.concatenate([np.ones(len(pre_fg)), np.zeros(len(pre_bg))]).astype(np.int32)
    X_post = np.concatenate([post_fg, post_bg], axis=0)
    y_post = np.concatenate([np.ones(len(post_fg)), np.zeros(len(post_bg))]).astype(np.int32)

    print("[*] silhouette pre  ...")
    sil_pre = _silhouette(X_pre, y_pre)
    print(f"    silhouette (pre)  = {sil_pre:.4f}")
    print("[*] silhouette post ...")
    sil_post = _silhouette(X_post, y_post)
    print(f"    silhouette (post) = {sil_post:.4f}")

    print(f"[*] projecting pre  with {args.method} ...")
    Z_pre, name_pre = _project_2d(X_pre, args.method, args.seed)
    print(f"[*] projecting post with {args.method} ...")
    Z_post, name_post = _project_2d(X_post, args.method, args.seed)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=False, sharey=False)
    for ax, Z, y, sil, tag in [
        (axes[0], Z_pre, y_pre, sil_pre, f"Pre-Evi block {bi}"),
        (axes[1], Z_post, y_post, sil_post, f"Post-Evi block {bi}"),
    ]:
        ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=4, c="#888", alpha=0.45, label="BG")
        ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=4, c="#d62728", alpha=0.6, label="FG")
        ax.set_title(f"{tag}\nsilhouette = {sil:.3f}", fontsize=12)
        ax.set_xlabel(f"{name_pre} dim 1"); ax.set_ylabel(f"{name_pre} dim 2")
        ax.legend(loc="best", fontsize=9, markerscale=2)
        ax.grid(alpha=0.2)

    fig.suptitle(f"Image-token geometry across the EvidenceRectifiedAdapter "
                 f"({len(pool)} val samples, block {bi})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(args.output_dir, args.filename)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] saved -> {save_path}")

    summary = os.path.join(args.output_dir, "token_umap_summary.txt")
    with open(summary, "w") as f:
        f.write(f"block_index={bi}\nnum_samples={len(pool)}\n"
                f"pre_fg={len(pre_fg)} pre_bg={len(pre_bg)}\n"
                f"post_fg={len(post_fg)} post_bg={len(post_bg)}\n"
                f"silhouette_pre={sil_pre:.6f}\nsilhouette_post={sil_post:.6f}\n"
                f"delta_silhouette={sil_post - sil_pre:+.6f}\n"
                f"projection={name_pre}\n")
    print(f"[*] wrote {summary}")


if __name__ == "__main__":
    main()
