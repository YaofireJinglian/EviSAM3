"""Inference-time ablation evaluation for EvfSam3.

Modes:
  full      : load ckpt as-is.
  evi_only  : load ckpt, set model.use_moe_prompt = False (skip MoE delta).
  moe_only  : load ckpt, expects a separately trained no-Evi checkpoint (caller
              must supply --checkpoint pointing to such a ckpt). No monkey-
              patch is applied; this mode just runs the supplied ckpt.

Prints metrics in two forms:
  - human-readable (same as test_evf.py)
  - one-line JSON prefixed with "METRICS_JSON: " for shell parsing.
"""
import argparse
import json
import os
import sys

import yaml
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# Make project root importable so we can reuse test_evf.evaluate logic.
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import datasets  # noqa: E402
import models    # noqa: E402

torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


class BEiT3Tokenizer:
    def __init__(self, spm_path, max_length=64):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)
        self.max_length = max_length

    def __call__(self, text):
        if isinstance(text, list):
            return [self._encode_one(t) for t in text]
        return self._encode_one(text)

    def _encode_one(self, text):
        ids = self.sp.EncodeAsIds(text)[: self.max_length]
        attn = [1] * len(ids) + [0] * (self.max_length - len(ids))
        ids = ids + [0] * (self.max_length - len(ids))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.bool),
        }


def evf_collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    imgs_evf = torch.stack([b[1] for b in batch], dim=0)
    targets = torch.stack([b[2] for b in batch], dim=0)
    captions = [b[3] for b in batch]
    metas = [b[4] for b in batch]
    return imgs, imgs_evf, targets, captions, metas


def evaluate(loader, model, tokenizer, thresholds):
    from tqdm import tqdm
    model.eval()
    cum_I = {t: 0.0 for t in thresholds}
    cum_U = {t: 0.0 for t in thresholds}
    per_sample_iou = {t: [] for t in thresholds}
    pbar = tqdm(total=len(loader), leave=False, desc="ablation") if local_rank == 0 else None

    with torch.no_grad():
        for batch in loader:
            imgs, imgs_evf, targets, captions, metas = batch
            imgs = imgs.cuda(); imgs_evf = imgs_evf.cuda()
            targets = targets.cuda().float()
            all_caps = [c[0] if isinstance(c, list) else c for c in captions]
            offsets = list(range(len(all_caps) + 1))
            tok = tokenizer(all_caps)
            input_ids = torch.stack([t["input_ids"] for t in tok]).cuda()
            attn_masks = torch.stack([t["attention_mask"] for t in tok]).cuda()
            offset_t = torch.tensor(offsets, dtype=torch.long).cuda()

            net = model.module if hasattr(model, "module") else model
            pred = net(imgs, imgs_evf, input_ids, attn_masks,
                       offset=offset_t, gt_mask=None)
            if isinstance(pred, tuple):
                pred = pred[0]
            ps = pred.sigmoid()
            gt = targets.unsqueeze(1) if targets.dim() == 3 else targets

            for t in thresholds:
                pb = (ps > t).float()
                I = (pb * gt).sum(dim=(-2, -1)).sum(dim=-1)
                U = pb.sum(dim=(-2, -1)).sum(dim=-1) + gt.sum(dim=(-2, -1)).sum(dim=-1) - I
                iou = I / (U + 1e-8)
                cum_I[t] += I.sum().item()
                cum_U[t] += U.sum().item()
                per_sample_iou[t].extend(iou.cpu().tolist())

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    results = {}
    for t in thresholds:
        It = torch.tensor(cum_I[t]).cuda(); Ut = torch.tensor(cum_U[t]).cuda()
        dist.all_reduce(It); dist.all_reduce(Ut)
        oIoU_t = (It / (Ut + 1e-8)).item()
        local_ious = torch.tensor(per_sample_iou[t]).cuda()
        gathered = [torch.zeros_like(local_ious) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local_ious)
        all_ious = torch.cat(gathered).cpu().numpy()
        results[f"P@{t}"] = float((all_ious > t).mean())
        results[f"mIoU@{t}"] = float(all_ious.mean())
        results[f"oIoU@{t}"] = float(oIoU_t)
    results["mIoU"] = float(np.mean([results[f"mIoU@{t}"] for t in thresholds]))
    results["oIoU"] = float(np.mean([results[f"oIoU@{t}"] for t in thresholds]))
    return results


def apply_ablation(model, mode):
    """Runtime-only modification; does not touch source files."""
    net = model
    if mode == "full":
        return
    if mode == "evi_only":
        # Disable MoE delta path. forward() guards on `if self.use_moe_prompt and ...`.
        if hasattr(net, "use_moe_prompt"):
            net.use_moe_prompt = False
        return
    if mode == "moe_only":
        # No safe runtime ablation: MoE consumes Evi's `evidence`. Caller must
        # supply a ckpt that was trained without Evi. We just run as-is.
        return
    raise ValueError(f"unknown ablation mode: {mode}")


def main(args):
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spm_path = config.get("beit3_spm", "${CKPT_ROOT}/beit3.spm")
    tokenizer = BEiT3Tokenizer(spm_path, max_length=64)

    split = args.split
    if split == "test":
        ds_config = config.get("test_dataset", config.get("val_dataset"))
    else:
        ds_config = config.get("val_dataset")

    dataset = datasets.make(ds_config["dataset"])
    dataset = datasets.make(ds_config["wrapper"], args={"dataset": dataset})
    if local_rank == 0:
        print(f"[ablation/{args.mode}] {split} dataset: size={len(dataset)}")

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=ds_config.get("batch_size", 1),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=sampler,
        collate_fn=evf_collate_fn,
    )

    model = models.make(config["model"]).cuda()

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        sd = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        if local_rank == 0:
            print(f"[ablation/{args.mode}] loaded ckpt: {args.checkpoint}")
    else:
        if local_rank == 0:
            print(f"[ablation/{args.mode}] WARNING: no checkpoint loaded.")

    apply_ablation(model, args.mode)
    if local_rank == 0:
        print(f"[ablation/{args.mode}] mode applied. "
              f"use_moe_prompt={getattr(model, 'use_moe_prompt', None)}")

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank,
        find_unused_parameters=True, broadcast_buffers=False,
    )

    thresholds = config.get("eval_thresholds", [0.5, 0.6, 0.7, 0.8, 0.9])
    results = evaluate(loader, model, tokenizer, thresholds)

    if local_rank == 0:
        print("\n" + "=" * 60)
        print(f"[ablation/{args.mode}] Results on {split}:")
        print("=" * 60)
        for k, v in sorted(results.items()):
            print(f"  {k}: {v:.4f}")
        print("=" * 60)
        # JSON line for shell parsing.
        payload = {"mode": args.mode, "split": split, "metrics": results}
        print("METRICS_JSON: " + json.dumps(payload))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/rrsisd-evf-sam3.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--mode", default="full",
                   choices=["full", "evi_only", "moe_only"])
    p.add_argument("--local-rank", type=int, default=0)
    args = p.parse_args()
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    main(args)
