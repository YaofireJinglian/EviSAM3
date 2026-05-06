"""
Test/Evaluation script for EviSAM3 on RRSIS-D dataset.

Usage:
  torchrun --nproc_per_node=1 test_evf.py \
    --config configs/rrsisd-evf-sam3.yaml \
    --checkpoint save/_rrsisd-evf-sam3/model_epoch_best.pth \
    --split test

Evaluates:
  - IoU at thresholds [0.5, 0.6, 0.7, 0.8, 0.9]
  - Mean IoU (mIoU)
  - Overall IoU (oIoU)
"""

import argparse
import os
import sys
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import datasets
import models

# ===== Distributed Init =====
torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


class BEiT3Tokenizer:
    """SentencePiece tokenizer for BEiT-3."""
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
        ids = self.sp.EncodeAsIds(text)
        ids = ids[:self.max_length]
        attention_mask = [1] * len(ids) + [0] * (self.max_length - len(ids))
        ids = ids + [0] * (self.max_length - len(ids))
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
        }


def evf_collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    imgs_evf = torch.stack([b[1] for b in batch], dim=0)
    targets = torch.stack([b[2] for b in batch], dim=0)
    captions = [b[3] for b in batch]
    metas = [b[4] for b in batch]
    return imgs, imgs_evf, targets, captions, metas


def evaluate(loader, model, tokenizer, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
    model.eval()

    # Per-threshold accumulators
    cum_I = {t: 0.0 for t in thresholds}  # cumulative intersection
    cum_U = {t: 0.0 for t in thresholds}  # cumulative union
    per_sample_iou = {t: [] for t in thresholds}
    total_samples = 0

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='test')
    else:
        pbar = None

    with torch.no_grad():
        for batch in loader:
            imgs, imgs_evf, targets, captions, metas = batch
            imgs = imgs.cuda()
            imgs_evf = imgs_evf.cuda()
            targets = targets.cuda().float()

            # Handle eval captions (may be list of lists)
            all_captions = []
            for cap in captions:
                if isinstance(cap, list):
                    all_captions.append(cap[0])  # use first caption
                else:
                    all_captions.append(cap)

            offsets = list(range(len(all_captions) + 1))

            tok_results = tokenizer(all_captions)
            input_ids = torch.stack([t['input_ids'] for t in tok_results]).cuda()
            attention_masks = torch.stack([t['attention_mask'] for t in tok_results]).cuda()
            offset_tensor = torch.tensor(offsets, dtype=torch.long).cuda()

            if hasattr(model, 'module'):
                pred_masks = model.module(
                    imgs, imgs_evf, input_ids, attention_masks,
                    offset=offset_tensor, gt_mask=None
                )
            else:
                pred_masks = model(
                    imgs, imgs_evf, input_ids, attention_masks,
                    offset=offset_tensor, gt_mask=None
                )

            if isinstance(pred_masks, tuple):
                pred_masks = pred_masks[0]

            pred_sigmoid = pred_masks.sigmoid()
            gt = targets.unsqueeze(1) if targets.dim() == 3 else targets

            for t in thresholds:
                pred_binary = (pred_sigmoid > t).float()
                intersection = (pred_binary * gt).sum(dim=(-2, -1)).sum(dim=-1)
                union = pred_binary.sum(dim=(-2, -1)).sum(dim=-1) + gt.sum(dim=(-2, -1)).sum(dim=-1) - intersection
                iou = intersection / (union + 1e-8)

                cum_I[t] += intersection.sum().item()
                cum_U[t] += union.sum().item()
                per_sample_iou[t].extend(iou.cpu().tolist())

            total_samples += imgs.shape[0]

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # All-reduce for DDP
    results = {}
    for t in thresholds:
        I_tensor = torch.tensor(cum_I[t]).cuda()
        U_tensor = torch.tensor(cum_U[t]).cuda()
        dist.all_reduce(I_tensor)
        dist.all_reduce(U_tensor)

        oIoU = (I_tensor / (U_tensor + 1e-8)).item()  # overall IoU

        # Gather per-sample IoUs
        local_ious = torch.tensor(per_sample_iou[t]).cuda()
        gathered = [torch.zeros_like(local_ious) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local_ious)
        all_ious = torch.cat(gathered).cpu().numpy()

        mIoU = all_ious.mean()

        # Precision at threshold
        precision = (all_ious > t).mean()

        results[f'P@{t}'] = precision
        results[f'mIoU@{t}'] = mIoU
        results[f'oIoU@{t}'] = oIoU

    # Compute combined metrics
    results['mIoU'] = np.mean([results[f'mIoU@{t}'] for t in thresholds])
    results['oIoU'] = np.mean([results[f'oIoU@{t}'] for t in thresholds])

    return results


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Tokenizer
    spm_path = config.get('beit3_spm', '${CKPT_ROOT}/beit3.spm')
    tokenizer = BEiT3Tokenizer(spm_path, max_length=64)

    # Dataset
    split = args.split
    if split == 'test':
        ds_config = config.get('test_dataset', config.get('val_dataset'))
    else:
        ds_config = config.get('val_dataset')

    dataset = datasets.make(ds_config['dataset'])
    dataset = datasets.make(ds_config['wrapper'], args={'dataset': dataset})

    if local_rank == 0:
        print(f'{split} dataset: size={len(dataset)}')

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=ds_config.get('batch_size', 1),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=sampler,
        collate_fn=evf_collate_fn,
    )

    # Model
    model = models.make(config['model']).cuda()

    # Load checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        if local_rank == 0:
            print(f'Loaded checkpoint: {args.checkpoint}')
    else:
        if local_rank == 0:
            print('WARNING: No checkpoint loaded!')

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )

    thresholds = config.get('eval_thresholds', [0.5, 0.6, 0.7, 0.8, 0.9])
    results = evaluate(loader, model, tokenizer, thresholds)

    if local_rank == 0:
        print('\n' + '=' * 60)
        print(f'Evaluation Results on {split} split:')
        print('=' * 60)
        for k, v in sorted(results.items()):
            print(f'  {k}: {v:.4f}')
        print('=' * 60)
        print(f'  mIoU: {results["mIoU"]:.4f}')
        print(f'  oIoU: {results["oIoU"]:.4f}')
        print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/rrsisd-evf-sam3.yaml')
    parser.add_argument('--checkpoint', default=None, help='Path to model checkpoint')
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    main(args)
