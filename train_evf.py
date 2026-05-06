"""
Training script for EviSAM3: Referring Remote Sensing Image Segmentation
on SAM3 backbone with EviSAM innovations (EvidenceRectifiedAdapter + MoE).

Usage:
  torchrun --nproc_per_node=N train_evf.py --config configs/rrsisd-evf-sam3.yaml
  torchrun --nproc_per_node=N train_evf.py --config configs/rrsisd-evf-sam3.yaml --resume ${LOG_ROOT}/model_epoch_last.pth

Metrics (RRSIS-D standard):
  Pr@0.5 ~ Pr@0.9, mIoU, oIoU
"""

import argparse
import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
from statistics import mean

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models

# ===== Distributed Init =====
torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


# ===== Tokenizer =====
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


# ===== Collate Function =====
def evf_collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    imgs_evf = torch.stack([b[1] for b in batch], dim=0)
    targets = torch.stack([b[2] for b in batch], dim=0)
    captions = [b[3] for b in batch]
    metas = [b[4] for b in batch]
    return imgs, imgs_evf, targets, captions, metas


# ===== Data Loader =====
def make_data_loader(spec, tag='', tokenizer=None):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    if local_rank == 0:
        print(f'{tag} dataset: size={len(dataset)}')

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=sampler,
        drop_last=(tag == 'train'),
        collate_fn=evf_collate_fn,
    )
    return loader


# ===== Evaluation (RRSIS-D standard: Pr@t, mIoU, oIoU) =====
def evaluate(loader, model, tokenizer, eval_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Standard RRSIS-D metrics:
      Pr@t  : fraction of samples whose per-sample IoU > t
      mIoU  : mean of per-sample IoU
      oIoU  : global intersection / global union
    """
    model.eval()

    pr_correct = {t: 0 for t in eval_thresholds}
    total_inter = 0.0
    total_union = 0.0
    sum_iou = 0.0
    cnt = 0

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    with torch.no_grad():
        for batch in loader:
            imgs, imgs_evf, targets, captions, metas = batch
            imgs = imgs.cuda()
            imgs_evf = imgs_evf.cuda()
            targets = targets.cuda().float()

            flat_captions = []
            offsets = [0]
            for cap in captions:
                if isinstance(cap, list):
                    flat_captions.append(cap[0])
                else:
                    flat_captions.append(cap)
                offsets.append(offsets[-1] + 1)

            tok_results = tokenizer(flat_captions)
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

            pred_binary = (pred_masks.sigmoid() > 0.5).float()
            gt = targets.unsqueeze(1) if targets.dim() == 3 else targets

            inter = (pred_binary * gt).sum(dim=(-2, -1)).squeeze(1)
            union = (pred_binary.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1)) - (pred_binary * gt).sum(dim=(-2, -1))).squeeze(1)
            per_iou = inter / (union + 1e-8)

            for t in eval_thresholds:
                pr_correct[t] += (per_iou > t).sum().item()
            total_inter += inter.sum().item()
            total_union += union.sum().item()
            sum_iou += per_iou.sum().item()
            cnt += per_iou.shape[0]

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # All-reduce across GPUs
    vals = [float(pr_correct[t]) for t in eval_thresholds] + [total_inter, total_union, sum_iou, float(cnt)]
    vals_t = torch.tensor(vals, device='cuda')
    dist.all_reduce(vals_t)
    vals = vals_t.tolist()
    n = max(vals[-1], 1)

    results = {}
    for i, t in enumerate(eval_thresholds):
        results[f'Pr@{t}'] = vals[i] / n
    results['mIoU'] = vals[-2] / n
    results['oIoU'] = vals[-4] / max(vals[-3], 1e-8)

    model.train()
    return results


# ===== Training =====
def train_one_epoch(loader, model, optimizer, tokenizer, epoch):
    model.train()
    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc=f'train epoch {epoch}')
    else:
        pbar = None

    loss_list = []
    for batch in loader:
        imgs, imgs_evf, targets, captions, metas = batch
        imgs = imgs.cuda()
        imgs_evf = imgs_evf.cuda()
        targets = targets.cuda().float()

        flat_captions = []
        offsets = [0]
        for cap in captions:
            if isinstance(cap, list):
                cap = cap[0]
            flat_captions.append(cap)
            offsets.append(offsets[-1] + 1)

        tok_results = tokenizer(flat_captions)
        input_ids = torch.stack([t['input_ids'] for t in tok_results]).cuda()
        attention_masks = torch.stack([t['attention_mask'] for t in tok_results]).cuda()
        offset_tensor = torch.tensor(offsets, dtype=torch.long).cuda()

        optimizer.zero_grad()

        pred_masks, loss = model(
            imgs, imgs_evf, input_ids, attention_masks,
            offset=offset_tensor, gt_mask=targets
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Gather loss
        batch_loss = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, loss.detach())
        loss_list.extend([l.item() for l in batch_loss])

        if pbar is not None:
            pbar.set_postfix(loss=f'{loss.item():.4f}')
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return mean(loss_list)


# ===== Save =====
def save_checkpoint(model, optimizer, scheduler, epoch, best_miou, save_path, name):
    state = {
        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_miou': best_miou,
    }
    torch.save(state, os.path.join(save_path, f'model_epoch_{name}.pth'))


# ===== Main =====
def main(config, save_path, args):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # Tokenizer
    spm_path = config.get('beit3_spm', '${CKPT_ROOT}/beit3.spm')
    tokenizer = BEiT3Tokenizer(spm_path, max_length=64)

    # Data
    train_loader = make_data_loader(config.get('train_dataset'), tag='train', tokenizer=tokenizer)
    val_loader = make_data_loader(config.get('val_dataset'), tag='val', tokenizer=tokenizer)
    test_loader = make_data_loader(config.get('test_dataset'), tag='test', tokenizer=tokenizer)

    # Model
    model = models.make(config['model']).cuda()

    # Load SAM3 checkpoint
    ckpt_path = config.get('sam_checkpoint')
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        ref_state_dict = model.state_dict()
        new_state_dict = {}

        for k, v in ckpt.items():
            if k.startswith("detector.backbone."):
                new_k = k.replace("detector.backbone.", "image_encoder.")
            elif "mask_decoder" in k:
                suffix = k.split("mask_decoder.")[-1]
                new_k = f"mask_decoder.{suffix}"
            elif "pe_layer" in k:
                suffix = k.split("pe_layer.")[-1]
                new_k = f"pe_layer.{suffix}"
            elif "no_mask_embed" in k:
                new_k = "no_mask_embed.weight"
            else:
                new_k = k

            if new_k in ref_state_dict:
                ref_shape = ref_state_dict[new_k].shape
                if v.shape != ref_shape:
                    if local_rank == 0:
                        print(f"Skipping {new_k}: ckpt {v.shape} vs model {ref_shape}")
                    continue
                new_state_dict[new_k] = v

        msg = model.load_state_dict(new_state_dict, strict=False)
        if local_rank == 0:
            print(f"Loaded SAM3 checkpoint: {len(msg.missing_keys)} missing keys")

    # Freeze SAM3 backbone, unfreeze adapters + BEiT-3 + MoE
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    # Setup optimizer with different LRs
    base_lr = config['optimizer']['args']['lr']
    bifit_lr = config.get('lr_bifit', 5e-5)
    moe_lr = config.get('lr_moe', 5e-5)

    bifit_names = [
        'mlp_adapters', 'space_adapters', 'text_hidden_fcs', 'text_feat_linear',
        'mask_linear', 'mask_product', 'vmtoken', 'meo_linear', 'text2one_linear',
    ]
    moe_names = ['spec_prompt_delta']

    bifit_params = []
    moe_params = []
    base_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(n in name for n in moe_names):
            moe_params.append(param)
        elif any(n in name for n in bifit_names):
            bifit_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {'params': base_params, 'lr': base_lr},
        {'params': bifit_params, 'lr': bifit_lr},
        {'params': moe_params, 'lr': moe_lr},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.get('weight_decay', 0.01)
    )

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total params: {total_params:,}')
        print(f'Trainable params: {trainable_params:,}')

    # DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val', 1)
    epoch_save = config.get('epoch_save', 10)
    lr_scheduler = CosineAnnealingLR(optimizer, epoch_max, eta_min=config.get('lr_min', 1e-7))
    eval_thresholds = config.get('eval_thresholds', [0.5, 0.6, 0.7, 0.8, 0.9])

    best_miou = 0.0
    start_epoch = 1

    # ===== Resume from checkpoint =====
    if args.resume:
        resume_path = args.resume
        if os.path.isdir(resume_path):
            resume_path = os.path.join(resume_path, 'model_epoch_last.pth')
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location='cpu')
            model.module.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                lr_scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            best_miou = ckpt.get('best_miou', 0.0)
            if local_rank == 0:
                print(f'Resumed from {resume_path}, epoch {ckpt["epoch"]}, best_miou={best_miou:.4f}')
        else:
            if local_rank == 0:
                print(f'WARNING: resume path {resume_path} not found, training from scratch')

    for epoch in range(start_epoch, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(train_loader, model, optimizer, tokenizer, epoch)
        lr_scheduler.step()

        if local_rank == 0:
            print(f'Epoch {epoch}/{epoch_max} | Train Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')
            save_checkpoint(model, optimizer, lr_scheduler, epoch, best_miou, save_path, 'last')

            if epoch % epoch_save == 0:
                save_checkpoint(model, optimizer, lr_scheduler, epoch, best_miou, save_path, f'ep{epoch}')

        if epoch_val is not None and epoch % epoch_val == 0:
            val_results = evaluate(val_loader, model, tokenizer, eval_thresholds) if val_loader is not None else None
            test_results = evaluate(test_loader, model, tokenizer, eval_thresholds) if test_loader is not None else None

            if local_rank == 0:
                # Print header on first eval epoch
                if epoch == start_epoch or (epoch == start_epoch + epoch_val - (start_epoch % epoch_val)):
                    metric_names = list((val_results or test_results).keys())
                    header = f'{"Epoch":>8} |'
                    for m in metric_names:
                        header += f'  {m:>6} Val  {m:>6} Test |'
                    print(header)
                    print('-' * len(header))

                metric_names = list((val_results or test_results).keys())
                line = f'{epoch:>4}/{epoch_max:<3} |'
                for m in metric_names:
                    v_val = val_results[m] if val_results else 0.0
                    v_test = test_results[m] if test_results else 0.0
                    line += f'  {v_val:>10.4f}  {v_test:>10.4f} |'
                print(line)

                # Save best by val mIoU
                cur_miou = val_results['mIoU'] if val_results else (test_results['mIoU'] if test_results else 0.0)
                if cur_miou > best_miou:
                    best_miou = cur_miou
                    save_checkpoint(model, optimizer, lr_scheduler, epoch, best_miou, save_path, 'best')
                    print(f'  -> New best mIoU: {best_miou:.4f}')

        dist.barrier()

    if local_rank == 0:
        print(f'Training complete. Best mIoU: {best_miou:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/rrsisd-evf-sam3.yaml')
    parser.add_argument('--resume', default=None, help='Path to checkpoint (.pth) or save dir to resume from')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('Config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('${LOG_ROOT}', save_name)

    main(config, save_path, args=args)
