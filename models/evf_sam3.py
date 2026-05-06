"""
EviSAM3 Model: EviSAM innovations ported to SAM3 backbone.

Architecture:
  1. BEiT-3 (Large) multimodal encoder -> visual features + text features + mask token
  2. SAM3 ViT backbone with EvidenceRectifiedAdapter in each block
  3. MLP_Adapter after each ViT block
  4. EvidenceDrivenExpertPrompting (MoE) for specialist delta
  5. SAM3 MaskDecoder for final mask prediction

Data flow:
  images (1008x1008) -> SAM3 backbone (with adapters + text/evidence)
  images_evf (224x224) + captions -> BEiT-3 -> visual_feat, text_feat, mask_token
  -> VL fusion -> mask prompt -> SAM3 decoder -> pred_mask (1008x1008)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from einops import rearrange

from models import register
from .sam3.build_sam3 import build_sam3_image_encoder_only
from .sam3.sam.mask_decoder import MaskDecoder
from .sam3.sam.transformer import TwoWayTransformer

# EviSAM innovation modules - ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from lib.evf_modules import (
    Adapter,
    EvidenceRectifiedAdapter,
    EvidenceDrivenExpertPrompting,
    VLFusionTokenModule,
    Identity3,
)
from lib.unilm.beit3.modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config


def dice_loss(inputs, targets, num_masks, scale=1000, eps=1e-6):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(inputs, targets, num_masks):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1).mean(1).sum() / (num_masks + 1e-8)
    return loss


@register('evf-sam3')
class EvfSam3Model(nn.Module):
    """
    EviSAM innovations on SAM3 backbone for referring remote sensing image segmentation.
    """

    def __init__(
        self,
        inp_size=1008,
        encoder_mode=None,
        loss='iou',
        # BEiT-3 config
        mm_extractor_scale='large',
        hidden_size=1024,
        out_dim=256,
        # Pretrained paths
        encoder_pretrained=None,
        # MoE config
        use_moe_prompt=True,
        moe_balance_weight=0.01,
        moe_num_extra_tokens=1,
        moe_gate_init=-6.0,
        moe_gamma_init=1e-3,
        moe_temperature=1.5,
        # SAM3 mask decoder
        prompt_embed_dim=256,
        sam_mask_decoder_extra_args=None,
        use_high_res_features_in_sam=True,
        directly_add_no_mem_embed=True,
        iou_prediction_use_sigmoid=False,
        pred_obj_scores=True,
        pred_obj_scores_mlp=False,
        use_multimask_token_for_obj_ptr=True,
        **kwargs,
    ):
        super().__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.use_moe_prompt = use_moe_prompt
        self.moe_balance_weight = moe_balance_weight
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.directly_add_no_mem_embed = directly_add_no_mem_embed

        # ===== SAM3 Image Encoder =====
        self.image_encoder = build_sam3_image_encoder_only()
        self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

        # Freeze SAM3 backbone, keep prompt_generator trainable
        for name, para in self.image_encoder.named_parameters():
            if "prompt_generator" not in name:
                para.requires_grad_(False)
            else:
                para.requires_grad_(True)

        # ===== Inject EvidenceRectifiedAdapter + MLP_Adapter into ViT blocks =====
        vit_backbone = self.image_encoder.vision_backbone.trunk
        self.num_blocks = len(vit_backbone.blocks)
        embed_dim = 1024  # SAM3 ViT embed_dim

        # Create adapters for each block
        self.mlp_adapters = nn.ModuleList()
        self.space_adapters = nn.ModuleList()
        self.adapter_scale = 0.5

        for i in range(self.num_blocks):
            self.mlp_adapters.append(Adapter(embed_dim, skip_connect=False))
            self.space_adapters.append(EvidenceRectifiedAdapter(embed_dim))

        # ===== BEiT-3 Multimodal Encoder =====
        if mm_extractor_scale == 'base':
            beit_config = _get_base_config()
        elif mm_extractor_scale == 'large':
            beit_config = _get_large_config()
        else:
            raise ValueError(f"mm_extractor_scale must be 'base' or 'large', got {mm_extractor_scale}")

        self.mm_extractor = BEiT3Wrapper(beit_config)
        if encoder_pretrained is not None and os.path.exists(encoder_pretrained):
            beit_state_dict = torch.load(encoder_pretrained, map_location='cpu')["model"]
            self.mm_extractor.load_state_dict(beit_state_dict, strict=False)

        for param in self.mm_extractor.parameters():
            param.requires_grad = True

        # ===== Projection layers =====
        in_dim = hidden_size
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        self.text_feat_linear = nn.Linear(1024, 256)
        for param in self.text_feat_linear.parameters():
            param.requires_grad = True

        # ===== Shared path (baseline Mask Prompt Generator) =====
        self.mask_linear = nn.Linear(1024, 1024)
        self.mask_product = nn.Linear(1024, 1)

        self.vmtoken = VLFusionTokenModule(d_model=1024, nhead=8)

        self.meo_linear = nn.Linear(1024, 256)
        self.text2one_linear = nn.Linear(256, 1)

        # Projection for mask prompt: 1 channel -> prompt_embed_dim channels
        self.mask_prompt_proj = nn.Sequential(
            nn.Conv2d(1, prompt_embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(prompt_embed_dim, prompt_embed_dim, kernel_size=1),
        )

        # ===== MoE: Evidence-Driven Expert Prompting =====
        self.spec_prompt_delta = EvidenceDrivenExpertPrompting(
            in_dim=1024,
            evidence_dim=256,
            text_dim=256,
            prompt_dim=256,
            num_extra_tokens=moe_num_extra_tokens,
            gate_init=moe_gate_init,
            gamma_init=moe_gamma_init,
            temperature=moe_temperature,
        )
        self.last_moe_balance = None

        # ===== SAM3 Mask Decoder =====
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.prompt_embed_dim = prompt_embed_dim

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=1,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=iou_prediction_use_sigmoid,
            pred_obj_scores=pred_obj_scores,
            pred_obj_scores_mlp=pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=use_multimask_token_for_obj_ptr,
            **(sam_mask_decoder_extra_args or {}),
        )

        # ===== Positional encoding for prompt =====
        self.pe_layer = PositionEmbeddingRandom(prompt_embed_dim // 2)
        self.image_embedding_size = inp_size // 14  # patch_size=14 -> 1008/14=72
        self.no_mask_embed = nn.Embedding(1, prompt_embed_dim)

        # ===== Loss =====
        self.loss_mode = loss

    def get_dense_pe(self):
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def postprocess_masks(self, masks, orig_hw):
        masks = masks.float()
        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks

    def _forward_backbone_with_adapters(self, images, text_feat):
        """
        Forward through SAM3 backbone with EvidenceRectifiedAdapter injected after each block.

        Args:
            images: (B, 3, 1008, 1008)
            text_feat: (B, L, 256) text features from BEiT-3

        Returns:
            features dict, evidences list
        """
        vit_backbone = self.image_encoder.vision_backbone.trunk
        neck = self.image_encoder.vision_backbone

        # Patch embedding
        x = vit_backbone.patch_embed(images)
        h, w = x.shape[1], x.shape[2]

        s = 0
        if vit_backbone.retain_cls_token:
            x = torch.cat([vit_backbone.class_embedding, x.flatten(1, 2)], dim=1)
            s = 1

        if vit_backbone.pos_embed is not None:
            from models.sam3.model.vitdet import get_abs_pos
            x = x + get_abs_pos(
                vit_backbone.pos_embed,
                vit_backbone.pretrain_use_cls_token,
                (h, w),
                vit_backbone.retain_cls_token,
                tiling=vit_backbone.tile_abs_pos,
            )

        x = vit_backbone.ln_pre(x)

        # Handcrafted features from PromptGenerator
        inp = images
        handcrafted_list = vit_backbone.prompt_generator.init_handcrafted(inp)

        # Prepare text for adapter: (B, L, 256) -> (L, B, 256)
        text_for_adapter = rearrange(text_feat, 'b l c -> l b c')
        evidence = None
        evidences = []

        outputs = []
        for i, blk in enumerate(vit_backbone.blocks):
            # Stage/depth index for PromptGenerator
            depth_per_stage = vit_backbone.depth_per_stage
            if i < depth_per_stage:
                stage_idx, rel_idx = 1, i
            elif i < depth_per_stage * 2:
                stage_idx, rel_idx = 2, i - depth_per_stage
            elif i < depth_per_stage * 3:
                stage_idx, rel_idx = 3, i - depth_per_stage * 2
            else:
                stage_idx, rel_idx = 4, i - depth_per_stage * 3

            current_handcrafted = handcrafted_list[stage_idx - 1]

            if str(stage_idx) in vit_backbone.prompt_generator.tuning_stage:
                resized_handcrafted = vit_backbone._resize_handcrafted(current_handcrafted, h, w)
                prompt_tuple = vit_backbone.prompt_generator.init_prompt(x, resized_handcrafted, stage_idx)
                x = vit_backbone.prompt_generator.get_prompt(x, prompt_tuple, stage_idx, rel_idx)

            # Run ViT block
            x = blk(x)

            # ===== EviSAM Innovation: Space Adapter (EvidenceRectifiedAdapter) =====
            # Need x in (B, H, W, C) format
            if x.ndim == 3:
                x_4d = x[:, s:].reshape(x.shape[0], h, w, -1)
            else:
                x_4d = x

            x_4d, text_for_adapter, evidence = self.space_adapters[i](
                x_4d, text_for_adapter, evidence
            )

            # ===== EviSAM Innovation: MLP Adapter =====
            mlp_adapter_out = self.mlp_adapters[i](vit_backbone.blocks[i].norm2(x_4d) if hasattr(vit_backbone.blocks[i], 'norm2') else x_4d)

            # Recombine
            if x.ndim == 3:
                x_flat = x_4d.reshape(x_4d.shape[0], -1, x_4d.shape[-1])
                if s > 0:
                    x = torch.cat([x[:, :s], x_flat], dim=1)
                else:
                    x = x_flat
                # Add MLP adapter residual
                mlp_flat = mlp_adapter_out.reshape(mlp_adapter_out.shape[0], -1, mlp_adapter_out.shape[-1])
                if s > 0:
                    x[:, s:] = x[:, s:] + self.adapter_scale * mlp_flat
                else:
                    x = x + self.adapter_scale * mlp_flat
            else:
                x = x_4d + self.adapter_scale * mlp_adapter_out

            # Extract intermediate features at full attention block indices
            full_attn_ids = vit_backbone.full_attn_ids
            if (i == full_attn_ids[-1]) or (
                vit_backbone.return_interm_layers and i in full_attn_ids
            ):
                if i == full_attn_ids[-1]:
                    x_out = vit_backbone.ln_post(x) if x.ndim == 3 else vit_backbone.ln_post(x)
                else:
                    x_out = x

                feats = x_out[:, s:] if x_out.ndim == 3 else x_out
                if feats.ndim == 3:
                    feats = feats.reshape(feats.shape[0], h, w, feats.shape[-1])
                if feats.ndim == 4:
                    feats = feats.permute(0, 3, 1, 2)

                outputs.append(feats)
                evidences.append(evidence)

        # Pass through neck (SimpleFPN: all scales from last feature map)
        scalp = self.image_encoder.scalp

        # Apply neck convolutions
        sam3_features = []
        sam3_pos = []
        neck_module = self.image_encoder.vision_backbone

        # Use the last output for neck processing
        x_last = outputs[-1]  # (B, C, H, W)

        # Apply scale convolutions from neck
        for conv_i in range(len(neck_module.convs)):
            feat_out = neck_module.convs[conv_i](x_last)
            pos_out = neck_module.position_encoding(feat_out)
            sam3_features.append(feat_out)
            sam3_pos.append(pos_out)

        if scalp > 0:
            sam3_features = sam3_features[:-scalp]
            sam3_pos = sam3_pos[:-scalp]

        output_dict = {
            "vision_features": sam3_features[-1],
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
        }

        return output_dict, evidences

    def forward(
        self,
        images,           # (B, 3, 1008, 1008)
        images_evf,       # (B, 3, 224, 224)
        input_ids,        # (B, L) token IDs
        attention_masks,  # (B, L) attention masks
        offset=None,      # offset for multi-caption alignment
        gt_mask=None,     # (B, 1008, 1008) ground truth
        inference=False,
    ):
        batch_size = images.shape[0]
        device = images.device

        # ===== Handle offset for multi-caption per image =====
        if offset is not None:
            images_evf_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_evf_i = (
                    images_evf[i].unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_evf_list.append(images_evf_i)
            images_evf = torch.cat(images_evf_list, dim=0)

        # ===== BEiT-3 multimodal encoding =====
        output = self.mm_extractor.beit3(
            visual_tokens=images_evf,
            textual_tokens=input_ids,
            text_padding_position=~attention_masks
        )
        _, len_all, _ = output["encoder_out"].shape
        _, text_len = input_ids.shape
        visual_len = len_all - text_len - 1

        visual_feat = output["encoder_out"][:, 1:visual_len + 1, ...]   # (B, V, 1024)
        text_feat_raw = output["encoder_out"][:, len_all - text_len:len_all, ...]  # (B, L, 1024)
        mask_token = output["encoder_out"][:, :1, ...]                  # (B, 1, 1024)

        # ===== Shared expert: VL fusion on mask_token =====
        visual_feat_seq = rearrange(visual_feat, 'b hw c -> hw b c')
        mask_token_seq = rearrange(mask_token, 'b m c -> m b c')

        mask_token_seq = self.vmtoken(
            tgt=mask_token_seq,
            memory=visual_feat_seq,
        )

        mask_token = rearrange(mask_token_seq, 'm b c -> b m c')
        visual_feat = rearrange(visual_feat_seq, 'hw b c -> b hw c')
        mask_token_expanded = self.mask_linear(mask_token).unsqueeze(1)  # (B,1,1,1024)

        # Base logits from shared path
        h_evf = int(visual_feat.shape[1] ** 0.5)
        visual_feat_2d = rearrange(visual_feat, 'b (h w) c -> b h w c', h=h_evf, w=h_evf)
        base = visual_feat_2d * mask_token_expanded   # (B, 14, 14, 1024)
        base_logits = self.mask_product(base)          # (B, 14, 14, 1)

        # ===== Text embeddings =====
        feat = self.text_hidden_fcs[0](mask_token)     # (B, 1, 256)
        text_feat = self.text_feat_linear(text_feat_raw)  # (B, L, 256)
        text_cls = feat[:, 0, :]                          # (B, 256)

        dtype_out = images.dtype

        # ===== SAM3 backbone with EvidenceRectifiedAdapter =====
        backbone_out, evidences = self._forward_backbone_with_adapters(images, text_feat)

        # Collect evidence: average across layers to keep token count consistent
        evidence = None
        valid_evidences = [e for e in evidences if e is not None]
        if valid_evidences:
            evidence = torch.stack(valid_evidences, dim=0).mean(dim=0)

        # ===== MoE: specialist experts =====
        logits = base_logits
        if self.use_moe_prompt and evidence is not None:
            dlogits, dtokens, gate, w, lbal = self.spec_prompt_delta(base, evidence, text_cls)
            self.last_moe_balance = lbal

            logits = base_logits + gate.view(-1, 1, 1, 1) * self.spec_prompt_delta.gamma * dlogits
            extra_tokens = gate.view(-1, 1, 1) * self.spec_prompt_delta.gamma_tok * dtokens
            feat = torch.cat([feat, extra_tokens], dim=1)
        else:
            self.last_moe_balance = None

        # ===== Dense mask prompt =====
        mask_prompt = torch.sigmoid(logits)                         # (B, 14, 14, 1)
        mask_prompt = rearrange(mask_prompt, 'b h w c -> b c h w')  # (B, 1, 14, 14)

        # ===== Prepare SAM3 features for decoder =====
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]

        if self.use_high_res_features_in_sam:
            feature_maps[0] = self.mask_decoder.conv_s0(feature_maps[0])
            feature_maps[1] = self.mask_decoder.conv_s1(feature_maps[1])

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]

        if self.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + torch.zeros(1, 1, 256, device=device)

        feats = [
            feat_v.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat_v, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]

        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        high_res_features = _features["high_res_feats"]
        image_embed = _features["image_embed"]
        H_feat, W_feat = image_embed.shape[-2], image_embed.shape[-1]

        # ===== Sparse embeddings from text =====
        # Use feat (B, 1+N, 256) as sparse prompt tokens
        sparse_embeddings = feat.to(dtype_out)

        # ===== Dense embeddings from mask prompt =====
        # Resize mask_prompt to match feature map
        if mask_prompt.shape[-2:] != (H_feat, W_feat):
            mask_prompt_resized = F.interpolate(
                mask_prompt.float(),
                size=(H_feat, W_feat),
                mode="bilinear",
                align_corners=False,
            ).to(dtype_out)
        else:
            mask_prompt_resized = mask_prompt.to(dtype_out)

        # Use no_mask_embed as base dense embedding, add projected mask prompt
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            batch_size, -1, H_feat, W_feat
        )
        # Project mask prompt from 1 channel to 256 channels before adding
        mask_prompt_projected = self.mask_prompt_proj(mask_prompt_resized)
        dense_embeddings = dense_embeddings + mask_prompt_projected

        # ===== Positional encoding =====
        image_pe = self.get_dense_pe()
        if image_pe.shape[-2] != H_feat or image_pe.shape[-1] != W_feat:
            image_pe = F.interpolate(
                image_pe, size=(H_feat, W_feat),
                mode='bilinear', align_corners=False
            )

        # ===== Mask Decoder =====
        low_res_masks, iou_predictions, sam_output_tokens, object_score_logits = self.mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # ===== Postprocess =====
        pred_masks = self.postprocess_masks(low_res_masks, (self.inp_size, self.inp_size))
        pred_masks = pred_masks.float()

        # ===== Loss computation =====
        if gt_mask is not None:
            gt = gt_mask.float().unsqueeze(1) if gt_mask.dim() == 3 else gt_mask.float()
            # CE + Dice loss
            ce_loss = sigmoid_ce_loss(pred_masks, gt, batch_size)
            d_loss = dice_loss(pred_masks, gt, batch_size)
            loss = 0.9 * ce_loss + 0.1 * d_loss

            return pred_masks, loss

        return pred_masks

    def infer(self, images, images_evf, input_ids, attention_masks):
        """Inference mode: returns predicted masks."""
        with torch.no_grad():
            pred_masks = self.forward(
                images, images_evf, input_ids, attention_masks,
                gt_mask=None, inference=True
            )
        return pred_masks


class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies."""
    def __init__(self, num_pos_feats=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * 3.14159265358979 * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size):
        h, w = size, size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W
