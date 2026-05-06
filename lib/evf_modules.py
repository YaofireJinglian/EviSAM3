"""
EviSAM Innovation Modules adapted for SAM3 backbone.

Key modules:
  - Adapter: lightweight MLP adapter (skip-connect)
  - VisionLanguageFusionModule: cross-attention VL/LV fusion
  - EvidenceRectifiedAdapter: text+evidence fusion in backbone blocks
  - ArcExpert: Adaptive Rotated Convolution expert
  - EvidenceDrivenExpertPrompting: MoE with 3 specialized experts
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from arc import AdaptiveRotatedConv2d, RountingFunction


# ============================================================
# Basic Adapter
# ============================================================
class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden)
        self.D_fc2 = nn.Linear(D_hidden, D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        return x + xs if self.skip_connect else xs


# ============================================================
# Vision-Language Fusion Module
# ============================================================
class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + tgt2
        return tgt


# ============================================================
# Identity3: pass-through for blocks without adapter
# ============================================================
class Identity3(nn.Module):
    def forward(self, x, text, evidence=None):
        return x, text, evidence


# ============================================================
# Evidence Rectified Adapter (from hieradet_v1.py)
# ============================================================
class EvidenceRectifiedAdapter(nn.Module):
    """
    Baseline VL/LV fusion + posterior evidence update + rectification.
    Adapted for SAM3's ViT backbone (D_features = 1024).

    Key properties:
      - NO dropout anywhere
      - delta-norm: LayerNorm applied to delta only
      - rectification gated with gate_init (-6 for closer-to-baseline start)
      - in-forward evidence memory (explicitly passed across blocks)
    """

    def __init__(
        self,
        D_features: int,
        mlp_ratio: float = 0.25,
        act_layer=nn.GELU,
        skip_connect: bool = True,
        evidence_dim: int = 256,
        evidence_tokens: int = 8,
        evidence_heads: int = 4,
        gate_init: float = -6.0,
        ls_init: float = 1e-3,
        mem_momentum: float = 0.6,
        vis_pool: int = 8,
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.mem_momentum = mem_momentum
        self.evidence_dim = evidence_dim
        self.vis_pool = vis_pool

        D_hidden = int(D_features * mlp_ratio)
        self.D_hidden = D_hidden

        # x -> hidden
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden)
        self.D_fc2 = nn.Linear(D_hidden, D_features)

        # text proj
        self.text_in = nn.Linear(256, D_hidden)
        self.text_out = nn.Linear(D_hidden, 256)

        # VL/LV (baseline path), no dropout
        self.vlfusion = VisionLanguageFusionModule(d_model=D_hidden, nhead=4, dropout=0.0)
        self.lvfusion = VisionLanguageFusionModule(d_model=D_hidden, nhead=4, dropout=0.0)

        # posterior evidence update
        self.evidence_queries = nn.Parameter(torch.randn(evidence_tokens, 1, evidence_dim) * 0.02)
        self.evidence_attn = nn.MultiheadAttention(
            embed_dim=evidence_dim, num_heads=evidence_heads, dropout=0.0, batch_first=False
        )

        self.txt_to_evi = nn.Linear(D_hidden, evidence_dim)
        self.vis_to_evi = nn.Linear(D_hidden, evidence_dim)

        self.evi_to_hid = nn.Linear(evidence_dim, D_hidden)

        # rectification (no dropout)
        self.ev_attn = nn.MultiheadAttention(D_hidden, num_heads=4, dropout=0.0, batch_first=False)
        self.el_attn = nn.MultiheadAttention(D_hidden, num_heads=4, dropout=0.0, batch_first=False)

        self.ev_norm = nn.LayerNorm(D_hidden)
        self.el_norm = nn.LayerNorm(D_hidden)

        # gates (strict init)
        self.gate_ev = nn.Linear(D_hidden, 1)
        self.gate_el = nn.Linear(D_hidden, 1)
        nn.init.zeros_(self.gate_ev.weight)
        nn.init.constant_(self.gate_ev.bias, gate_init)
        nn.init.zeros_(self.gate_el.weight)
        nn.init.constant_(self.gate_el.bias, gate_init)

        # LayerScale for rectification
        self.ls_ev = nn.Parameter(ls_init * torch.ones(D_hidden))
        self.ls_el = nn.Parameter(ls_init * torch.ones(D_hidden))

    def _pool_vision_tokens(self, xs_sbc, h, w):
        x2d = rearrange(xs_sbc, '(hh ww) b c -> b hh ww c', hh=h, ww=w)
        x2d = x2d.permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(x2d, output_size=(self.vis_pool, self.vis_pool))
        pooled = pooled.permute(2, 3, 0, 1).reshape(self.vis_pool * self.vis_pool, x2d.shape[0], x2d.shape[1])
        return pooled

    def _update_evidence_posterior(self, text_lbc, text_h_lbc, xs_sbc, h, w, evidence):
        B = text_lbc.shape[1]
        q = self.evidence_queries.repeat(1, B, 1)

        mem_text = text_lbc
        mem_text_h = self.txt_to_evi(text_h_lbc)

        vis_pooled = self._pool_vision_tokens(xs_sbc, h, w)
        mem_vis = self.vis_to_evi(vis_pooled)

        mem = torch.cat([mem_text, mem_text_h, mem_vis], dim=0)

        cur_evi, _ = self.evidence_attn(query=q, key=mem, value=mem)
        if evidence is None:
            return cur_evi
        m = self.mem_momentum
        return m * evidence + (1.0 - m) * cur_evi

    def forward(self, x, text, evidence=None):
        """
        x:    (B, H, W, D_features)
        text: (L, B, 256)
        evidence: (E, B, 256) or None
        return: x, text_out, evidence
        """
        dtype = x.dtype
        text = text.to(dtype)
        if evidence is not None:
            evidence = evidence.to(dtype)

        # x/text -> hidden
        xs = self.D_fc1(x)
        xs = self.act(xs)
        B, h, w, C = xs.shape
        xs = rearrange(xs, 'b hh ww c -> (hh ww) b c')

        text_init = text
        text_h = self.text_in(text)

        # baseline VL/LV fusion
        xs = self.vlfusion(tgt=xs, memory=text_h)
        text_h = self.lvfusion(tgt=text_h, memory=xs)

        # posterior evidence update
        evidence = self._update_evidence_posterior(
            text_lbc=text_init, text_h_lbc=text_h, xs_sbc=xs,
            h=h, w=w, evidence=evidence,
        )

        evi_h = self.evi_to_hid(evidence)
        evi_pool = evi_h.mean(dim=0)

        # rectification
        dxs2, _ = self.ev_attn(query=xs, key=evi_h, value=evi_h)
        dxs2 = self.ev_norm(dxs2)
        g_ev = torch.sigmoid(self.gate_ev(evi_pool)).view(1, B, 1)
        xs = xs + g_ev * (dxs2 * self.ls_ev.view(1, 1, -1))

        dth2, _ = self.el_attn(query=text_h, key=evi_h, value=evi_h)
        dth2 = self.el_norm(dth2)
        g_el = torch.sigmoid(self.gate_el(evi_pool)).view(1, B, 1)
        text_h = text_h + g_el * (dth2 * self.ls_el.view(1, 1, -1))

        # restore outputs
        text_out = self.text_out(text_h)
        alpha = 0.2
        text_out = (1 - alpha) * text_init + alpha * text_out

        xs = rearrange(xs, '(hh ww) b c -> b hh ww c', hh=h, ww=w)
        xs = self.D_fc2(xs)

        x = x + xs if self.skip_connect else xs
        return x, text_out, evidence


# ============================================================
# ARC Expert
# ============================================================
class ArcExpert(nn.Module):
    def __init__(self, in_dim=1024, kernel_number=1, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale

        self.pre = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.GELU()
        )
        r1 = RountingFunction(in_channels=in_dim, kernel_number=kernel_number)
        self.arc1 = AdaptiveRotatedConv2d(
            in_dim, in_dim, 3, padding=1,
            rounting_func=r1, bias=False, kernel_number=kernel_number
        )
        r2 = RountingFunction(in_channels=in_dim, kernel_number=kernel_number)
        self.arc2 = AdaptiveRotatedConv2d(
            in_dim, in_dim, 3, padding=1,
            rounting_func=r2, bias=False, kernel_number=kernel_number
        )
        self.post = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        x0 = x
        x = self.pre(x)
        x = F.gelu(self.arc1(x))
        x = self.arc2(x)
        x = self.post(x)
        return x0 + self.res_scale * x


# ============================================================
# Evidence-Driven Expert Prompting (MoE)
# ============================================================
class EvidenceDrivenExpertPrompting(nn.Module):
    """
    Three specialized experts: detail / elongated(ARC) / clutter
    Routing: evidence + CLS(text) joint routing.

    Input:
      - base_bhwc: (B,H,W,1024)
      - evidence:  (E,B,256) or (B,E,256) or None
      - text_cls:  (B,256)

    Output:
      - delta_logits: (B,H,W,1)
      - delta_tokens: (B,N,256)
      - gate: (B,1)
      - w: (B,3)
      - l_balance: scalar
    """
    def __init__(
        self,
        in_dim: int = 1024,
        evidence_dim: int = 256,
        text_dim: int = 256,
        prompt_dim: int = 256,
        num_extra_tokens: int = 1,
        gate_init: float = -6.0,
        gamma_init: float = 1e-3,
        temperature: float = 1.5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.evidence_dim = evidence_dim
        self.text_dim = text_dim
        self.prompt_dim = prompt_dim
        self.num_experts = 3
        self.num_extra_tokens = num_extra_tokens
        self.temperature = temperature

        # text modulation coefficient (init 0 => initially only evidence routing)
        self.beta_text = nn.Parameter(torch.tensor(0.0))

        # router
        router_in = evidence_dim * 2
        self.router = nn.Sequential(
            nn.Linear(router_in, evidence_dim),
            nn.GELU(),
            nn.Linear(evidence_dim, self.num_experts),
        )

        # gate
        self.gate_fc = nn.Linear(router_in, 1)
        nn.init.zeros_(self.gate_fc.weight)
        nn.init.constant_(self.gate_fc.bias, gate_init)

        # layerscale
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.gamma_tok = nn.Parameter(torch.tensor(gamma_init))

        def dwconv(kh, kw):
            return nn.Conv2d(in_dim, in_dim, (kh, kw),
                             padding=(kh // 2, kw // 2), groups=in_dim)

        # Expert-0: detail (3x3)
        self.exp_detail = nn.Sequential(
            dwconv(3, 3),
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, 1),
            nn.GELU(),
        )

        # Expert-1: elongated (ARC)
        self.exp_elong = ArcExpert(in_dim=in_dim, kernel_number=1)

        # Expert-2: clutter (GroupNorm + 1x1)
        self.exp_clut_norm = nn.GroupNorm(num_groups=32, num_channels=in_dim)
        self.exp_clutter = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.GELU(),
        )

        self.experts = nn.ModuleList([self.exp_detail, self.exp_elong, self.exp_clutter])

        # dense delta logits heads
        self.dense_heads = nn.ModuleList([nn.Conv2d(in_dim, 1, 1) for _ in range(self.num_experts)])
        # delta tokens heads
        self.token_heads = nn.ModuleList([
            nn.Linear(in_dim, num_extra_tokens * prompt_dim) for _ in range(self.num_experts)
        ])

        # zero init for safe start
        for h in self.dense_heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        for h in self.token_heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    @staticmethod
    def _normalize_evidence(evidence):
        if evidence.dim() != 3:
            raise ValueError(f"evidence must be 3D, got shape={tuple(evidence.shape)}")
        if evidence.shape[-1] != 256:
            raise ValueError(f"evidence last dim should be 256, got shape={tuple(evidence.shape)}")
        if evidence.shape[0] < evidence.shape[1]:
            return evidence.permute(1, 0, 2).contiguous()
        return evidence

    @staticmethod
    def balance_loss(w):
        p = w.mean(dim=0)
        K = p.numel()
        return K * torch.sum(p * p)

    def forward(self, base_bhwc, evidence, text_cls):
        B, H, W, C = base_bhwc.shape

        if evidence is None or text_cls is None:
            z_logits = base_bhwc.new_zeros((B, H, W, 1))
            z_tokens = base_bhwc.new_zeros((B, self.num_extra_tokens, self.prompt_dim))
            gate = base_bhwc.new_zeros((B, 1))
            w = base_bhwc.new_zeros((B, self.num_experts))
            lbal = base_bhwc.new_zeros(())
            return z_logits, z_tokens, gate, w, lbal

        evidence = self._normalize_evidence(evidence)
        e = evidence.mean(dim=0)   # (B, 256)
        t = text_cls               # (B, 256)

        r = torch.cat([e, self.beta_text * t], dim=-1)
        w = torch.softmax(self.router(r) / self.temperature, dim=-1)
        gate = torch.sigmoid(self.gate_fc(r))

        x = base_bhwc.permute(0, 3, 1, 2).contiguous()

        x0 = self.experts[0](x)                          # detail
        x1 = self.experts[1](x)                          # elongated
        x2 = self.experts[2](self.exp_clut_norm(x))      # clutter

        dense_list = [
            self.dense_heads[0](x0).permute(0, 2, 3, 1).contiguous(),
            self.dense_heads[1](x1).permute(0, 2, 3, 1).contiguous(),
            self.dense_heads[2](x2).permute(0, 2, 3, 1).contiguous(),
        ]

        def tok(xk, head):
            gap = F.adaptive_avg_pool2d(xk, 1).flatten(1)
            return head(gap).view(B, self.num_extra_tokens, self.prompt_dim)

        token_list = [tok(x0, self.token_heads[0]),
                      tok(x1, self.token_heads[1]),
                      tok(x2, self.token_heads[2])]

        delta_logits = x.new_zeros((B, H, W, 1))
        delta_tokens = base_bhwc.new_zeros((B, self.num_extra_tokens, self.prompt_dim))

        for k in range(self.num_experts):
            wk_dense = w[:, k].view(B, 1, 1, 1)
            delta_logits = delta_logits + wk_dense * dense_list[k]
            wk_tok = w[:, k].view(B, 1, 1)
            delta_tokens = delta_tokens + wk_tok * token_list[k]

        lbal = self.balance_loss(w)
        return delta_logits, delta_tokens, gate, w, lbal


# ============================================================
# Transformer Encoder Layer (for mask_token fusion)
# ============================================================
class VLFusionTokenModule(nn.Module):
    """Cross-attention: mask_token attends to visual features (from evf_sam_v3.py vmtoken)."""
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=None,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt * tgt2  # multiplicative fusion (from EviSAM)
        return tgt
