"""The student: their released 32M MLM encoder plus a fresh MTR head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def patch_sdpa_dtype():
    """Their MultiHeadAttention hardcodes the padding mask to float32:

        mask = mask.to(dtype=torch.float32); mask = (1.0 - mask) * -1e9

    and passes it to scaled_dot_product_attention. Under autocast or .half() the
    queries are fp16 and CUDA refuses:

        RuntimeError: invalid dtype for bias - should match query's dtype

    CPU silently promotes, so this only ever appears on GPU. Wrapping SDPA leaves
    their model file untouched; they resolve F.scaled_dot_product_attention at
    call time, so the wrapper is picked up.

    The clamp matters independently: -1e9 cast to fp16 becomes -inf, so we pin it
    to the largest finite negative and keep the arithmetic finite.
    """
    if getattr(F.scaled_dot_product_attention, "_dtype_patched", False):
        return
    orig = F.scaled_dot_product_attention

    def safe(query, key, value, attn_mask=None, **kw):
        if attn_mask is not None and attn_mask.dtype not in (torch.bool, query.dtype):
            attn_mask = attn_mask.to(query.dtype).clamp_(min=torch.finfo(query.dtype).min)
        return orig(query, key, value, attn_mask=attn_mask, **kw)

    safe._dtype_patched = True
    F.scaled_dot_product_attention = safe


class Student(nn.Module):
    """Encoder + MLM head (both pretrained) + MTR head (fresh).

    The released MLM checkpoints contain only embed / transformer / sequence_head
    -- no descriptor head -- so the MTR head starts random. Expect its loss to
    dominate for the first few hundred steps while it calibrates.
    """

    def __init__(self, init_path, n_desc=99):
        super().__init__()
        patch_sdpa_dtype()
        self.backbone = AutoModel.from_pretrained(
            init_path, trust_remote_code=True, use_safetensors=True)
        d = self.backbone.config.embed_dim
        # Same shape as their pretraining head: Linear(d,d) -> SiLU -> Linear(d,99)
        self.mtr_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, n_desc))
        self.embed_dim = d

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return {
            "logits": out.logits,            # (B, T, 405)
            "mean_pool": out.mean_pool,      # (B, d)
            "mtr": self.mtr_head(out.mean_pool),
        }

    def param_groups(self, lr_backbone, lr_head, weight_decay=0.01):
        """No weight decay on norms and biases -- standard, and it matters here
        because LayerNorm gains sit at ~1.0 and decaying them fights the
        pretrained initialisation we deliberately started from."""
        decay, no_decay = [], []
        for n, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        head = list(self.mtr_head.parameters())
        return [
            {"params": decay, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr_backbone, "weight_decay": 0.0},
            {"params": head, "lr": lr_head, "weight_decay": weight_decay},
        ]
