"""Simple sequence-structure fusion baselines for Contact-EC.

These models reuse the same contact encoder and flat FC prediction head as
FusionV2FlatFC. They isolate whether the reported fusion gain requires the
current gated additive cross-attention block, or whether simpler fusion is enough.
"""

import torch
import torch.nn as nn

from models.fusion_v2 import AttentionContactEncoder
from models.fusion_v2_flatfc import FlatFCHead


class _Projection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FusionConcatFlatFC(nn.Module):
    """Concatenate projected ESM-2 and contact features, then classify."""

    def __init__(
        self,
        n_classes: list,
        esm_dim: int = 1280,
        contact_dim: int = 512,
        fusion_dim: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.contact_encoder = AttentionContactEncoder(out_dim=contact_dim)
        self.esm_proj = _Projection(esm_dim, fusion_dim, dropout)
        self.contact_proj = _Projection(contact_dim, fusion_dim, dropout)
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = FlatFCHead(fusion_dim, n_classes, dropout=dropout)

    def forward(self, esm_emb, cmap, *args, **kwargs):
        esm_feat = self.esm_proj(esm_emb)
        contact_feat = self.contact_proj(self.contact_encoder(cmap))
        fused = self.fuse(torch.cat([esm_feat, contact_feat], dim=-1))
        return self.head(fused)


class FusionSumFlatFC(nn.Module):
    """Sum projected ESM-2 and contact features, then classify."""

    def __init__(
        self,
        n_classes: list,
        esm_dim: int = 1280,
        contact_dim: int = 512,
        fusion_dim: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.contact_encoder = AttentionContactEncoder(out_dim=contact_dim)
        self.esm_proj = _Projection(esm_dim, fusion_dim, dropout)
        self.contact_proj = _Projection(contact_dim, fusion_dim, dropout)
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = FlatFCHead(fusion_dim, n_classes, dropout=dropout)

    def forward(self, esm_emb, cmap, *args, **kwargs):
        esm_feat = self.esm_proj(esm_emb)
        contact_feat = self.contact_proj(self.contact_encoder(cmap))
        fused = self.dropout(torch.relu(self.norm(esm_feat + contact_feat)))
        return self.head(fused)


class FusionGatedMLPFlatFC(nn.Module):
    """Learn a feature-wise gate between projected ESM-2 and contact features."""

    def __init__(
        self,
        n_classes: list,
        esm_dim: int = 1280,
        contact_dim: int = 512,
        fusion_dim: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.contact_encoder = AttentionContactEncoder(out_dim=contact_dim)
        self.esm_proj = _Projection(esm_dim, fusion_dim, dropout)
        self.contact_proj = _Projection(contact_dim, fusion_dim, dropout)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )
        self.post = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = FlatFCHead(fusion_dim, n_classes, dropout=dropout)

    def forward(self, esm_emb, cmap, *args, **kwargs):
        esm_feat = self.esm_proj(esm_emb)
        contact_feat = self.contact_proj(self.contact_encoder(cmap))
        gate = self.gate(torch.cat([esm_feat, contact_feat], dim=-1))
        fused = gate * esm_feat + (1.0 - gate) * contact_feat
        return self.head(self.post(fused))
