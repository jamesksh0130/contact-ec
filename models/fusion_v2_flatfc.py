"""
FusionModel V2-FlatFC: Exp 2 ablation
  - AttentionContactEncoder (identical to fusion_v2)
  - GCA Gated Fusion (identical to fusion_v2)
  - Flat independent FC heads  ← only change vs fusion_v2

Purpose: isolate the contribution of the HierarchicalTransformerHead.
Compare:
  B1            — ESM-2 + flat FC, no contact map
  B4 (this)     — ESM-2 + contact map + GCA + flat FC  (head ablation)
  Contact-EC    — ESM-2 + contact map + GCA + Transformer head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from models.fusion_v2 import AttentionContactEncoder   # reuse encoder


class FlatFCHead(nn.Module):
    """4 independent flat FC heads — same as B1 style."""
    def __init__(self, in_dim: int, n_classes: list, dropout: float = 0.3):
        super().__init__()
        nc1, nc2, nc3, nc4 = n_classes
        self.head1 = nn.Linear(in_dim, nc1)
        self.head2 = nn.Linear(in_dim, nc2)
        self.head3 = nn.Linear(in_dim, nc3)
        self.head4 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim, nc4),
        )

    def forward(self, feat):
        return [self.head1(feat), self.head2(feat),
                self.head3(feat), self.head4(feat)]


class FusionV2FlatFC(nn.Module):
    """Contact-EC with flat FC heads instead of Hierarchical Transformer Head."""
    def __init__(self, n_classes: list, esm_dim: int = 1280,
                 contact_dim: int = 512, fusion_dim: int = 1024,
                 num_heads: int = 8, dropout: float = 0.3):
        super().__init__()

        self.contact_encoder = AttentionContactEncoder(out_dim=contact_dim)

        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.contact_kv_proj = nn.Linear(contact_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.gate_fc = nn.Linear(contact_dim, fusion_dim)

        self.head = FlatFCHead(fusion_dim, n_classes, dropout=dropout)

    def forward(self, esm_emb, cmap, *args, **kwargs):
        contact_feat = self.contact_encoder(cmap)
        esm_proj     = self.esm_proj(esm_emb)
        contact_kv   = self.contact_kv_proj(contact_feat)

        attn_out, _ = self.cross_attn(
            query=esm_proj.unsqueeze(1),
            key=contact_kv.unsqueeze(1),
            value=contact_kv.unsqueeze(1),
        )
        attn_out = attn_out.squeeze(1)

        gate  = torch.sigmoid(self.gate_fc(contact_feat))
        fused = esm_proj + gate * attn_out

        return self.head(fused)
