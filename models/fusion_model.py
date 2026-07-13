"""
Gated Cross-Attention Fusion: ESM-2 + Contact Map (3-channel) + Hierarchical Head

Path A: ESM-2 embedding (1280,) → proj → esm_feat (1024)
Path B: Contact Map (3,256,256) → ResNet-50 → 512 → proj → contact_kv (1024)

Fusion (Gated Cross-Attention):
  1) Cross-Attention: query=esm_feat, key/value=contact_kv
     → sequence context가 structural feature의 어느 부분을 볼지 결정
  2) Gate = sigmoid(W_g · contact_512) ∈ (0,1)^1024
     → contact 기여도를 동적으로 조절
  3) fused = esm_feat + gate ⊙ attn_out  (Residual)
     → 최소한 ESM-2 수준 보장

CLEAN-Contact (Yang et al., 2024) 대비 차별화:
  - Gated Cross-Attention vs 단순 concat + contrastive
  - 의미 있는 3채널 (all/short-range/long-range) vs 동일 채널 3복사
  - 계층적 multi-label EC 예측 (4-level cascade) vs single-label flat
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50


class ContactEncoder(nn.Module):
    """3-channel Contact Map → 512-d feature (ResNet-50 backbone)."""
    def __init__(self, out_dim: int = 512):
        super().__init__()
        backbone = resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
        backbone.fc = nn.Identity()   # ResNet-50 global avg pool → 2048-d
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(2048, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, cmap):                # (B, 3, 256, 256)
        return self.proj(self.backbone(cmap))   # (B, out_dim)


class HierarchicalHead(nn.Module):
    """Trunk feature → 4-level hierarchical prediction (cascaded logit conditioning)."""
    def __init__(self, in_dim: int, n_classes: list, dropout: float = 0.3):
        super().__init__()
        nc1, nc2, nc3, nc4 = n_classes

        self.head1 = nn.Linear(in_dim, nc1)

        self.head2 = nn.Sequential(
            nn.Linear(in_dim + nc1, 512), nn.ReLU(inplace=True),
            nn.Linear(512, nc2),
        )
        self.head3 = nn.Sequential(
            nn.Linear(in_dim + nc2, 512), nn.ReLU(inplace=True),
            nn.Linear(512, nc3),
        )
        self.head4 = nn.Sequential(
            nn.Linear(in_dim + nc3, 1024), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, nc4),
        )

    def forward(self, feat):
        l1 = self.head1(feat)
        l2 = self.head2(torch.cat([feat, l1], dim=1))
        l3 = self.head3(torch.cat([feat, l2], dim=1))
        l4 = self.head4(torch.cat([feat, l3], dim=1))
        return [l1, l2, l3, l4]


class FusionModel(nn.Module):
    """
    Gated Cross-Attention Fusion Model

    Forward:
      1. contact_feat = ContactEncoder(cmap)              (B, 512)
      2. esm_proj     = LayerNorm(ReLU(W_e · esm_emb))   (B, 1024)
      3. contact_kv   = W_kv · contact_feat               (B, 1024)
      4. attn_out     = CrossAttention(Q=esm_proj, K=contact_kv, V=contact_kv)
      5. gate         = sigmoid(W_g · contact_feat)        (B, 1024)
      6. fused        = esm_proj + gate ⊙ attn_out        (B, 1024)
      7. [l1,l2,l3,l4] = HierarchicalHead(fused)
    """
    def __init__(self, n_classes: list, esm_dim: int = 1280,
                 contact_dim: int = 512, fusion_dim: int = 1024,
                 num_heads: int = 8, dropout: float = 0.3):
        super().__init__()

        self.contact_encoder = ContactEncoder(out_dim=contact_dim)

        # ESM-2 → fusion_dim (base feature)
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Contact → fusion_dim (key/value for cross-attention)
        self.contact_kv_proj = nn.Linear(contact_dim, fusion_dim)

        # Cross-Attention: ESM-2 query ← Contact key/value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gate: contact 기여도 동적 조절
        self.gate_fc = nn.Linear(contact_dim, fusion_dim)

        self.head = HierarchicalHead(fusion_dim, n_classes, dropout)

    def forward(self, esm_emb, cmap, *args, **kwargs):
        # Contact encoding
        contact_feat = self.contact_encoder(cmap)            # (B, 512)

        # ESM-2 projection (base: 최소한 이 수준 보장)
        esm_proj = self.esm_proj(esm_emb)                   # (B, 1024)

        # Contact key/value projection
        contact_kv = self.contact_kv_proj(contact_feat)     # (B, 1024)

        # Cross-Attention: sequence가 structural feature에서 무엇을 볼지 결정
        attn_out, _ = self.cross_attn(
            query = esm_proj.unsqueeze(1),                   # (B, 1, 1024)
            key   = contact_kv.unsqueeze(1),                 # (B, 1, 1024)
            value = contact_kv.unsqueeze(1),                 # (B, 1, 1024)
        )
        attn_out = attn_out.squeeze(1)                       # (B, 1024)

        # Gating: contact 기여도를 0~1로 조절
        gate = torch.sigmoid(self.gate_fc(contact_feat))     # (B, 1024)

        # Residual Fusion: ESM-2 base + gated structural correction
        fused = esm_proj + gate * attn_out                   # (B, 1024)

        return self.head(fused)                              # [l1,l2,l3,l4]

    def freeze_contact_encoder(self):
        for p in self.contact_encoder.parameters():
            p.requires_grad = False

    def unfreeze_contact_encoder(self):
        for p in self.contact_encoder.parameters():
            p.requires_grad = True
