"""
FusionModel V2: Attention Pooling + Hierarchical Transformer Head

변경점 (vs fusion_model.py):
  1. ContactEncoder: GlobalAvgPool → Attention Pooling
     - ResNet-50 layer4 출력 (B, 2048, 8×8) = 64 공간 위치 유지
     - 학습 가능한 attention으로 중요 위치 가중 합산
     - 활성 부위(active site) 공간 정보 보존
  2. HierarchicalTransformerHead: 독립 FC 4개 → Transformer
     - 4 레벨을 sequence로 처리 (level embedding)
     - L1~L4가 서로 attend → 계층 관계 implicit 학습
     - B2 cascade failure 없음 (soft attention, hard decision 아님)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# ── 1. Attention Pooling Contact Encoder ────────────────────
class AttentionContactEncoder(nn.Module):
    """
    3-channel Contact Map → 512-d feature

    ResNet-50 layer4 출력 (B, 2048, 8, 8)을 공간별 attention으로 pooling.
    GlobalAvgPool 대비: 활성 부위 근처의 contact 패턴에 더 높은 가중치.
    """
    def __init__(self, out_dim: int = 512):
        super().__init__()
        backbone = resnet50(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)

        # layer4까지만 사용 (avgpool, fc 제거)
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )  # 출력: (B, 2048, 8, 8)

        # Attention: 각 공간 위치의 중요도 학습
        self.attn = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),   # (B, 1, 8, 8)
        )

        self.proj = nn.Sequential(
            nn.Linear(2048, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, cmap):                         # (B, 3, 256, 256)
        feat = self.features(cmap)                   # (B, 2048, 8, 8)
        B, C, H, W = feat.shape

        # Attention weights over 64 spatial positions
        attn_w = self.attn(feat)                     # (B, 1, 8, 8)
        attn_w = attn_w.view(B, 1, H*W)             # (B, 1, 64)
        attn_w = F.softmax(attn_w, dim=-1)           # (B, 1, 64)

        # Weighted sum
        feat_flat = feat.view(B, C, H*W)             # (B, 2048, 64)
        pooled = (feat_flat * attn_w).sum(dim=-1)    # (B, 2048)

        return self.proj(pooled)                     # (B, out_dim)


# ── 2. Hierarchical Transformer Head ────────────────────────
class HierarchicalTransformerHead(nn.Module):
    """
    fused (1024-dim) → 4-level Transformer → L1/L2/L3/L4 logits

    4개 레벨을 token sequence로 처리:
      [L1_token, L2_token, L3_token, L4_token]
      각 token = fused + level_embedding
      Transformer가 레벨 간 관계를 attention으로 학습

    HIT-EC와의 차이:
      - HIT-EC: 서열을 직접 Transformer로 처리 (Global+Local flow)
      - 우리: GCA fused representation 위에 level-wise Transformer 추가
    """
    def __init__(self, in_dim: int, n_classes: list,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        nc1, nc2, nc3, nc4 = n_classes
        self.n_levels = 4
        self.in_dim   = in_dim

        # 레벨별 구분 임베딩 (L1/L2/L3/L4가 어느 레벨인지 구분)
        self.level_emb = nn.Embedding(4, in_dim)

        # Transformer encoder (레벨 간 상호작용)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=num_heads,
            dim_feedforward=in_dim * 2,
            dropout=dropout, batch_first=True,
            norm_first=True,                  # Pre-LN (학습 안정성)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 각 레벨별 분류 head
        self.head1 = nn.Linear(in_dim, nc1)
        self.head2 = nn.Linear(in_dim, nc2)
        self.head3 = nn.Linear(in_dim, nc3)
        self.head4 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_dim, nc4),
        )

    def forward(self, feat):                         # (B, in_dim)
        B = feat.shape[0]

        # 4개 레벨 토큰 구성: fused + level_embedding
        level_ids = torch.arange(4, device=feat.device)     # [0,1,2,3]
        level_embs = self.level_emb(level_ids)               # (4, in_dim)
        tokens = feat.unsqueeze(1) + level_embs.unsqueeze(0) # (B, 4, in_dim)

        # Transformer: 레벨 간 attention
        out = self.transformer(tokens)                        # (B, 4, in_dim)

        # 각 레벨 토큰 → logits
        l1 = self.head1(out[:, 0, :])   # (B, nc1)
        l2 = self.head2(out[:, 1, :])   # (B, nc2)
        l3 = self.head3(out[:, 2, :])   # (B, nc3)
        l4 = self.head4(out[:, 3, :])   # (B, nc4)

        return [l1, l2, l3, l4]


# ── 3. FusionModel V2 ────────────────────────────────────────
class FusionModelV2(nn.Module):
    """
    Fusion GCA V2:
      - AttentionContactEncoder (spatial info preserved)
      - GCA Fusion (동일)
      - HierarchicalTransformerHead (level interaction)

    비교:
      V1 (fusion_model.py): GlobalAvgPool + 독립 FC
      V2 (this):            AttentionPool + Transformer Head
    """
    def __init__(self, n_classes: list, esm_dim: int = 1280,
                 contact_dim: int = 512, fusion_dim: int = 1024,
                 num_heads: int = 8, dropout: float = 0.3,
                 head_layers: int = 2):
        super().__init__()

        # V2 핵심: Attention Pooling
        self.contact_encoder = AttentionContactEncoder(out_dim=contact_dim)

        # GCA Fusion (V1과 동일)
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

        # V2 핵심: Hierarchical Transformer Head
        self.head = HierarchicalTransformerHead(
            fusion_dim, n_classes, dropout=dropout, num_layers=head_layers
        )

    def forward(self, esm_emb, cmap, *args, **kwargs):
        contact_feat = self.contact_encoder(cmap)            # (B, 512)
        esm_proj     = self.esm_proj(esm_emb)               # (B, 1024)
        contact_kv   = self.contact_kv_proj(contact_feat)   # (B, 1024)

        attn_out, _ = self.cross_attn(
            query=esm_proj.unsqueeze(1),
            key=contact_kv.unsqueeze(1),
            value=contact_kv.unsqueeze(1),
        )
        attn_out = attn_out.squeeze(1)

        gate  = torch.sigmoid(self.gate_fc(contact_feat))
        fused = esm_proj + gate * attn_out                   # (B, 1024)

        return self.head(fused)
