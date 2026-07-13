"""
FusionModel V3: Contact-Pair Aware Sequence-Structure Co-Encoding

핵심 변경 (vs V2):
  - Contact Encoder: ResNet(전체 압축) → ContactPairEncoder(쌍별 인코딩)
  - 각 contact(i,j)에 대해 ESM-2 token_i + token_j 직접 참조
  - 위치 대응이 살아있는 상태에서 서열+구조 정보 통합
  - K개 contact pair → Self-Attention → 512-dim 구조 표현

V2와의 차이:
  V2: ESM_CLS(압축후) ↔ ResNet_feat(압축후) → GCA
  V3: ESM_CLS(global) + ContactPair_i,j(local) → ContactPairEncoder → GCA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion_v2 import HierarchicalTransformerHead


# ── 1. Contact Pair Encoder ──────────────────────────────────
class ContactPairEncoder(nn.Module):
    """
    (B, K, 2, esm_dim) → (B, out_dim)

    각 contact pair (i,j):
      [token_i | token_j] (2*esm_dim) → Linear → pair_dim
    K개 pair → Self-Attention → Attention Pooling → out_dim

    직관:
      token_i, token_j는 ESM-2가 학습한 진화적 서열 맥락
      contact(i,j)는 두 잔기가 3D에서 가깝다는 구조 사실
      둘을 concat → "서열 맥락이 살아있는 구조 접촉" 인코딩
    """
    def __init__(self, esm_dim: int = 1280, pair_dim: int = 512,
                 out_dim: int = 512, n_heads: int = 8,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.pair_dim = pair_dim

        # 각 pair 인코딩: 2*esm_dim → pair_dim
        self.pair_proj = nn.Sequential(
            nn.Linear(2 * esm_dim, pair_dim),
            nn.LayerNorm(pair_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # K개 pair 간 상호작용 (Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pair_dim, nhead=n_heads,
            dim_feedforward=pair_dim * 2,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Attention Pooling: K개 → 1개
        self.attn_pool = nn.Linear(pair_dim, 1)
        self.out_proj  = nn.Sequential(
            nn.Linear(pair_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pair_emb: torch.Tensor) -> torch.Tensor:
        """
        pair_emb: (B, K, 2, esm_dim)  float32
        returns:  (B, out_dim)
        """
        B, K, _, D = pair_emb.shape

        # (B, K, 2*D) → (B, K, pair_dim)
        pairs = pair_emb.view(B, K, 2 * D)
        x = self.pair_proj(pairs)                      # (B, K, pair_dim)

        # K pair 간 Self-Attention
        x = self.transformer(x)                        # (B, K, pair_dim)

        # Attention Pooling over K
        attn_w = F.softmax(self.attn_pool(x), dim=1)  # (B, K, 1)
        pooled = (x * attn_w).sum(dim=1)              # (B, pair_dim)

        return self.out_proj(pooled)                   # (B, out_dim)


# ── 2. FusionModel V3 ────────────────────────────────────────
class FusionModelV3(nn.Module):
    """
    Fusion GCA V3: Contact-Pair Aware

    구조:
      ESM-2 CLS (1280) → esm_proj → (B, 1024)
                                         ↓ GCA
      ContactPair (K,2,1280) → ContactPairEncoder → (B, 512)
                              → contact_kv_proj → (B, 1024)

    V2와 동일한 GCA + HierarchicalTransformerHead 재사용.
    ContactPairEncoder만 새로 추가.

    입력:
      esm_emb:  (B, 1280)       — ESM-2 CLS token (기존 .npy)
      pair_emb: (B, K, 2, 1280) — contact pair tokens (새 .npy, float16→float32)
    """
    def __init__(self, n_classes: list, esm_dim: int = 1280,
                 contact_dim: int = 512, fusion_dim: int = 1024,
                 n_pairs: int = 32, num_heads: int = 8,
                 dropout: float = 0.3, head_layers: int = 2):
        super().__init__()

        # V3 핵심: Contact Pair Encoder
        self.contact_pair_enc = ContactPairEncoder(
            esm_dim=esm_dim,
            pair_dim=contact_dim,
            out_dim=contact_dim,
            n_heads=8,
            n_layers=2,
            dropout=dropout,
        )

        # GCA Fusion (V2와 동일)
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

        # Hierarchical Transformer Head (V2와 동일)
        self.head = HierarchicalTransformerHead(
            fusion_dim, n_classes, dropout=dropout, num_layers=head_layers
        )

    def forward(self, esm_emb: torch.Tensor,
                pair_emb: torch.Tensor, *args, **kwargs):
        """
        esm_emb:  (B, 1280)
        pair_emb: (B, K, 2, 1280)  float16 → float32 변환 후 입력
        """
        pair_emb = pair_emb.float()                           # float16 → float32

        contact_feat = self.contact_pair_enc(pair_emb)        # (B, 512)
        esm_proj     = self.esm_proj(esm_emb)                 # (B, 1024)
        contact_kv   = self.contact_kv_proj(contact_feat)     # (B, 1024)

        attn_out, _ = self.cross_attn(
            query=esm_proj.unsqueeze(1),
            key=contact_kv.unsqueeze(1),
            value=contact_kv.unsqueeze(1),
        )
        attn_out = attn_out.squeeze(1)

        gate  = torch.sigmoid(self.gate_fc(contact_feat))
        fused = esm_proj + gate * attn_out                    # (B, 1024)

        return self.head(fused)

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        pair  = sum(p.numel() for p in self.contact_pair_enc.parameters())
        return {"total": total, "contact_pair_enc": pair,
                "other": total - pair}
