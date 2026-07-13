"""
B1: ESM-2 + Flat FC
입력: 사전 추출된 ESM-2 임베딩 (B, 1280) → FC → 4개 독립 head
계층 가중치 없이 단순 flat classification.
"""
import torch.nn as nn


class ESM2FC(nn.Module):
    """
    B1: ESM-2 임베딩 → FC(1280→512) → 4-head
    계층 구조 없이 레벨별 독립 분류.
    """
    def __init__(self, n_classes: list[int], esm_dim: int = 1280,
                 hidden: int = 512, dropout: float = 0.3):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(esm_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList([
            nn.Linear(hidden, nc) for nc in n_classes
        ])

    def forward(self, esm_emb, *args, **kwargs):
        # esm_emb: (B, 1280)
        feat = self.backbone(esm_emb)
        return [h(feat) for h in self.heads]
