"""
B2: ESM-2 + Hierarchical FC
B1과 동일한 구조이나 학습 시 계층 가중치 loss를 적용하고
상위 레벨 예측을 하위 레벨 입력에 concat하여 계층성 반영.
"""
import torch
import torch.nn as nn


class ESM2Hierarchical(nn.Module):
    """
    B2: ESM-2 → shared trunk → 4 hierarchical heads
    각 head는 shared feature + 이전 레벨의 logit을 입력으로 받아
    계층 구조 정보를 활용.
    """
    def __init__(self, n_classes: list[int], esm_dim: int = 1280,
                 trunk_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.n_classes = n_classes

        # 공유 trunk
        self.trunk = nn.Sequential(
            nn.Linear(esm_dim, trunk_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Level 1: trunk → nc1
        self.head1 = nn.Linear(trunk_dim, n_classes[0])

        # Level 2: trunk + l1_logits → nc2
        self.head2 = nn.Sequential(
            nn.Linear(trunk_dim + n_classes[0], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_classes[1]),
        )

        # Level 3: trunk + l2_logits → nc3
        self.head3 = nn.Sequential(
            nn.Linear(trunk_dim + n_classes[1], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_classes[2]),
        )

        # Level 4: trunk + l3_logits → nc4
        self.head4 = nn.Sequential(
            nn.Linear(trunk_dim + n_classes[2], 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, n_classes[3]),
        )

    def forward(self, esm_emb, *args, **kwargs):
        feat = self.trunk(esm_emb)          # (B, trunk_dim)

        l1 = self.head1(feat)               # (B, nc1)
        l2 = self.head2(torch.cat([feat, l1], dim=1))
        l3 = self.head3(torch.cat([feat, l2], dim=1))
        l4 = self.head4(torch.cat([feat, l3], dim=1))

        return [l1, l2, l3, l4]
