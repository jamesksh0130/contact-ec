"""
B3: Contact Map (3-channel) + 2D ResNet-18
입력: Contact Map (B, 3, 256, 256) → ResNet-18 → 512-d → 4 heads
ESM-2 미사용. 구조 정보만. (Ablation: Full model의 contact encoder 단독 성능 측정)
"""
import torch.nn as nn
from torchvision.models import resnet18


class ContactResNet(nn.Module):
    """
    B3: 2D ResNet-18 (in_channels=3) → 512-d feature → 4 heads
    3채널: all contacts / short-range / long-range
    """
    def __init__(self, n_classes: list, out_dim: int = 512,
                 dropout: float = 0.3):
        super().__init__()

        backbone = resnet18(weights=None)
        # 3채널 입력 (binary / short-range / long-range contact)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
        backbone.fc = nn.Identity()   # 512-d
        self.backbone = backbone

        self.proj = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList([
            nn.Linear(out_dim, nc) for nc in n_classes
        ])

    def forward(self, esm_emb, cmap, *args, **kwargs):
        # cmap: (B, 3, 256, 256)
        feat = self.backbone(cmap)          # (B, 512)
        feat = self.proj(feat)
        return [h(feat) for h in self.heads]
