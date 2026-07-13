"""
B0: 1D CNN Baseline
입력: 아미노산 서열 → one-hot (B, 21, L) → 1D Conv 스택 → Global AvgPool → FC → 4 heads
ESM-2, Contact Map 모두 미사용. 순수 서열 정보만.
"""
import torch
import torch.nn as nn

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"   # 21 (X=unknown)
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}


def seq_to_onehot(sequences: list[str], max_len: int = 1024) -> torch.Tensor:
    """서열 리스트 → (B, 21, max_len) one-hot tensor."""
    B = len(sequences)
    x = torch.zeros(B, len(AA_VOCAB), max_len)
    for b, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_len]):
            idx = AA_TO_IDX.get(aa, AA_TO_IDX["X"])
            x[b, idx, j] = 1.0
    return x


class Conv1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    B0: 1D CNN
    서열 → 1D Conv 스택 → Global AvgPool → FC(512) → 4 heads
    """
    def __init__(self, n_classes: list[int], max_len: int = 1024,
                 dropout: float = 0.3):
        super().__init__()
        self.max_len = max_len

        # 1D Conv 스택: 21 → 128 → 256 → 512
        self.encoder = nn.Sequential(
            Conv1DBlock(21,  128, kernel=7),
            nn.MaxPool1d(2),                  # L/2
            Conv1DBlock(128, 256, kernel=5),
            nn.MaxPool1d(2),                  # L/4
            Conv1DBlock(256, 512, kernel=3),
            Conv1DBlock(512, 512, kernel=3),
            nn.AdaptiveAvgPool1d(1),          # (B, 512, 1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 4-level 독립 헤드
        self.heads = nn.ModuleList([
            nn.Linear(512, nc) for nc in n_classes
        ])

    def forward(self, sequences: list[str], device=None):
        x = seq_to_onehot(sequences, self.max_len)
        if device is not None:
            x = x.to(device)
        feat = self.fc(self.encoder(x))
        return [h(feat) for h in self.heads]   # [(B, nc1), ..., (B, nc4)]
