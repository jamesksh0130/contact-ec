"""
FusionV3 + ESM-2 Last N Layers Fine-tuning

Phase 1 checkpoint (frozen ESM-2) → Phase 2: last 2 layers unfrozen
Forward:
  sequences + cmap(3ch, 256x256) →
    ESM-2 residue tokens (on-the-fly) →
    contact pair extraction (from cmap channel 2: long-range) →
    ContactPairEncoder → GCA → HierarchicalTransformerHead
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, EsmModel
from models.fusion_v3 import ContactPairEncoder, HierarchicalTransformerHead

K_PAIRS    = 32
LR_THRESH  = 12   # |i-j| >= 12 (256-grid 기준 long-range)
CMAP_SIZE  = 256
ESM_MODEL  = "facebook/esm2_t33_650M_UR50D"
ESM_DIM    = 1280
N_LAYERS   = 33


def _select_top_k(cmap_1ch: torch.Tensor, k: int = K_PAIRS) -> torch.Tensor:
    """
    (H, W) binary contact map (torch, float) → (k, 2) long-range 위치 (int64).
    배치 처리 없이 single sample. 균일 샘플링, seed=42.
    """
    cmap_np = cmap_1ch.cpu().numpy().astype(bool)
    H, W    = cmap_np.shape
    idx     = np.arange(H)
    diff    = np.abs(idx[:, None] - idx[None, :])
    lr_mask = diff >= LR_THRESH
    upper   = np.triu(np.ones((H, W), dtype=bool), k=1)
    valid   = cmap_np & lr_mask & upper
    pos     = np.argwhere(valid)           # (N, 2)
    if len(pos) == 0:
        pos = np.argwhere(np.triu(cmap_np, k=1))
    if len(pos) == 0:
        return torch.zeros(k, 2, dtype=torch.long)
    rng = np.random.default_rng(seed=42)
    chosen = rng.choice(len(pos), k, replace=(len(pos) < k))
    return torch.from_numpy(pos[chosen].astype(np.int64))   # (k, 2)


def _grid_to_res(grid: torch.Tensor, seq_len: int) -> torch.Tensor:
    """(k, 2) grid coords → residue indices, clamped to [0, min(seq_len,1024)-1]"""
    r = (grid.float() * seq_len / CMAP_SIZE).long()
    return r.clamp(0, min(seq_len, 1024) - 1)


class FusionV3ESMFt(nn.Module):
    """FusionV3 + ESM-2 last N layers end-to-end fine-tuning."""

    def __init__(self, n_classes: list, esm_dim: int = ESM_DIM,
                 contact_dim: int = 512, fusion_dim: int = 1024,
                 k_pairs: int = K_PAIRS, unfreeze_layers: int = 2,
                 dropout: float = 0.1, head_layers: int = 2):
        super().__init__()
        self.k_pairs         = k_pairs
        self.unfreeze_layers = unfreeze_layers

        # ESM-2 (전체 frozen으로 시작, gradient checkpointing으로 메모리 절약)
        self.tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
        self.esm       = EsmModel.from_pretrained(ESM_MODEL)
        self.esm.gradient_checkpointing_enable()
        self._freeze_esm_all()

        # ContactPairEncoder (V3와 동일)
        self.contact_encoder = ContactPairEncoder(
            esm_dim=esm_dim, pair_dim=contact_dim,
            out_dim=contact_dim, n_heads=8, n_layers=2, dropout=dropout,
        )

        # GCA Fusion (V3와 동일)
        esm_proj_dim = fusion_dim
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, esm_proj_dim),
            nn.LayerNorm(esm_proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.contact_kv_proj = nn.Linear(contact_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8,
            dropout=dropout, batch_first=True,
        )
        self.gate_fc = nn.Linear(contact_dim, fusion_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )

        # HierarchicalTransformerHead (V2/V3와 동일)
        self.head = HierarchicalTransformerHead(
            fusion_dim, n_classes, dropout=dropout, num_layers=head_layers,
        )

    # ── ESM-2 freeze/unfreeze ─────────────────────────────────────
    def _freeze_esm_all(self):
        for p in self.esm.parameters():
            p.requires_grad = False

    def unfreeze_esm_last(self, n: int = None):
        n = n or self.unfreeze_layers
        self._freeze_esm_all()
        for layer in self.esm.encoder.layer[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        unfrozen = sum(p.numel() for p in self.esm.parameters() if p.requires_grad)
        print(f"ESM-2 last {n} layers unfrozen: {unfrozen/1e6:.1f}M params")

    def get_param_groups(self, lr_esm: float = 1e-5, lr_rest: float = 1e-4):
        esm_params  = [p for p in self.esm.parameters() if p.requires_grad]
        rest_params = [p for n, p in self.named_parameters()
                       if not n.startswith("esm.") and p.requires_grad]
        return [
            {"params": esm_params,  "lr": lr_esm,  "name": "esm_finetune"},
            {"params": rest_params, "lr": lr_rest, "name": "rest"},
        ]

    # ── Forward ────────────────────────────────────────────────────
    def forward(self, sequences: list, cmap: torch.Tensor, **kwargs):
        """
        sequences : list[str], length B
        cmap      : (B, 3, 256, 256) float32  — channel 2 = long-range contacts
        """
        device = cmap.device
        B = len(sequences)

        # ── ESM-2 forward ───────────────────────────────────────
        enc = self.tokenizer(
            sequences, return_tensors="pt",
            padding=True, truncation=True, max_length=1024,
        ).to(device)

        if any(p.requires_grad for p in self.esm.parameters()):
            esm_out = self.esm(**enc)
        else:
            with torch.no_grad():
                esm_out = self.esm(**enc)

        tok_emb = esm_out.last_hidden_state   # (B, L+2, 1280)  [CLS] token=0
        esm_emb = tok_emb[:, 0, :]           # (B, 1280)

        # ── Contact pair extraction (from cmap channel 2) ────────
        # cmap channel 2 = long-range contacts (256×256)
        cmap_lr   = cmap[:, 2, :, :]          # (B, 256, 256)
        pair_list = []
        for b in range(B):
            grid_pos = _select_top_k(cmap_lr[b], self.k_pairs)   # (k, 2)
            seq_len  = enc["attention_mask"][b].sum().item() - 2  # exclude [CLS],[EOS]
            res_pos  = _grid_to_res(grid_pos, seq_len)            # (k, 2)
            # tok_emb: index 0=[CLS], 1..L=residues, L+1=[EOS]
            res_pos_shifted = (res_pos + 1).clamp(0, tok_emb.shape[1] - 1)  # +1 for [CLS]
            emb_i = tok_emb[b, res_pos_shifted[:, 0], :]   # (k, 1280)
            emb_j = tok_emb[b, res_pos_shifted[:, 1], :]   # (k, 1280)
            pair  = torch.stack([emb_i, emb_j], dim=1)     # (k, 2, 1280)
            pair_list.append(pair)

        pair_emb = torch.stack(pair_list).float()           # (B, k, 2, 1280)

        # ── ContactPairEncoder ────────────────────────────────────
        contact_feat = self.contact_encoder(pair_emb)       # (B, 512)

        # ── GCA Fusion ────────────────────────────────────────────
        esm_proj   = self.esm_proj(esm_emb)                 # (B, 1024)
        contact_kv = self.contact_kv_proj(contact_feat)     # (B, 1024)
        attn_out, _ = self.cross_attn(
            query=esm_proj.unsqueeze(1),
            key=contact_kv.unsqueeze(1),
            value=contact_kv.unsqueeze(1),
        )
        attn_out = attn_out.squeeze(1)
        gate  = torch.sigmoid(self.gate_fc(contact_feat))
        fused = esm_proj + gate * attn_out
        fused = self.out_proj(fused)                        # (B, 1024)

        return self.head(fused)
