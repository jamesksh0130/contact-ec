"""
Fusion + ESM-2 End-to-End Fine-tuning

CLEAN-Contact와의 차별점:
  1. GCA (Gated Cross-Attention) — 단순 addition 아님
  2. 4-level 계층 EC 예측 (cascade)
  3. 3채널 Contact Map (all/short/long-range)
  4. ESM-2 last N layer fine-tuning end-to-end
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel
from models.fusion_model import ContactEncoder, HierarchicalHead


class FusionESMFinetune(nn.Module):
    """
    ESM-2 last N layers 실제 fine-tuning + GCA Fusion.

    Phase 1: ESM-2 완전 frozen, Contact+GCA+Head만 학습 (lr=1e-3)
    Phase 2: ESM-2 last unfreeze_layers층 unfreeze      (lr_esm=1e-5, lr_rest=1e-4)
    """
    ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
    ESM_DIM   = 1280
    N_LAYERS  = 33

    def __init__(self, n_classes: list, contact_dim: int = 512,
                 fusion_dim: int = 1024, num_heads: int = 8,
                 dropout: float = 0.3, unfreeze_layers: int = 4):
        super().__init__()
        self.unfreeze_layers = unfreeze_layers

        # ESM-2 로드 (전체 frozen으로 시작)
        self.tokenizer = AutoTokenizer.from_pretrained(self.ESM_MODEL)
        self.esm       = EsmModel.from_pretrained(self.ESM_MODEL)
        self._freeze_esm_all()

        # Contact Map Encoder (ResNet-50, 3채널)
        self.contact_encoder = ContactEncoder(out_dim=contact_dim)

        # GCA Fusion (fusion_model.py와 동일 구조)
        self.esm_proj = nn.Sequential(
            nn.Linear(self.ESM_DIM, fusion_dim),
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
        self.head    = HierarchicalHead(fusion_dim, n_classes, dropout)

    # ── ESM-2 freeze/unfreeze ─────────────────────────────────────
    def _freeze_esm_all(self):
        for p in self.esm.parameters():
            p.requires_grad = False

    def unfreeze_esm_last(self, n: int = None):
        """마지막 n층 + final LayerNorm unfreeze."""
        n = n or self.unfreeze_layers
        self._freeze_esm_all()
        for layer in self.esm.encoder.layer[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        ln = getattr(self.esm.embeddings, "layer_norm", None)
        if ln is not None:
            for p in ln.parameters():
                p.requires_grad = True
        unfrozen = sum(p.numel() for p in self.esm.parameters() if p.requires_grad)
        print(f"ESM-2 last {n} layers unfrozen: {unfrozen/1e6:.1f}M params")

    def get_param_groups(self, lr_esm: float = 1e-5, lr_rest: float = 1e-4):
        """ESM-2 fine-tuning 파라미터 그룹 (차등 lr)."""
        esm_params  = [p for p in self.esm.parameters() if p.requires_grad]
        rest_params = [p for n, p in self.named_parameters()
                       if not n.startswith("esm.") and p.requires_grad]
        return [
            {"params": esm_params,  "lr": lr_esm,  "name": "esm_finetune"},
            {"params": rest_params, "lr": lr_rest, "name": "rest"},
        ]

    # ── Forward ────────────────────────────────────────────────────
    def forward(self, sequences: list, cmap: torch.Tensor, **kwargs):
        device = cmap.device

        # ESM-2: sequences → CLS token embedding
        enc = self.tokenizer(
            sequences, return_tensors="pt",
            padding=True, truncation=True, max_length=1024,
        ).to(device)

        if any(p.requires_grad for p in self.esm.parameters()):
            esm_out = self.esm(**enc)
        else:
            with torch.no_grad():
                esm_out = self.esm(**enc)
        esm_emb = esm_out.last_hidden_state[:, 0, :]  # CLS token (B, 1280)

        # Contact Map → feature
        contact_feat = self.contact_encoder(cmap)           # (B, 512)

        # GCA Fusion (동일 구조)
        esm_proj    = self.esm_proj(esm_emb)                # (B, 1024)
        contact_kv  = self.contact_kv_proj(contact_feat)    # (B, 1024)
        attn_out, _ = self.cross_attn(
            query=esm_proj.unsqueeze(1),
            key=contact_kv.unsqueeze(1),
            value=contact_kv.unsqueeze(1),
        )
        attn_out = attn_out.squeeze(1)
        gate  = torch.sigmoid(self.gate_fc(contact_feat))
        fused = esm_proj + gate * attn_out                  # (B, 1024)

        return self.head(fused)
