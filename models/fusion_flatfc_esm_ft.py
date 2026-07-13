"""
FusionFlatFCESMFt: FusionV2FlatFC + live ESM-2 end-to-end fine-tuning.

기존 FusionV2FlatFC (사전 계산 임베딩 사용)에 ESM-2를 내장하여
E2E fine-tuning이 가능하도록 확장.

아키텍처:
  - ESM-2 (live, last N layers unfreeze)
  - AttentionContactEncoder (동일)
  - GCA Fusion (동일)
  - FlatFCHead (동일) + BCEWithLogitsLoss

초기화:
  1. load_flatfc_checkpoint()로 FusionV2FlatFC 가중치 로드
  2. ESM-2는 HuggingFace에서 frozen으로 로드
  3. unfreeze_esm_last(6)으로 마지막 6층 unfreeze
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel

from models.fusion_v2 import AttentionContactEncoder
from models.fusion_v2_flatfc import FlatFCHead


class FusionFlatFCESMFt(nn.Module):
    ESM_MODEL  = "facebook/esm2_t33_650M_UR50D"
    ESM_DIM    = 1280
    N_LAYERS   = 33

    def __init__(self, n_classes: list, contact_dim: int = 512,
                 fusion_dim: int = 1024, num_heads: int = 8,
                 dropout: float = 0.3, unfreeze_layers: int = 6):
        super().__init__()
        self.unfreeze_layers = unfreeze_layers

        # ESM-2 (frozen 으로 시작)
        self.tokenizer = AutoTokenizer.from_pretrained(self.ESM_MODEL)
        self.esm       = EsmModel.from_pretrained(self.ESM_MODEL)
        self._freeze_esm_all()

        # Contact encoder + GCA — FusionV2FlatFC와 완전 동일
        self.contact_encoder  = AttentionContactEncoder(out_dim=contact_dim)
        self.esm_proj         = nn.Sequential(
            nn.Linear(self.ESM_DIM, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.contact_kv_proj  = nn.Linear(contact_dim, fusion_dim)
        self.cross_attn       = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.gate_fc          = nn.Linear(contact_dim, fusion_dim)

        # FlatFCHead (binary CE 학습)
        self.head = FlatFCHead(fusion_dim, n_classes, dropout)

    # ── ESM-2 freeze / unfreeze ──────────────────────────────────────────────
    def _freeze_esm_all(self):
        for p in self.esm.parameters():
            p.requires_grad = False

    def unfreeze_esm_last(self, n: int = None):
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
        esm_params  = [p for p in self.esm.parameters() if p.requires_grad]
        rest_params = [p for nm, p in self.named_parameters()
                       if not nm.startswith("esm.") and p.requires_grad]
        return [
            {"params": esm_params,  "lr": lr_esm,  "name": "esm_finetune"},
            {"params": rest_params, "lr": lr_rest, "name": "rest"},
        ]

    # ── FusionV2FlatFC 체크포인트 로드 ───────────────────────────────────────
    def load_flatfc_checkpoint(self, ckpt_path: str):
        """FusionV2FlatFC 가중치를 비-ESM 파트에 로드."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        own  = self.state_dict()
        loaded, skipped = 0, 0
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                loaded += 1
            else:
                skipped += 1
        self.load_state_dict(own)
        print(f"FlatFC 체크포인트 로드: {loaded}개 파라미터 매핑, {skipped}개 스킵")

    # ── Forward (일반, live ESM-2) ───────────────────────────────────────────
    def forward(self, sequences: list, cmap: torch.Tensor, **kwargs):
        device = cmap.device
        enc = self.tokenizer(
            sequences, return_tensors="pt",
            padding=True, truncation=True, max_length=1024,
        ).to(device)

        if any(p.requires_grad for p in self.esm.parameters()):
            esm_out = self.esm(**enc)
        else:
            with torch.no_grad():
                esm_out = self.esm(**enc)
        esm_emb = esm_out.last_hidden_state[:, 0, :]  # CLS (B, 1280)

        return self._gca_forward(esm_emb, cmap)

    # ── Forward (캐시, Layer26 hidden state) ─────────────────────────────────
    def forward_cached(self, layer26_h: torch.Tensor,
                       attn_mask: torch.Tensor, cmap: torch.Tensor):
        """Layer 26 이후만 실행 (frozen 0-26 캐시 활용)."""
        device = cmap.device
        layer26_h = layer26_h.to(device)
        attn_mask = attn_mask.to(device)
        extended  = (1.0 - attn_mask[:, None, None, :]) * -10000.0

        h = layer26_h
        for lyr in self.esm.encoder.layer[self.N_LAYERS - self.unfreeze_layers:]:
            h = lyr(h, extended)[0]
        esm_emb = h[:, 0, :]  # CLS

        return self._gca_forward(esm_emb, cmap)

    # ── GCA Fusion (공통) ────────────────────────────────────────────────────
    def _gca_forward(self, esm_emb: torch.Tensor, cmap: torch.Tensor):
        contact_feat = self.contact_encoder(cmap)           # (B, 512)
        esm_proj     = self.esm_proj(esm_emb)               # (B, 1024)
        contact_kv   = self.contact_kv_proj(contact_feat)   # (B, 1024)

        attn_out, _ = self.cross_attn(
            query=esm_proj.unsqueeze(1),
            key=contact_kv.unsqueeze(1),
            value=contact_kv.unsqueeze(1),
        )
        attn_out = attn_out.squeeze(1)
        gate     = torch.sigmoid(self.gate_fc(contact_feat))
        fused    = esm_proj + gate * attn_out               # (B, 1024)

        return self.head(fused)                             # [l1, l2, l3, l4]
