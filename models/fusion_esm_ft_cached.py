"""
Fusion + ESM-2 Partial-Cache Fine-tuning (Phase 2 전용)

Phase 2 최적화:
  - Layer 0-26: 미리 캐싱된 hidden states 로드 (frozen → 항상 동일 출력)
  - Layer 27-32: unfrozen, 배치마다 실시간 계산 (학습 대상)
  - 속도: ~3-4× 빠름 (ESM-2 계산의 82% 절약)
  - 결과: 전통적 Phase 2와 수학적으로 동일

사용법:
  # Phase 2 캐시 학습 시작
  python scripts/train_phase2_cached.py --resume expa_e2e_phase1_best.pt
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel
from models.fusion_model import ContactEncoder, HierarchicalHead
from models.fusion_esm_ft import FusionESMFinetune


class FusionESMFinetuneV2(FusionESMFinetune):
    """
    FusionESMFinetune + Layer 26 캐시 지원.
    Phase 1: 기존 forward(sequences, cmap) 그대로 사용
    Phase 2: forward_cached(layer26_h, attn_mask, cmap) 사용
    """

    def forward_cached(
        self,
        layer26_h: torch.Tensor,    # (B, max_seq_len, 1280) fp32, padded
        attn_mask: torch.Tensor,    # (B, max_seq_len) float, 1=valid 0=padding
        cmap: torch.Tensor,         # (B, 3, 256, 256)
    ) -> list:
        """
        Phase 2 forward with partial ESM-2 cache.

        layer26_h : hidden states after ESM-2 layer 26 (last frozen layer)
        attn_mask : binary attention mask (1=valid token, 0=padding)
        """
        device = cmap.device
        layer26_h = layer26_h.to(device)
        attn_mask = attn_mask.to(device)

        # Extended attention mask: (B, 1, 1, max_len), 0=valid, -10000=padding
        extended_mask = (1.0 - attn_mask[:, None, None, :]) * -10000.0

        # Run layers 27-32 (unfrozen, trainable)
        h = layer26_h
        for layer_module in self.esm.encoder.layer[self.N_LAYERS - self.unfreeze_layers:]:
            h = layer_module(h, extended_mask)[0]

        # CLS token (index 0 = BOS token in ESM-2)
        esm_emb = h[:, 0, :]   # (B, 1280)

        # GCA Fusion (identical to normal forward)
        contact_feat = self.contact_encoder(cmap)           # (B, 512)
        esm_proj     = self.esm_proj(esm_emb)               # (B, 1024)
        contact_kv   = self.contact_kv_proj(contact_feat)   # (B, 1024)
        attn_out, _  = self.cross_attn(
            query=esm_proj.unsqueeze(1),
            key=contact_kv.unsqueeze(1),
            value=contact_kv.unsqueeze(1),
        )
        attn_out = attn_out.squeeze(1)
        gate  = torch.sigmoid(self.gate_fc(contact_feat))
        fused = esm_proj + gate * attn_out                  # (B, 1024)

        return self.head(fused)

    def unfreeze_esm_last(self, n: int = None):
        """unfreeze_layers를 self.unfreeze_layers로 업데이트 후 상위 메서드 호출."""
        n = n or self.unfreeze_layers
        self.unfreeze_layers = n   # forward_cached에서 사용
        super().unfreeze_esm_last(n)
