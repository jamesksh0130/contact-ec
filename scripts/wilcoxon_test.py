"""
Wilcoxon Signed-Rank Test: V2 vs B2 (Contact Map 기여 통계 검정)

연구 목적:
  "Contact Map 추가가 서열만 사용하는 것보다 통계적으로 유의미하게 좋은가?"

방법:
  1. V2와 B2로 test set 전체 추론 → 샘플별 L4 F1 계산
  2. Paired Wilcoxon signed-rank test (n=23,046 샘플)
  3. p < 0.05 → Contact Map 기여 통계적으로 유의

출력:
  outputs/results/wilcoxon_v2_vs_b2.json
"""
import sys, pickle, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

ROOT = Path("/home/user/Desktop/unlv")
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs/config.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import ProteinDataset, collate_fn
from models.fusion_v2 import FusionModelV2
from models.esm2_hierarchical import ESM2Hierarchical

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5


@torch.no_grad()
def get_per_sample_f1(model, loader, model_name, n_l4):
    """각 샘플의 L4 multi-label F1 반환 (shape: N,)"""
    model.eval()
    all_probs  = []
    all_labels = []

    for batch in tqdm(loader, desc=f"{model_name} 추론"):
        esm_emb, cmap, _, _, _, l4_mh, _ = batch
        esm_emb = esm_emb.to(DEVICE)
        cmap    = cmap.to(DEVICE)
        logits  = model(esm_emb, cmap)
        probs   = torch.sigmoid(logits[3]).cpu().numpy()  # L4
        all_probs.append(probs)
        all_labels.append(l4_mh.numpy())

    probs  = np.concatenate(all_probs,  axis=0)   # (N, n_l4)
    labels = np.concatenate(all_labels, axis=0)   # (N, n_l4)

    preds = (probs >= THRESHOLD).astype(np.float32)

    # 샘플별 F1: 각 샘플에 대해 F1을 계산
    per_sample_f1 = []
    for i in range(len(preds)):
        tp = (preds[i] * labels[i]).sum()
        fp = (preds[i] * (1 - labels[i])).sum()
        fn = ((1 - preds[i]) * labels[i]).sum()
        if tp + fp + fn == 0:
            per_sample_f1.append(1.0)   # 둘 다 예측 없고 정답 없음 → 완벽
        else:
            p = tp / (tp + fp + 1e-8)
            r = tp / (tp + fn + 1e-8)
            f = 2 * p * r / (p + r + 1e-8)
            per_sample_f1.append(float(f))

    return np.array(per_sample_f1), probs, labels


def main():
    print(f"Device: {DEVICE}")

    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    n_l4 = n_classes[3]

    # 테스트셋 데이터로더
    test_ds = ProteinDataset(
        ids_file      = str(ROOT / CFG["paths"]["splits_dir"] / "test_ids.txt"),
        meta_csv      = str(ROOT / CFG["paths"]["meta_csv"]),
        embed_dir     = ROOT / CFG["paths"]["embed_dir"],
        cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
        label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
    )
    loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)
    print(f"테스트셋: {len(test_ds):,}개")

    # ── V2 추론 ───────────────────────────────────────────────
    ckpt_v2 = torch.load(ROOT / "outputs/checkpoints/fusion_v2_ml_best.pt",
                         map_location=DEVICE, weights_only=False)
    model_v2 = FusionModelV2(n_classes,
                              esm_dim=CFG["model"]["esm2_dim"],
                              contact_dim=CFG["model"]["resnet_out_dim"],
                              fusion_dim=CFG["model"]["fusion_dim"],
                              dropout=0.0).to(DEVICE)
    model_v2.load_state_dict(ckpt_v2["model"])
    f1_v2, probs_v2, labels = get_per_sample_f1(model_v2, loader, "V2", n_l4)
    del model_v2
    torch.cuda.empty_cache()
    print(f"V2 mean sample F1: {f1_v2.mean():.4f}")

    # ── B2 추론 ───────────────────────────────────────────────
    ckpt_b2 = torch.load(ROOT / "outputs/checkpoints/b2_ml_best.pt",
                         map_location=DEVICE, weights_only=False)
    model_b2 = ESM2Hierarchical(n_classes,
                                 esm_dim=CFG["model"]["esm2_dim"],
                                 dropout=0.0).to(DEVICE)
    model_b2.load_state_dict(ckpt_b2["model"])
    f1_b2, probs_b2, _ = get_per_sample_f1(model_b2, loader, "B2", n_l4)
    del model_b2
    torch.cuda.empty_cache()
    print(f"B2 mean sample F1: {f1_b2.mean():.4f}")

    # ── Wilcoxon signed-rank test ─────────────────────────────
    diff = f1_v2 - f1_b2
    n_better = (diff > 0).sum()
    n_worse  = (diff < 0).sum()
    n_tie    = (diff == 0).sum()
    print(f"\nV2 better: {n_better}, B2 better: {n_worse}, Tie: {n_tie}")

    stat, p_value = wilcoxon(f1_v2, f1_b2, alternative="greater")
    print(f"Wilcoxon stat={stat:.2f}, p={p_value:.2e}")

    # effect size (rank-biserial correlation)
    n = len(diff[diff != 0])
    r = 1 - (2 * stat) / (n * (n + 1) / 2) if n > 0 else 0.0

    # micro F1 전체
    preds_v2 = (probs_v2 >= THRESHOLD).astype(np.float32)
    preds_b2 = (probs_b2 >= THRESHOLD).astype(np.float32)

    def micro_f1(preds, labels):
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        return float(2 * p * r / (p + r + 1e-8))

    mf1_v2 = micro_f1(preds_v2, labels)
    mf1_b2 = micro_f1(preds_b2, labels)

    result = {
        "test": "Wilcoxon signed-rank test (V2 > B2, one-sided)",
        "n_samples": int(len(f1_v2)),
        "V2_micro_f1": round(mf1_v2, 4),
        "B2_micro_f1": round(mf1_b2, 4),
        "V2_mean_sample_f1": round(float(f1_v2.mean()), 4),
        "B2_mean_sample_f1": round(float(f1_b2.mean()), 4),
        "n_V2_better": int(n_better),
        "n_B2_better": int(n_worse),
        "n_tie": int(n_tie),
        "wilcoxon_statistic": float(stat),
        "p_value": float(p_value),
        "effect_size_r": round(float(r), 4),
        "significant": bool(p_value < 0.05),
        "interpretation": (
            "V2 (ESM-2 + Contact Map)의 샘플별 F1이 B2 (서열만)보다 "
            f"통계적으로 유의미하게 높음 (p={p_value:.2e}, r={r:.3f})"
            if p_value < 0.05 else
            "통계적으로 유의미한 차이 없음"
        )
    }

    out_path = ROOT / "outputs/results/wilcoxon_v2_vs_b2.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
