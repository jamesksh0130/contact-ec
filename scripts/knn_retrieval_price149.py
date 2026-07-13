"""
kNN Retrieval Inference for Price-149 (OOD benchmark)

전략: V2 fused embedding (1024-dim) 공간에서 훈련셋 nearest neighbor를 찾아 EC 결정.
CLEAN-Contact의 contrastive kNN과 동일한 원리.

Steps:
  1. V2 모델로 훈련셋 전체 fused embedding 추출 (216k × 1024)
  2. Price-149 fused embedding 추출 (144 × 1024)
  3. cosine similarity top-K → multi-hot EC majority vote
  4. F1 계산 및 저장
"""
import os, sys, pickle, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

ROOT = Path("/home/user/Desktop/unlv")
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs/config.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import ProteinDataset, collate_fn
from models.fusion_v2 import FusionModelV2

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPT   = ROOT / "outputs/checkpoints/fusion_v2_ml_best.pt"
K_LIST = [1, 3, 5, 10, 20]   # 평가할 K 값들


# ── fused embedding 추출 함수 ─────────────────────────────────
@torch.no_grad()
def extract_fused(model, loader):
    """V2 forward에서 GCA fused (1024-d) 벡터를 hook으로 추출."""
    embeddings = []
    accessions = []

    # forward hook 등록
    fused_cache = []
    def _hook(module, input, output):
        fused_cache.append(output.cpu())

    # gate 이후의 fused 계산 부분을 hook으로 캡처
    # FusionModelV2.forward에서 fused = esm_proj + gate * attn_out 후 head 전달
    # head의 입력 = fused → head 모듈의 forward input 캡처
    hook = model.head.register_forward_pre_hook(
        lambda m, inp: fused_cache.append(inp[0].detach().cpu())
    )

    model.eval()
    for batch in tqdm(loader, desc="fused embedding 추출"):
        esm_emb, cmap, _, _, _, _, acc = batch
        esm_emb = esm_emb.to(DEVICE)
        cmap    = cmap.to(DEVICE)
        _ = model(esm_emb, cmap)
        accessions.extend(list(acc))

    hook.remove()

    embeddings = torch.cat(fused_cache, dim=0)  # (N, 1024)
    return embeddings, accessions


def main():
    print(f"Device: {DEVICE}")

    # 라벨 인코더
    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    print(f"L4 classes: {n_classes[3]}")

    # 모델 로드
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = FusionModelV2(n_classes,
                          esm_dim=CFG["model"]["esm2_dim"],
                          contact_dim=CFG["model"]["resnet_out_dim"],
                          fusion_dim=CFG["model"]["fusion_dim"],
                          dropout=0.0).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"체크포인트 로드: val_f1={ckpt.get('micro_f1', 'N/A'):.4f}")

    # ── Step 1: 훈련셋 fused embedding ───────────────────────
    train_cache = ROOT / "outputs/results/train_fused_embeddings.pt"

    if train_cache.exists():
        print(f"캐시 로드: {train_cache}")
        saved = torch.load(train_cache, weights_only=False)
        train_embs = saved["embeddings"]   # (N_train, 1024)
        train_accs = saved["accessions"]
    else:
        print("훈련셋 fused embedding 추출 중 (216k 단백질)...")
        train_ds = ProteinDataset(
            ids_file      = str(ROOT / CFG["paths"]["splits_dir"] / "train_ids.txt"),
            meta_csv      = str(ROOT / CFG["paths"]["meta_csv"]),
            embed_dir     = ROOT / CFG["paths"]["embed_dir"],
            cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
            label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
        )
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=False,
                                  num_workers=4, collate_fn=collate_fn, pin_memory=True)
        train_embs, train_accs = extract_fused(model, train_loader)
        torch.save({"embeddings": train_embs, "accessions": train_accs}, train_cache)
        print(f"훈련셋 저장: {train_embs.shape}")

    # 훈련셋 L4 multi-hot 라벨 구성
    import pandas as pd
    meta = pd.read_csv(ROOT / CFG["paths"]["meta_csv"])
    acc2mh = {}
    n_l4 = n_classes[3]
    for _, row in meta.iterrows():
        uid = row["accession"]
        idxs_str = str(row.get("l4_all_idxs", ""))
        if idxs_str and idxs_str != "nan":
            idxs = [int(x) for x in idxs_str.split("|") if x.strip().isdigit()]
            mh = np.zeros(n_l4, dtype=np.float32)
            for idx in idxs:
                if 0 <= idx < n_l4:
                    mh[idx] = 1.0
            acc2mh[uid] = mh

    # 훈련셋 multi-hot 행렬 구성 (accession 순서에 맞춰)
    train_mh = np.zeros((len(train_accs), n_l4), dtype=np.float32)
    for i, acc in enumerate(train_accs):
        if acc in acc2mh:
            train_mh[i] = acc2mh[acc]
    train_mh_t = torch.from_numpy(train_mh)  # (N_train, n_l4)
    print(f"훈련셋 multi-hot: {train_mh_t.shape}")

    # ── Step 2: Price-149 fused embedding ────────────────────
    print("\nPrice-149 fused embedding 추출 중...")
    price_ds = ProteinDataset(
        ids_file      = str(ROOT / "data/splits/price149_ids.txt"),
        meta_csv      = str(ROOT / "data/processed/price149_meta.csv"),
        embed_dir     = ROOT / CFG["paths"]["embed_dir"],
        cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
        label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
    )
    price_loader = DataLoader(price_ds, batch_size=64, shuffle=False,
                              num_workers=0, collate_fn=collate_fn, pin_memory=False)
    price_embs, price_accs = extract_fused(model, price_loader)
    print(f"Price-149 embedding: {price_embs.shape}")

    # Price-149 정답 라벨
    price_meta = pd.read_csv(ROOT / "data/processed/price149_meta.csv")
    price_mh = np.zeros((len(price_accs), n_l4), dtype=np.float32)
    acc_to_idx = {acc: i for i, acc in enumerate(price_accs)}
    for _, row in price_meta.iterrows():
        uid = row["accession"]
        if uid not in acc_to_idx:
            continue
        idxs_str = str(row.get("l4_all_idxs", ""))
        if idxs_str and idxs_str != "nan":
            idxs = [int(x) for x in idxs_str.split("|") if x.strip().isdigit()]
            for idx in idxs:
                if 0 <= idx < n_l4:
                    price_mh[acc_to_idx[uid]][idx] = 1.0
    price_mh_t = torch.from_numpy(price_mh)

    # ── Step 3: kNN 추론 ──────────────────────────────────────
    print("\nkNN 추론 중...")
    # cosine similarity 계산
    train_norm = F.normalize(train_embs, dim=1)   # (N_train, 1024)
    price_norm = F.normalize(price_embs, dim=1)   # (N_price, 1024)

    # GPU로 유사도 계산 (대용량)
    results = {}
    chunk = 512
    sim_matrix = []
    for i in range(0, len(price_norm), chunk):
        q = price_norm[i:i+chunk].to(DEVICE)
        s = train_norm.to(DEVICE)
        sim = torch.mm(q, s.T)   # (chunk, N_train)
        sim_matrix.append(sim.cpu())
    sim_matrix = torch.cat(sim_matrix, dim=0)   # (N_price, N_train)
    print(f"유사도 행렬: {sim_matrix.shape}")

    for K in K_LIST:
        topk_vals, topk_idx = sim_matrix.topk(K, dim=1)  # (N_price, K)

        # K개 이웃의 multi-hot 평균 → threshold=0.5로 예측
        neighbor_mh = train_mh_t[topk_idx]    # (N_price, K, n_l4)
        pred_score  = neighbor_mh.mean(dim=1)  # (N_price, n_l4)
        pred_mh     = (pred_score >= 0.5).float()

        # micro F1 계산
        tp = (pred_mh * price_mh_t).sum().item()
        fp = (pred_mh * (1 - price_mh_t)).sum().item()
        fn = ((1 - pred_mh) * price_mh_t).sum().item()
        prec   = tp / (tp + fp + 1e-8)
        rec    = tp / (tp + fn + 1e-8)
        f1     = 2 * prec * rec / (prec + rec + 1e-8)
        results[f"k{K}"] = {"precision": round(prec, 4),
                             "recall": round(rec, 4),
                             "micro_f1": round(f1, 4)}
        print(f"  K={K:2d}: Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

    # ── Step 4: 결과 저장 ─────────────────────────────────────
    out = {
        "model": "fusion_v2_knn_retrieval",
        "benchmark": "Price-149",
        "method": "cosine kNN on GCA fused embedding (1024-d)",
        "n_train": len(train_accs),
        "n_test": len(price_accs),
        "CLEAN_Contact_F1": 0.525,
        "results_by_K": results,
    }
    out_path = ROOT / "outputs/results/fusion_v2_price149_knn_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n결과 저장: {out_path}")

    best_k = max(results, key=lambda k: results[k]["micro_f1"])
    best_f1 = results[best_k]["micro_f1"]
    print(f"\n최고 성능: {best_k} → F1={best_f1:.4f}  (CLEAN-Contact: 0.525)")


if __name__ == "__main__":
    main()
