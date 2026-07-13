"""
전체 테스트셋(N=124) 재평가 스크립트
- test_meta_full.csv + test_ids_full.txt 사용
- 4개 모델(B1, B2, B3, Contact-EC) 모두 평가
- 고정 threshold=0.5 사용 (ecbench_eval.py와 동일 프로토콜)
"""
import sys, argparse, pickle, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config_ecbench.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import ProteinDataset, collate_fn

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODELS = {
    "b1_esm2_fc":        "outputs/checkpoints/ecbench_b1_best.pt",
    "b2_esm2_hier":      "outputs/checkpoints/ecbench_b2_hard_val_best.pt",
    "b3_contact":        "outputs/checkpoints/ecbench_b3_phase1_best.pt",
    "fusion_v2":         "outputs/checkpoints/ecbench_fv2_phase2_best.pt",
    "fusion_v2_flatfc":  "outputs/checkpoints/ecbench_b4_flatfc_best.pt",
    # "fusion_v2_3b":    "outputs/checkpoints/ecbench_3b_fv2_best.pt",  # 학습 완료 후 활성화
}


def build_model(name, n_classes):
    if name == "b1_esm2_fc":
        from models.esm2_fc import ESM2FC
        return ESM2FC(n_classes, esm_dim=CFG["model"]["esm2_dim"], dropout=0.0)
    elif name == "b2_esm2_hier":
        from models.esm2_hierarchical import ESM2Hierarchical
        return ESM2Hierarchical(n_classes, esm_dim=CFG["model"]["esm2_dim"], dropout=0.0)
    elif name == "b3_contact":
        from models.contact_resnet import ContactResNet
        return ContactResNet(n_classes, dropout=0.0)
    elif name == "fusion_v2":
        from models.fusion_v2 import FusionModelV2
        return FusionModelV2(n_classes,
                             esm_dim=CFG["model"]["esm2_dim"],
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=0.0)
    elif name == "fusion_v2_flatfc":
        from models.fusion_v2_flatfc import FusionV2FlatFC
        return FusionV2FlatFC(n_classes,
                              esm_dim=CFG["model"]["esm2_dim"],
                              contact_dim=CFG["model"]["resnet_out_dim"],
                              fusion_dim=CFG["model"]["fusion_dim"],
                              dropout=0.0)
    elif name == "fusion_v2_3b":
        from models.fusion_v2 import FusionModelV2
        return FusionModelV2(n_classes,
                             esm_dim=2560,
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=0.0)


@torch.no_grad()
def run_inference(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="  추론", leave=False):
        esm_emb, cmap, _, _, _, l4_mh, _ = batch
        logits = model(esm_emb.to(DEVICE), cmap.to(DEVICE))
        all_probs.append(torch.sigmoid(logits[3]).cpu().numpy())
        all_labels.append(l4_mh.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(probs, labels, train_rare_mask, model_name, split):
    preds = (probs >= 0.5).astype(np.int32)
    micro  = f1_score(labels, preds, average="micro",    zero_division=0)
    wtd    = f1_score(labels, preds, average="weighted", zero_division=0)
    macro  = f1_score(labels, preds, average="macro",    zero_division=0)
    prec   = precision_score(labels, preds, average="micro", zero_division=0)
    rec    = recall_score(labels, preds, average="micro",    zero_division=0)
    rare   = f1_score(labels[:, train_rare_mask], preds[:, train_rare_mask],
                      average="micro", zero_division=0) if train_rare_mask.sum() > 0 else float("nan")

    print(f"\n  [{model_name}]  {split}  N={labels.shape[0]}")
    print(f"    Micro F1    : {micro:.4f}")
    print(f"    Weighted F1 : {wtd:.4f}")
    print(f"    Macro F1    : {macro:.4f}")
    print(f"    Precision   : {prec:.4f}  Recall: {rec:.4f}")
    print(f"    Rare-EC F1  : {rare:.4f}  ({train_rare_mask.sum()} classes)")

    return {
        "model": model_name, "split": split, "n": int(labels.shape[0]),
        "micro_f1": round(float(micro), 4),
        "weighted_f1": round(float(wtd), 4),
        "macro_f1": round(float(macro), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "rare_ec_f1": round(float(rare), 4) if not np.isnan(rare) else None,
    }


def main():
    import pandas as pd

    # 라벨 인코더
    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    n_l4 = n_classes[3]
    print(f"Level-4 클래스 수: {n_l4}")

    # 희귀 클래스 마스크
    train_meta = ROOT / "data" / "ecbench" / "processed" / "train_meta.csv"
    train_ids_f = ROOT / CFG["paths"]["splits_dir"] / "train_ids.txt"
    train_ids = set(train_ids_f.read_text().strip().split())
    meta_df   = pd.read_csv(train_meta)
    train_df  = meta_df[meta_df["accession"].isin(train_ids)]
    counts    = np.zeros(n_l4, dtype=np.int64)
    for idx in train_df["l4_idx"].dropna().astype(int):
        if 0 <= idx < n_l4:
            counts[idx] += 1
    rare_mask = counts <= 25
    print(f"희귀 클래스 (N≤25): {rare_mask.sum()}/{n_l4}")

    # 데이터 로더
    def make_loader(ids_file, meta_csv, bs=256):
        ds = ProteinDataset(
            ids_file=str(ids_file),
            meta_csv=str(meta_csv),
            embed_dir=ROOT / CFG["paths"]["embed_dir"],
            cmap_dir=ROOT / CFG["paths"]["cmap_dir"],
            label_enc_pkl=ROOT / CFG["paths"]["label_enc"],
        )
        return DataLoader(ds, batch_size=bs, shuffle=False,
                          num_workers=4, collate_fn=collate_fn, pin_memory=True)

    splits = ROOT / CFG["paths"]["splits_dir"]
    proc   = ROOT / "data" / "ecbench" / "processed"

    loaders = {
        "Swiss-Prot 2023-01 (full N=124)": make_loader(
            splits / "test_ids_full.txt", proc / "test_meta_full.csv"),
        "Swiss-Prot 2023-01 (prev N=101)": make_loader(
            splits / "test_ids.txt", proc / "test_meta.csv"),
        "Price-149": make_loader(
            splits / "price149_ids.txt", proc / "price149_meta.csv"),
    }

    all_results = {}
    print("=" * 65)
    for model_name, ckpt_path in MODELS.items():
        ckpt = torch.load(ROOT / ckpt_path, map_location=DEVICE, weights_only=False)
        model = build_model(model_name, n_classes).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        val_f1 = ckpt.get("micro_f1", "?")
        print(f"\n{'='*65}")
        print(f"  모델: {model_name}  (val micro_f1={val_f1})")
        print(f"{'='*65}")

        results = []
        for split_name, loader in loaders.items():
            probs, labels = run_inference(model, loader)
            r = compute_metrics(probs, labels, rare_mask, model_name, split_name)
            results.append(r)

        all_results[model_name] = results

    # 결과 저장
    out = ROOT / "outputs" / "results" / "eval_full_testset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out}")

    # 요약 테이블
    print(f"\n{'='*65}")
    print("요약 테이블 (Micro F1)")
    print(f"{'='*65}")
    header = f"{'모델':<18}{'N=124 (full)':<16}{'N=101 (prev)':<16}{'Price-149':<12}"
    print(header)
    print("-" * 65)
    for model_name, results in all_results.items():
        r_full = next((r for r in results if "full" in r["split"]), {})
        r_prev = next((r for r in results if "prev" in r["split"]), {})
        r_p149 = next((r for r in results if "Price" in r["split"]), {})
        print(f"{model_name:<18}"
              f"{r_full.get('micro_f1','?'):<16}"
              f"{r_prev.get('micro_f1','?'):<16}"
              f"{r_p149.get('micro_f1','?'):<12}")


if __name__ == "__main__":
    main()
