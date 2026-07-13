"""
EC-Bench 공식 평가 스크립트

두 테스트셋 평가:
  1. Swiss-Prot 2023-01 (468개, 공식 EC-Bench 테스트)
  2. Price-149 (149개, OOD 박테리아 단백질)

EC-Bench 논문의 기존 방법 F1 수치와 직접 비교 가능.

사용법:
  python scripts/ecbench_eval.py \
    --checkpoint outputs/checkpoints/ecbench_fusion_v2_phase1_best.pt \
    --model fusion_v2 \
    [--threshold 0.5]
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
_cfg_file = "config_ecbench.yaml"
for _i, _a in enumerate(sys.argv):
    if _a == "--config" and _i + 1 < len(sys.argv):
        _cfg_file = sys.argv[_i + 1]
        break
with open(ROOT / "configs" / _cfg_file) as f:
    CFG = yaml.safe_load(f)

from models.dataset import ProteinDataset, collate_fn

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def build_model(model_name, n_classes):
    if model_name == "b2_esm2_hier":
        from models.esm2_hierarchical import ESM2Hierarchical
        return ESM2Hierarchical(n_classes, esm_dim=CFG["model"]["esm2_dim"], dropout=0.0)
    elif model_name in ("b1_esm2_fc", "b1"):
        from models.esm2_fc import ESM2FC
        return ESM2FC(n_classes, esm_dim=CFG["model"]["esm2_dim"], dropout=0.0)
    elif model_name == "fusion_v2":
        from models.fusion_v2 import FusionModelV2
        return FusionModelV2(n_classes,
                             esm_dim=CFG["model"]["esm2_dim"],
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=0.0)
    elif model_name == "b3_contact":
        from models.contact_resnet import ContactResNet
        return ContactResNet(n_classes, dropout=0.0)
    elif model_name == "fusion_v2_flatfc":
        from models.fusion_v2_flatfc import FusionV2FlatFC
        return FusionV2FlatFC(n_classes,
                              esm_dim=CFG["model"]["esm2_dim"],
                              contact_dim=CFG["model"]["resnet_out_dim"],
                              fusion_dim=CFG["model"]["fusion_dim"],
                              dropout=0.0)
    elif model_name == "fusion_v2_3b":
        from models.fusion_v2 import FusionModelV2
        return FusionModelV2(n_classes,
                             esm_dim=CFG["model"].get("esm2_dim_3b", 2560),
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=0.0)
    else:
        raise ValueError(f"미지원 모델: {model_name}")


@torch.no_grad()
def run_inference(model, loader, n_l4, threshold):
    model.eval()
    all_probs  = []
    all_labels = []
    all_accs   = []

    for batch in tqdm(loader, desc="추론", leave=False):
        esm_emb, cmap, _, _, _, l4_mh, accs = batch
        esm_emb = esm_emb.to(DEVICE)
        cmap    = cmap.to(DEVICE)

        logits  = model(esm_emb, cmap)
        probs   = torch.sigmoid(logits[3]).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(l4_mh.numpy())
        all_accs.extend(accs)

    probs  = np.concatenate(all_probs,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return probs, labels, all_accs


def compute_metrics(probs, labels, threshold, encoder, split_name,
                    train_rare_mask=None):
    preds = (probs >= threshold).astype(np.int32)
    n     = labels.shape[0]

    micro_f1     = f1_score(labels, preds, average="micro",    zero_division=0)
    macro_f1     = f1_score(labels, preds, average="macro",    zero_division=0)
    weighted_f1  = f1_score(labels, preds, average="weighted", zero_division=0)
    micro_pre    = precision_score(labels, preds, average="micro", zero_division=0)
    micro_rec    = recall_score(labels, preds, average="micro",    zero_division=0)

    # 희귀 클래스 (≤25 train 샘플) F1 — 훈련 세트 기준 (HIT-EC 방식)
    if train_rare_mask is not None:
        rare_mask = train_rare_mask
    else:
        rare_mask = labels.sum(axis=0) <= 25   # fallback: 테스트셋 기준
    if rare_mask.sum() > 0:
        rare_f1 = f1_score(labels[:, rare_mask], preds[:, rare_mask],
                           average="micro", zero_division=0)
    else:
        rare_f1 = float("nan")

    print(f"\n{'='*55}")
    print(f"  [{split_name}]  N={n}")
    print(f"  Micro F1    : {micro_f1:.4f}   ← HIT-EC 비교용")
    print(f"  Weighted F1 : {weighted_f1:.4f}   ← EC-Bench 비교용")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"  Precision   : {micro_pre:.4f}")
    print(f"  Recall      : {micro_rec:.4f}")
    print(f"  Rare-EC F1  : {rare_f1:.4f}  (N≤25 train, {rare_mask.sum()}개 클래스)")
    print(f"{'='*55}")

    return {
        "split":        split_name,
        "n_samples":    n,
        "micro_f1":     round(float(micro_f1),    4),
        "weighted_f1":  round(float(weighted_f1),  4),
        "macro_f1":     round(float(macro_f1),    4),
        "precision":    round(float(micro_pre),   4),
        "recall":       round(float(micro_rec),   4),
        "rare_ec_f1":   round(float(rare_f1),     4) if not np.isnan(rare_f1) else None,
        "rare_class_count": int(rare_mask.sum()),
        "threshold":    threshold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model",      default="fusion_v2",
                        choices=["b1_esm2_fc", "b2_esm2_hier", "fusion_v2",
                                 "b3_contact", "fusion_v2_flatfc", "fusion_v2_3b"])
    parser.add_argument("--config",     default="config_ecbench.yaml",
                        help="config 파일명 (configs/ 디렉토리 기준)")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--val_hard",   action="store_true",
                        help="val_hard 평가 추가 실행")
    args = parser.parse_args()

    # 라벨 인코더
    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    n_l4      = n_classes[3]
    print(f"클래스 수: {n_classes}")

    # 체크포인트 로드
    ckpt  = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model = build_model(args.model, n_classes).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"체크포인트 로드: {args.checkpoint}")
    if "micro_f1" in ckpt:
        print(f"  (val micro_f1={ckpt['micro_f1']:.4f}, epoch={ckpt.get('epoch','?')})")

    # ── 훈련셋 기준 희귀 클래스 마스크 (HIT-EC 방식: N≤25 in training) ──
    import pandas as pd
    train_rare_mask = None
    train_meta_path = ROOT / "data" / "processed" / "dataset_meta.csv"
    train_ids_path  = ROOT / "data" / "splits" / "train_ids.txt"
    if train_meta_path.exists() and train_ids_path.exists():
        train_ids = set(open(train_ids_path).read().split())
        meta_df   = pd.read_csv(train_meta_path)
        train_df  = meta_df[meta_df['accession'].isin(train_ids)]
        counts    = np.zeros(n_l4, dtype=np.int64)
        for idx in train_df['l4_idx'].dropna().astype(int):
            if 0 <= idx < n_l4:
                counts[idx] += 1
        train_rare_mask = counts <= 25
        n_rare = train_rare_mask.sum()
        print(f"훈련셋 N≤25 희귀 클래스: {n_rare}/{n_l4} "
              f"({n_rare/n_l4:.1%}) — HIT-EC 방식")

    results = {}

    # ── 0. val_hard (선택적) ─────────────────────────────────
    if args.val_hard:
        vh_splits = ROOT / CFG["paths"]["splits_dir"] / "val_hard_ids.txt"
        vh_meta   = ROOT / "data" / "ecbench" / "processed" / "train_meta.csv"
        if vh_splits.exists():
            ds_vh = ProteinDataset(
                ids_file      = str(vh_splits),
                meta_csv      = str(vh_meta),
                embed_dir     = ROOT / CFG["paths"]["embed_dir"],
                cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
                label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
            )
            loader_vh = DataLoader(ds_vh, batch_size=args.batch_size, shuffle=False,
                                   num_workers=4, collate_fn=collate_fn, pin_memory=True)
            probs_vh, labels_vh, _ = run_inference(model, loader_vh, n_l4, args.threshold)
            results["val_hard"] = compute_metrics(
                probs_vh, labels_vh, args.threshold, encoders, "val_hard (≤30% id)",
                train_rare_mask=train_rare_mask
            )
        else:
            print(f"val_hard split 파일 없음: {vh_splits}")

    # ── 1. EC-Bench 공식 테스트 (Swiss-Prot 2023-01) ─────────
    test_splits = ROOT / CFG["paths"]["splits_dir"] / "test_ids.txt"
    if test_splits.exists():
        ds = ProteinDataset(
            ids_file      = str(test_splits),
            meta_csv      = str(ROOT / "data/ecbench/processed/test_meta.csv"),
            embed_dir     = ROOT / CFG["paths"]["embed_dir"],
            cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
            label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)
        probs, labels, accs = run_inference(model, loader, n_l4, args.threshold)
        results["swissprot_2023"] = compute_metrics(
            probs, labels, args.threshold, encoders, "Swiss-Prot 2023-01",
            train_rare_mask=train_rare_mask
        )
    else:
        print(f"테스트셋 split 파일 없음: {test_splits}")

    # ── 2. Price-149 ─────────────────────────────────────────
    p149_splits = ROOT / CFG["paths"]["splits_dir"] / "price149_ids.txt"
    if p149_splits.exists():
        ds149 = ProteinDataset(
            ids_file      = str(p149_splits),
            meta_csv      = str(ROOT / "data/ecbench/processed/price149_meta.csv"),
            embed_dir     = ROOT / CFG["paths"]["embed_dir"],
            cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
            label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
        )
        loader149 = DataLoader(ds149, batch_size=args.batch_size, shuffle=False,
                               num_workers=4, collate_fn=collate_fn, pin_memory=True)
        probs149, labels149, accs149 = run_inference(model, loader149, n_l4, args.threshold)
        results["price149"] = compute_metrics(
            probs149, labels149, args.threshold, encoders, "Price-149 (OOD)",
            train_rare_mask=train_rare_mask
        )

    # ── 결과 저장 ──────────────────────────────────────────────
    ckpt_stem = Path(args.checkpoint).stem
    out_path  = ROOT / CFG["paths"]["result_dir"] / f"ecbench_eval_{ckpt_stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"checkpoint": args.checkpoint,
                   "model": args.model,
                   "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")

    # ── 참고 수치 (비교 대상 논문, 프로토콜 상이 주의) ─────────
    print("\n── 참고 수치 ── (★ 프로토콜 다름, 직접 비교 불가)")
    print("  [HIT-EC cross-val, Swiss-Prot+PDB ~200K, micro F1]")
    hitec_ref = {
        "HIT-EC":    0.93, "CLEAN": 0.88,
        "ECPICK":    0.82, "DeepECT": 0.79,
    }
    for name, f1 in hitec_ref.items():
        print(f"    {name:<12} micro_f1={f1:.3f}")
    print("  [CLEAN, New-392/Price-149, per-protein F1]")
    clean_ref = {
        "CLEAN-Contact (New-392)":  0.566,
        "CLEAN (New-392)":          0.504,
        "CLEAN-Contact (Price-149)":0.525,
        "CLEAN (Price-149)":        0.452,
    }
    for name, f1 in clean_ref.items():
        print(f"    {name:<30} F1={f1:.3f}")
    if "swissprot_2023" in results:
        r = results["swissprot_2023"]
        print(f"\n  [Ours — EC-Bench temporal split]")
        print(f"    Micro F1    = {r['micro_f1']:.4f}")
        print(f"    Weighted F1 = {r['weighted_f1']:.4f}")
        print(f"    Macro F1    = {r['macro_f1']:.4f}")
        if r.get('rare_ec_f1'):
            print(f"    Rare-EC F1  = {r['rare_ec_f1']:.4f}  (N≤25 train)")


if __name__ == "__main__":
    main()
