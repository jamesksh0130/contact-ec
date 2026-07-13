"""
Per-class threshold tuning (EC-Bench "learnt" 방식)

val_hard 셋에서 클래스별 최적 threshold를 탐색한 뒤
공식 테스트셋(Swiss-Prot 2023-01)과 Price-149에 적용.

사용법:
  python scripts/tune_threshold.py \
    --checkpoint outputs/checkpoints/ecbench_fv2_phase2_best.pt \
    --model fusion_v2

EC-Bench 논문의 "learnt" 모델 (per-class threshold) vs.
"regular" 모델 (고정 threshold=0.5) 비교.
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


# ── 모델 생성 ─────────────────────────────────────────────────────────────────
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
    else:
        raise ValueError(f"미지원 모델: {model_name}")


# ── 추론 ──────────────────────────────────────────────────────────────────────
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


# ── 클래스별 threshold 탐색 ───────────────────────────────────────────────────
def tune_thresholds(probs, labels, n_grid=51):
    """val_hard에서 각 Level-4 클래스의 F1을 최대화하는 threshold 탐색."""
    n_classes = probs.shape[1]
    thresholds = np.full(n_classes, 0.5)          # default fallback
    grid = np.linspace(0.05, 0.95, n_grid)

    pos_classes = 0
    for c in tqdm(range(n_classes), desc="  threshold 탐색", leave=False):
        y = labels[:, c]
        if y.sum() == 0:                          # 양성 샘플 없으면 0.5 유지
            continue
        pos_classes += 1
        p = probs[:, c]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t

    print(f"  양성 클래스 {pos_classes}/{n_classes}개에 대해 threshold 탐색 완료")
    return thresholds


# ── 평가 ──────────────────────────────────────────────────────────────────────
def evaluate(probs, labels, thresholds, split_name, train_rare_mask=None):
    preds = (probs >= thresholds[np.newaxis, :]).astype(np.int32)

    micro_f1    = f1_score(labels, preds, average="micro",    zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    macro_f1    = f1_score(labels, preds, average="macro",    zero_division=0)
    micro_pre   = precision_score(labels, preds, average="micro", zero_division=0)
    micro_rec   = recall_score(labels, preds, average="micro",    zero_division=0)

    rare_f1 = float("nan")
    if train_rare_mask is not None and train_rare_mask.sum() > 0:
        rare_f1 = f1_score(labels[:, train_rare_mask],
                           preds[:, train_rare_mask],
                           average="micro", zero_division=0)

    print(f"\n{'='*55}")
    print(f"  [{split_name}]  N={labels.shape[0]}  (per-class threshold)")
    print(f"  Micro F1    : {micro_f1:.4f}")
    print(f"  Weighted F1 : {weighted_f1:.4f}")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"  Precision   : {micro_pre:.4f}")
    print(f"  Recall      : {micro_rec:.4f}")
    if not np.isnan(rare_f1):
        print(f"  Rare-EC F1  : {rare_f1:.4f}  (N≤25 train, {train_rare_mask.sum()}개)")
    print(f"{'='*55}")

    return {
        "split": split_name, "n_samples": int(labels.shape[0]),
        "micro_f1": round(float(micro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "precision": round(float(micro_pre), 4),
        "recall": round(float(micro_rec), 4),
        "rare_ec_f1": round(float(rare_f1), 4) if not np.isnan(rare_f1) else None,
        "method": "per_class_threshold",
    }


def make_loader(ids_file, meta_csv, batch_size):
    ds = ProteinDataset(
        ids_file=str(ids_file),
        meta_csv=str(meta_csv),
        embed_dir=ROOT / CFG["paths"]["embed_dir"],
        cmap_dir=ROOT / CFG["paths"]["cmap_dir"],
        label_enc_pkl=ROOT / CFG["paths"]["label_enc"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, collate_fn=collate_fn, pin_memory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="fusion_v2",
                        choices=["b1_esm2_fc", "b2_esm2_hier", "fusion_v2", "b3_contact"])
    parser.add_argument("--n_grid",     type=int, default=51,
                        help="threshold 탐색 grid 크기 (기본 51 = 0.05~0.95 간격 0.018)")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    splits_dir   = ROOT / CFG["paths"]["splits_dir"]
    proc_dir     = ROOT / "data" / "ecbench" / "processed"

    # 라벨 인코더
    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    n_l4 = n_classes[3]
    print(f"클래스 수: {n_classes}")

    # 체크포인트 로드
    ckpt  = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model = build_model(args.model, n_classes).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"체크포인트 로드: {args.checkpoint}")
    if "micro_f1" in ckpt:
        print(f"  (val micro_f1={ckpt['micro_f1']:.4f}, epoch={ckpt.get('epoch','?')})")

    # 훈련셋 희귀 클래스 마스크
    import pandas as pd
    train_rare_mask = None
    train_meta = proc_dir / "train_meta.csv"
    train_ids_f = splits_dir / "train_ids.txt"
    if train_meta.exists() and train_ids_f.exists():
        train_ids = set(train_ids_f.read_text().strip().split("\n"))
        meta_df   = pd.read_csv(train_meta)
        train_df  = meta_df[meta_df["accession"].isin(train_ids)]
        counts    = np.zeros(n_l4, dtype=np.int64)
        for idx in train_df["l4_idx"].dropna().astype(int):
            if 0 <= idx < n_l4:
                counts[idx] += 1
        train_rare_mask = counts <= 25
        print(f"훈련셋 N≤25 희귀 클래스: {train_rare_mask.sum()}/{n_l4} "
              f"({train_rare_mask.mean():.1%})")

    # ── Step 1: val_hard에서 threshold 탐색 ───────────────────────────────────
    val_hard_ids = splits_dir / "val_hard_ids.txt"
    val_meta     = proc_dir / "train_meta.csv"   # train_meta에 val_hard IDs 포함
    if not val_hard_ids.exists():
        raise FileNotFoundError(f"val_hard_ids.txt 없음: {val_hard_ids}")

    print("\n[1/3] val_hard 추론 중 (threshold 탐색용)…")
    val_loader = make_loader(val_hard_ids, val_meta, args.batch_size)
    val_probs, val_labels = run_inference(model, val_loader)
    print(f"  val_hard shape: probs={val_probs.shape}, labels={val_labels.shape}")

    # 고정 threshold=0.5 기준선
    fixed_preds = (val_probs >= 0.5).astype(int)
    fixed_f1 = f1_score(val_labels, fixed_preds, average="micro", zero_division=0)
    print(f"  [기준] val_hard micro F1 @0.5 = {fixed_f1:.4f}")

    print("\n[2/3] 클래스별 threshold 탐색 중…")
    thresholds = tune_thresholds(val_probs, val_labels, n_grid=args.n_grid)

    tuned_preds = (val_probs >= thresholds[np.newaxis, :]).astype(int)
    tuned_f1 = f1_score(val_labels, tuned_preds, average="micro", zero_division=0)
    print(f"  [결과] val_hard micro F1 @per-class = {tuned_f1:.4f}  "
          f"(vs. 0.5 기준 +{tuned_f1-fixed_f1:+.4f})")

    # threshold 분포 통계
    print(f"  threshold 통계: "
          f"min={thresholds.min():.3f}  "
          f"mean={thresholds.mean():.3f}  "
          f"median={np.median(thresholds):.3f}  "
          f"max={thresholds.max():.3f}")

    # threshold 저장
    ckpt_stem = Path(args.checkpoint).stem
    thr_path  = ROOT / "outputs" / "results" / f"thresholds_{ckpt_stem}.npy"
    np.save(thr_path, thresholds)
    print(f"  threshold 저장: {thr_path}")

    # ── Step 2: 공식 테스트셋 평가 ────────────────────────────────────────────
    results = {}
    print("\n[3/3] 테스트셋 평가 중…")

    test_ids = splits_dir / "test_ids.txt"
    test_meta = proc_dir / "test_meta.csv"
    if test_ids.exists() and test_meta.exists():
        print("  Swiss-Prot 2023-01 추론 중…")
        test_loader = make_loader(test_ids, test_meta, args.batch_size)
        test_probs, test_labels = run_inference(model, test_loader)

        # 고정 threshold 기준 (비교용)
        fp = (test_probs >= 0.5).astype(int)
        ff1 = f1_score(test_labels, fp, average="micro", zero_division=0)
        fwf1 = f1_score(test_labels, fp, average="weighted", zero_division=0)
        frec = recall_score(test_labels, fp, average="micro", zero_division=0)
        print(f"\n  [고정 @0.5] micro={ff1:.4f}  weighted={fwf1:.4f}  recall={frec:.4f}")

        results["swissprot_2023"] = evaluate(
            test_probs, test_labels, thresholds, "Swiss-Prot 2023-01",
            train_rare_mask=train_rare_mask)

    price_ids = splits_dir / "price149_ids.txt"
    price_meta = proc_dir / "price149_meta.csv"
    if price_ids.exists() and price_meta.exists():
        print("\n  Price-149 추론 중…")
        p149_loader = make_loader(price_ids, price_meta, args.batch_size)
        p149_probs, p149_labels = run_inference(model, p149_loader)

        fp2 = (p149_probs >= 0.5).astype(int)
        ff2 = f1_score(p149_labels, fp2, average="micro", zero_division=0)
        print(f"  [고정 @0.5] micro={ff2:.4f}")

        results["price149"] = evaluate(
            p149_probs, p149_labels, thresholds, "Price-149 (OOD)",
            train_rare_mask=train_rare_mask)

    # 결과 저장
    out_path = ROOT / "outputs" / "results" / f"tuned_eval_{ckpt_stem}.json"
    with open(out_path, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "model": args.model,
                   "n_grid": args.n_grid, "threshold_file": str(thr_path),
                   "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
