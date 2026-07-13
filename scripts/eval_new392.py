"""
New-392 / Price-149 평가 스크립트 (CLEAN 논문 비교용)

검증된 CLEAN/CLEAN-Contact F1 수치 (논문 본문에서 직접 확인):
  New-392  (CLEAN-Contact paper, Commun.Bio 2024, p.2 text, P-value selection):
    CLEAN-Contact F1=0.566, CLEAN F1=0.504
    ProteInfer F1=0.309 (CLEAN Science2023 p.2 text, max-sep selection)
    DeepEC F1=0.230 (CLEAN Science2023 p.2 text, max-sep)

  Price-149 (CLEAN-Contact paper, Commun.Bio 2024, p.2 text, P-value selection):
    CLEAN-Contact F1=0.525, CLEAN F1=0.452

  ※ CLEAN Science2023 F1=0.865: 자체 Swiss-Prot split (<50% id), max-sep 알고리즘
     → New-392와 다른 테스트셋/알고리즘이므로 직접 비교 불가

사용법:
  python scripts/eval_new392.py \
    --checkpoint outputs/checkpoints/ecbench_fv2_phase1_best.pt \
    --model fusion_v2 \
    [--dataset new392|price149|both]
    [--threshold 0.5]
"""
import sys, argparse, pickle, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config_ecbench.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import _make_3ch_cmap

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DATA_DIR = ROOT / "data/new392"
EMBED_DIR = ROOT / "data/processed/embeddings"
CMAP_DIR  = ROOT / "data/processed/contact_maps"
NEW392_CSV  = DATA_DIR / "new392_ec_labels.csv"
PRICE149_CSV = DATA_DIR / "price149_labels.csv"

# CLEAN-Contact 논문 수치 (Yang et al., 2024, Communications Biology, Fig.2a/2b)
# P-value EC number selection algorithm 기준
# 논문 본문에서 직접 검증된 수치만 포함
CLEAN_NUMBERS = {
    "new392": {
        # CLEAN-Contact paper 본문 p.2 (P-value selection, 392 proteins)
        "CLEAN-Contact":                    0.566,
        "CLEAN (P-value)":                  0.504,
        # CLEAN Science2023 본문 p.2 (max-sep selection)
        "ProteInfer":                       0.309,
        "DeepEC (max-sep)":                 0.230,
        # ※ 0.865는 New-392가 아니라 자체 Swiss-Prot split (<50% id)
        "CLEAN (own split, max-sep)":       0.865,
    },
    "price149": {
        # CLEAN-Contact paper 본문 p.2 (P-value selection, 149 proteins)
        "CLEAN-Contact":                    0.525,
        "CLEAN (P-value)":                  0.452,
    },
}


# ── 데이터셋 ─────────────────────────────────────────────────
class New392Dataset(Dataset):
    def __init__(self, csv_path, encoders, embed_dir=EMBED_DIR, cmap_dir=CMAP_DIR):
        import pandas as pd
        self.df = pd.read_csv(csv_path, sep="\t")
        self.enc_l4 = encoders["level4"]
        self.embed_dir = Path(embed_dir)
        self.cmap_dir  = Path(cmap_dir)

        # EC 라벨 → multi-hot 변환
        n_l4 = len(self.enc_l4.classes_)
        class_set = set(self.enc_l4.classes_)

        self.valid_rows = []
        self.labels_mh  = []
        self.missing_emb = []

        for _, row in self.df.iterrows():
            uid  = row["Entry"]
            ecs  = [e.strip() for e in str(row["EC number"]).split(";")]
            emb_path = self.embed_dir / f"{uid}.npy"

            if not emb_path.exists():
                self.missing_emb.append(uid)
                continue

            # 라벨 인코더에 있는 EC만 사용
            valid_ecs = [ec for ec in ecs if ec in class_set]
            if not valid_ecs:
                continue

            mh = np.zeros(n_l4, dtype=np.float32)
            for ec in valid_ecs:
                idx = np.where(self.enc_l4.classes_ == ec)[0]
                if len(idx) > 0:
                    mh[idx[0]] = 1.0

            self.valid_rows.append(uid)
            self.labels_mh.append(mh)

        print(f"  데이터셋: {len(self.valid_rows)}개 유효 / {len(self.df)}개 전체")
        if self.missing_emb:
            print(f"  임베딩 없음: {len(self.missing_emb)}개 → 제외")
        n_covered = sum(1 for l in self.labels_mh if l.sum() > 0)
        print(f"  라벨 인코더 커버: {n_covered}개 ({n_covered/len(self.valid_rows)*100:.1f}%)")

    def __len__(self):
        return len(self.valid_rows)

    def __getitem__(self, idx):
        uid  = self.valid_rows[idx]
        emb  = np.load(self.embed_dir / f"{uid}.npy").astype(np.float32)
        cmap_path = self.cmap_dir / f"{uid}.npy"
        if cmap_path.exists():
            raw  = np.load(cmap_path).astype(np.float32)
            cmap = _make_3ch_cmap(raw)          # (3,256,256)
        else:
            cmap = np.zeros((3, 256, 256), dtype=np.float32)
        return (
            torch.tensor(emb),
            torch.tensor(cmap),
            torch.tensor(self.labels_mh[idx]),
            uid,
        )


def collate_fn(batch):
    embs, cmaps, labels, uids = zip(*batch)
    return (
        torch.stack(embs),
        torch.stack(cmaps),
        torch.stack(labels),
        list(uids),
    )


# ── 모델 빌더 ─────────────────────────────────────────────────
def build_model(model_name, n_classes):
    if model_name == "b1_esm2_fc":
        from models.esm2_fc import ESM2FC
        return ESM2FC(n_classes, esm_dim=CFG["model"]["esm2_dim"], dropout=0.0)
    elif model_name == "b2_esm2_hier":
        from models.esm2_hierarchical import ESM2Hierarchical
        return ESM2Hierarchical(n_classes, esm_dim=CFG["model"]["esm2_dim"], dropout=0.0)
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
    else:
        raise ValueError(f"미지원 모델: {model_name}")


# ── 추론 ─────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, threshold):
    model.eval()
    all_probs, all_labels = [], []
    for emb, cmap, labels, _ in loader:
        emb   = emb.to(DEVICE)
        cmap  = cmap.to(DEVICE)
        logits = model(emb, cmap)
        probs  = torch.sigmoid(logits[3]).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def per_protein_f1(probs, labels, threshold):
    """CLEAN 방식 per-protein F1: 각 단백질별 F1의 평균.
    - 예측 없는 단백질: F1=0 (threshold 초과 없음)
    - Top-1 F1도 같이 계산 (항상 예측)
    """
    n = len(probs)
    pp_f1_scores = []
    top1_correct = 0

    for i in range(n):
        p = probs[i]
        gt = labels[i]   # multi-hot ground truth
        n_pos = int(gt.sum())

        # top-1 accuracy
        top1_idx = int(np.argmax(p))
        if gt[top1_idx] == 1.0:
            top1_correct += 1

        # threshold-based per-protein F1
        pred = (p >= threshold).astype(np.float32)
        tp = float((pred * gt).sum())
        n_pred = float(pred.sum())

        if n_pred == 0 and n_pos == 0:
            pp_f1_scores.append(1.0)
        elif n_pred == 0 or n_pos == 0:
            pp_f1_scores.append(0.0)
        else:
            prec = tp / n_pred
            rec  = tp / n_pos
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            pp_f1_scores.append(f1)

    top1_acc = top1_correct / n
    avg_pp_f1 = float(np.mean(pp_f1_scores))
    return avg_pp_f1, top1_acc


def compute_metrics(probs, labels, threshold, split_name, clean_ref=None):
    preds = (probs >= threshold).astype(np.int32)
    tgt   = labels.astype(np.int32)
    n     = len(probs)

    micro    = f1_score(tgt, preds, average="micro",    zero_division=0)
    macro    = f1_score(tgt, preds, average="macro",    zero_division=0)
    weighted = f1_score(tgt, preds, average="weighted", zero_division=0)
    prec     = precision_score(tgt, preds, average="micro", zero_division=0)
    rec      = recall_score(tgt, preds, average="micro",    zero_division=0)

    # CLEAN-방식 per-protein F1 (여러 threshold)
    pp_f1_05, top1_acc = per_protein_f1(probs, labels, threshold=0.5)
    pp_f1_01, _        = per_protein_f1(probs, labels, threshold=0.1)

    n_with_pred = int((preds.sum(axis=1) > 0).sum())

    print(f"\n{'='*60}")
    print(f"  [{split_name}]  N={n}")
    print(f"  ── Multi-label 메트릭 (threshold={threshold}) ──────────────")
    print(f"  Micro F1    : {micro:.4f}   (일반 multi-label 평가)")
    print(f"  Weighted F1 : {weighted:.4f}")
    print(f"  Macro F1    : {macro:.4f}")
    print(f"  Precision   : {prec:.4f}  /  Recall: {rec:.4f}")
    print(f"  예측 있는 샘플: {n_with_pred}/{n} ({n_with_pred/n*100:.1f}%)")
    print(f"  ── CLEAN 방식 per-protein F1 ───────────────────────────────")
    print(f"  Per-protein F1 (thr=0.5): {pp_f1_05:.4f}")
    print(f"  Per-protein F1 (thr=0.1): {pp_f1_01:.4f}")
    print(f"  Top-1 Accuracy: {top1_acc:.4f}  (가장 높은 확률 1개 선택)")
    if clean_ref:
        print(f"\n  ── CLEAN 논문 F1과 비교 (per-protein F1 기준) ──────────")
        for name, val in clean_ref.items():
            diff = pp_f1_05 - val
            sign = "+" if diff >= 0 else ""
            print(f"    {name:<20}: {val:.3f}  →  우리 (thr=0.5): {pp_f1_05:.4f}  ({sign}{diff:.4f})")
    print(f"{'='*60}")

    return {
        "split":           split_name,
        "n_samples":       n,
        "micro_f1":        round(float(micro),    4),
        "weighted_f1":     round(float(weighted), 4),
        "macro_f1":        round(float(macro),    4),
        "precision":       round(float(prec),     4),
        "recall":          round(float(rec),      4),
        "per_protein_f1_05": round(float(pp_f1_05), 4),
        "per_protein_f1_01": round(float(pp_f1_01), 4),
        "top1_accuracy":   round(float(top1_acc), 4),
        "n_with_pred":     n_with_pred,
    }


# ── 메인 ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model",      default="fusion_v2",
                        choices=["b1_esm2_fc", "b2_esm2_hier", "fusion_v2",
                                 "b3_contact", "fusion_v2_flatfc"])
    parser.add_argument("--dataset",    default="both",
                        choices=["new392", "price149", "both"])
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--batch_size", type=int,   default=128)
    args = parser.parse_args()

    # 라벨 인코더
    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    print(f"클래스 수 (L4): {n_classes[3]}")

    # 모델 로드
    ckpt  = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model = build_model(args.model, n_classes).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"체크포인트 로드: {args.checkpoint}")
    if "micro_f1" in ckpt:
        print(f"  (val micro_f1={ckpt['micro_f1']:.4f}, epoch={ckpt.get('epoch','?')})")

    all_results = {}

    # ── New-392 ──────────────────────────────────────────────
    if args.dataset in ("new392", "both"):
        print(f"\n[New-392 로드]")
        ds = New392Dataset(NEW392_CSV, encoders)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)
        probs, labels = run_inference(model, loader, args.threshold)
        all_results["new392"] = compute_metrics(
            probs, labels, args.threshold,
            "New-392", CLEAN_NUMBERS["new392"]
        )

    # ── Price-149 ────────────────────────────────────────────
    if args.dataset in ("price149", "both"):
        print(f"\n[Price-149 로드]")
        ds = New392Dataset(PRICE149_CSV, encoders)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)
        probs, labels = run_inference(model, loader, args.threshold)
        all_results["price149"] = compute_metrics(
            probs, labels, args.threshold,
            "Price-149", CLEAN_NUMBERS["price149"]
        )

    # ── 결과 저장 ─────────────────────────────────────────────
    ckpt_stem = Path(args.checkpoint).stem
    out_path  = ROOT / "outputs/results" / f"new392_eval_{ckpt_stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"checkpoint": args.checkpoint,
                   "model": args.model,
                   "threshold": args.threshold,
                   "results": all_results,
                   "reference": CLEAN_NUMBERS}, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
