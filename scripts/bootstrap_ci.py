"""
Bootstrap 95% Confidence Interval for Multi-label L4 Micro F1

Usage:
  # V2 random test
  python scripts/bootstrap_ci.py --model fusion_v2 \
      --checkpoint outputs/checkpoints/fusion_v2_ml_best.pt \
      --split test --gpu 0

  # V2 cluster test
  python scripts/bootstrap_ci.py --model fusion_v2 \
      --checkpoint outputs/checkpoints/fusion_v2_ml_best.pt \
      --split cluster_test --gpu 0

  # B2 random test
  python scripts/bootstrap_ci.py --model b2_esm2_hier \
      --checkpoint outputs/checkpoints/b2_ml_best.pt \
      --split test --gpu 0

  # 캐시된 probs/labels 재사용 (빠른 재계산)
  python scripts/bootstrap_ci.py --load_cache outputs/results/bootstrap_cache_v2_test.npz
"""
import argparse, pickle, yaml, json, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import ProteinDataset, collate_fn


# ── 모델 빌드 ─────────────────────────────────────────────────────────────────
def build_model(model_name, n_classes):
    if model_name == "b0_cnn":
        from models.baseline_cnn import BaselineCNN
        return BaselineCNN(n_classes, dropout=0.0)
    elif model_name == "b1_esm2_fc":
        from models.esm2_fc import ESM2FC
        return ESM2FC(n_classes, esm_dim=CFG["model"]["esm2_dim"], dropout=0.0)
    elif model_name == "b2_esm2_hier":
        from models.esm2_hierarchical import ESM2Hierarchical
        return ESM2Hierarchical(n_classes, esm_dim=CFG["model"]["esm2_dim"], dropout=0.0)
    elif model_name == "b3_contact":
        from models.contact_resnet import ContactResNet
        return ContactResNet(n_classes, dropout=0.0)
    elif model_name == "fusion_v2":
        from models.fusion_v2 import FusionModelV2
        return FusionModelV2(n_classes, esm_dim=CFG["model"]["esm2_dim"],
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"], dropout=0.0)
    elif model_name == "fusion_v3":
        from models.fusion_v3 import FusionModelV3
        return FusionModelV3(n_classes, esm_dim=CFG["model"]["esm2_dim"],
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"], dropout=0.0)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── 추론: sigmoid 확률 + multi-hot 라벨 수집 ──────────────────────────────────
@torch.no_grad()
def collect_probs_labels(model, loader, model_name, device):
    model.eval()
    all_probs, all_labels, all_masks = [], [], []

    for batch in loader:
        esm_emb, cmap, sequences, labels, masks, l4_mh, _ = batch
        esm_emb = esm_emb.to(device)
        cmap    = cmap.to(device)

        if model_name == "b0_cnn":
            logits = model(sequences, device=device)
        else:
            logits = model(esm_emb, cmap)

        l4_prob = torch.sigmoid(logits[3]).cpu().numpy()
        all_probs.append(l4_prob)
        all_labels.append(l4_mh.numpy())
        all_masks.append(masks[:, 3].numpy())   # m4 마스크

    probs  = np.concatenate(all_probs,  axis=0)   # (N, n_l4)
    labels = np.concatenate(all_labels, axis=0)   # (N, n_l4)
    masks  = np.concatenate(all_masks,  axis=0)   # (N,)

    # 유효 샘플만 필터 (m4==1)
    valid = masks == 1
    return probs[valid], labels[valid]


# ── Bootstrap CI 계산 ─────────────────────────────────────────────────────────
def bootstrap_ci(probs, labels, threshold=0.5, n_iter=1000, alpha=0.05, seed=42):
    """
    Returns:
        point_f1  : float  — 전체 샘플 F1 (bootstrap 없이)
        mean_f1   : float  — bootstrap 평균
        ci_low    : float  — (alpha/2) percentile
        ci_high   : float  — (1 - alpha/2) percentile
        std_f1    : float  — bootstrap 표준편차
    """
    rng = np.random.default_rng(seed)
    N   = len(probs)

    # Point estimate (전체 데이터)
    pred_bin = (probs >= threshold).astype(np.int32)
    zero_rows = pred_bin.sum(axis=1) == 0
    if zero_rows.any():
        pred_bin[zero_rows] = 0
        pred_bin[zero_rows, probs[zero_rows].argmax(axis=1)] = 1
    point_f1 = f1_score(labels.astype(np.int32), pred_bin, average="micro", zero_division=0)
    point_pr = precision_score(labels.astype(np.int32), pred_bin, average="micro", zero_division=0)
    point_rc = recall_score(labels.astype(np.int32), pred_bin,    average="micro", zero_division=0)

    # Bootstrap
    boot_f1s = np.zeros(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, N, size=N)          # 복원 추출
        bp  = probs[idx]
        bl  = labels[idx]
        bp_bin = (bp >= threshold).astype(np.int32)
        zero_rows = bp_bin.sum(axis=1) == 0
        if zero_rows.any():
            bp_bin[zero_rows] = 0
            bp_bin[zero_rows, bp[zero_rows].argmax(axis=1)] = 1
        boot_f1s[i] = f1_score(bl.astype(np.int32), bp_bin, average="micro", zero_division=0)

    ci_low  = float(np.percentile(boot_f1s, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_f1s, 100 * (1 - alpha / 2)))

    return {
        "point_f1":  round(float(point_f1), 4),
        "point_prec": round(float(point_pr), 4),
        "point_rec":  round(float(point_rc), 4),
        "mean_f1":   round(float(boot_f1s.mean()), 4),
        "std_f1":    round(float(boot_f1s.std()), 4),
        "ci_low":    round(ci_low,  4),
        "ci_high":   round(ci_high, 4),
        "n_samples": int(N),
        "n_iter":    n_iter,
        "threshold": threshold,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=None)
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--split",       default="test",
                        choices=["val", "test", "cluster_test", "cluster_val", "price149"])
    parser.add_argument("--gpu",         type=int, default=0)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--n_iter",      type=int, default=1000,
                        help="Bootstrap 반복 횟수 (기본 1000)")
    parser.add_argument("--load_cache",  default=None,
                        help="캐시된 probs/labels .npz 파일 경로 (추론 생략)")
    parser.add_argument("--save_cache",  default=None,
                        help="probs/labels를 .npz로 저장할 경로")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # ── probs/labels 로드 또는 추론 ─────────────────────────────────────────
    if args.load_cache:
        print(f"캐시 로드: {args.load_cache}")
        cache = np.load(args.load_cache)
        probs, labels = cache["probs"], cache["labels"]
        split_tag = Path(args.load_cache).stem
    else:
        assert args.model and args.checkpoint, \
            "--model과 --checkpoint가 필요합니다 (또는 --load_cache 사용)"

        # 라벨 인코더
        with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
            encoders = pickle.load(f)
        n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]

        # 모델 로드
        ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = build_model(args.model, n_classes).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"모델 로드: {args.checkpoint}")
        print(f"Val best F1: {ckpt.get('micro_f1', 'N/A')}")

        # 데이터셋
        if args.split == "price149":
            ids_file = str(ROOT / "data" / "splits" / "price149_ids.txt")
            meta_csv = str(ROOT / "data" / "processed" / "price149_meta.csv")
        else:
            ids_file = str(ROOT / CFG["paths"]["splits_dir"] / f"{args.split}_ids.txt")
            meta_csv = str(ROOT / CFG["paths"]["meta_csv"])

        ds = ProteinDataset(
            ids_file      = ids_file,
            meta_csv      = meta_csv,
            embed_dir     = ROOT / CFG["paths"]["embed_dir"],
            cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
            label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)

        print(f"데이터셋({args.split}): {len(ds):,}개 → 추론 시작...")
        t0 = time.time()
        probs, labels = collect_probs_labels(model, loader, args.model, device)
        print(f"추론 완료: {time.time()-t0:.1f}s  유효 샘플: {len(probs):,}")

        split_tag = f"{args.model}_{args.split}"

        # 캐시 저장
        cache_path = args.save_cache or str(
            ROOT / "outputs" / "results" / f"bootstrap_cache_{split_tag}.npz"
        )
        np.savez_compressed(cache_path, probs=probs, labels=labels)
        print(f"캐시 저장: {cache_path}")

    # ── Bootstrap ───────────────────────────────────────────────────────────
    print(f"\nBootstrap CI 계산 중 (n_iter={args.n_iter}, threshold={args.threshold})...")
    t0 = time.time()
    result = bootstrap_ci(probs, labels,
                          threshold=args.threshold,
                          n_iter=args.n_iter)
    print(f"완료: {time.time()-t0:.1f}s")

    # ── 결과 출력 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Bootstrap 95% CI — L4 Multi-label Micro F1")
    print(f"{'='*60}")
    print(f"  샘플 수       : {result['n_samples']:,}")
    print(f"  Threshold     : {result['threshold']}")
    print(f"  Bootstrap iter: {result['n_iter']}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Point F1      : {result['point_f1']:.4f}")
    print(f"  Point Prec    : {result['point_prec']:.4f}")
    print(f"  Point Recall  : {result['point_rec']:.4f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Bootstrap mean: {result['mean_f1']:.4f}")
    print(f"  Bootstrap std : {result['std_f1']:.4f}")
    print(f"  95% CI        : [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
    print(f"  ─────────────────────────────────────────")
    print(f"  논문 표기 형식:")
    print(f"  {result['point_f1']:.4f} (95% CI: {result['ci_low']:.4f}–{result['ci_high']:.4f})")
    print(f"{'='*60}")

    # HIT-EC 비교
    hit_ec = 0.9300
    if result['ci_low'] > hit_ec:
        print(f"\n  ✅ 95% CI 하한 ({result['ci_low']:.4f}) > HIT-EC ({hit_ec})")
        print(f"     → 통계적으로 유의미하게 HIT-EC 초과")
    elif result['point_f1'] > hit_ec:
        print(f"\n  ⚠️  Point F1 > HIT-EC지만 95% CI 하한 ({result['ci_low']:.4f}) ≤ HIT-EC")
        print(f"     → 통계적 유의성 불충분 — threshold 조정 또는 fine-tuning 권장")
    else:
        print(f"\n  ❌ Point F1 ≤ HIT-EC")

    # 저장
    if not args.load_cache:
        out_path = ROOT / "outputs" / "results" / f"bootstrap_ci_{split_tag}.json"
    else:
        out_path = ROOT / "outputs" / "results" / f"bootstrap_ci_{split_tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
