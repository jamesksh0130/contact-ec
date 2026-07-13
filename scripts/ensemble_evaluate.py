"""
앙상블 평가 스크립트 — FusionV2 + FusionV3 로짓 평균
사용법:
  python scripts/ensemble_evaluate.py \
      --v2_ckpt outputs/checkpoints/fusion_v2_best.pt \
      --v3_ckpt outputs/checkpoints/fusion_v3_best.pt \
      --split test --gpu 0

  # 가중 앙상블 (V3 가중치 0.6)
  python scripts/ensemble_evaluate.py ... --w2 0.4 --w3 0.6

  # Hierarchical constraint 추론
  python scripts/ensemble_evaluate.py ... --hierarchical_inference
"""
import argparse, pickle, yaml, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import ContactPairDataset, collate_fn_v3
from models.fusion_v2 import FusionModelV2
from models.fusion_v3 import FusionModelV3

DEVICE = "cuda:0"


def build_parent_child_matrix(encoders, parent_level: int, child_level: int) -> torch.Tensor:
    parent_cls = encoders[f"level{parent_level}"].classes_
    child_cls  = encoders[f"level{child_level}"].classes_
    p_dict = {c: i for i, c in enumerate(parent_cls)}
    M = torch.zeros(len(parent_cls), len(child_cls), dtype=torch.bool)
    for j, cc in enumerate(child_cls):
        parts  = cc.split(".")
        prefix = ".".join(parts[:parent_level])
        if prefix in p_dict:
            M[p_dict[prefix], j] = True
    return M


@torch.no_grad()
def run_inference_both(v2_model, v3_model, loader, w2: float, w3: float,
                       hier_maps=None):
    """두 모델의 로짓을 가중 평균해 예측 생성."""
    v2_model.eval()
    v3_model.eval()
    all_preds, all_labels, all_masks = [], [], []
    NEG_INF = -1e9

    for batch in loader:
        esm_emb, cmap, sequences, labels, masks, l4_mh, accs, pair_emb = batch
        esm_emb  = esm_emb.to(DEVICE)
        cmap     = cmap.to(DEVICE)
        pair_emb = pair_emb.to(DEVICE)

        # V2 추론 (esm_emb + cmap)
        lg2 = v2_model(esm_emb, cmap)
        # V3 추론 (esm_emb + pair_emb)
        lg3 = v3_model(esm_emb, pair_emb)

        # 가중 평균 (레벨별)
        lg = []
        for i in range(4):
            avg = w2 * lg2[i].cpu().float() + w3 * lg3[i].cpu().float()
            lg.append(avg)

        if hier_maps is not None:
            M12, M23, M34 = hier_maps
            B = lg[0].shape[0]
            l1_pred = lg[0].argmax(dim=1)
            for b in range(B):
                lg[1][b][~M12[l1_pred[b]]] = NEG_INF
            l2_pred = lg[1].argmax(dim=1)
            for b in range(B):
                lg[2][b][~M23[l2_pred[b]]] = NEG_INF
            l3_pred = lg[2].argmax(dim=1)
            for b in range(B):
                lg[3][b][~M34[l3_pred[b]]] = NEG_INF
            l4_pred = torch.sigmoid(lg[3]).argmax(dim=1).numpy()
            batch_preds = np.stack(
                [l1_pred.numpy(), l2_pred.numpy(), l3_pred.numpy(), l4_pred], axis=1)
        else:
            l4_pred = torch.sigmoid(lg[3]).argmax(dim=1).numpy()
            batch_preds = np.stack(
                [lg[i].argmax(dim=1).numpy() for i in range(3)] + [l4_pred], axis=1)

        all_preds.extend(batch_preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_masks.extend(masks.numpy().tolist())

    return all_preds, all_labels, all_masks


def compute_metrics(preds, labels, masks):
    results = {}
    for lvl_idx, lvl_name in enumerate(["level1", "level2", "level3", "level4"]):
        valid_p, valid_l = [], []
        for p, l, m in zip(preds, labels, masks):
            if m[lvl_idx] == 1 and l[lvl_idx] >= 0:
                valid_p.append(p[lvl_idx])
                valid_l.append(l[lvl_idx])
        if not valid_p:
            continue
        results[lvl_name] = {
            "n_samples":          len(valid_p),
            "accuracy":           round(accuracy_score(valid_l, valid_p), 4),
            "micro_f1":           round(f1_score(valid_l, valid_p, average="micro",    zero_division=0), 4),
            "macro_f1":           round(f1_score(valid_l, valid_p, average="macro",    zero_division=0), 4),
            "weighted_f1":        round(f1_score(valid_l, valid_p, average="weighted", zero_division=0), 4),
            "weighted_precision": round(precision_score(valid_l, valid_p, average="weighted", zero_division=0), 4),
            "weighted_recall":    round(recall_score(valid_l, valid_p, average="weighted",    zero_division=0), 4),
        }
    return results


def get_rare_classes(meta_csv, splits_dir, threshold=25):
    import pandas as pd
    meta = pd.read_csv(meta_csv)
    train_ids = set(Path(splits_dir).joinpath("train_ids.txt").read_text().strip().split("\n"))
    train_meta = meta[meta["accession"].isin(train_ids)]
    counts = train_meta["l4_idx"].value_counts()
    return set(counts[counts <= threshold].index.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_ckpt", default="outputs/checkpoints/fusion_v2_best.pt")
    parser.add_argument("--v3_ckpt", default="outputs/checkpoints/fusion_v3_best.pt")
    parser.add_argument("--split",   default="test", choices=["val", "test"])
    parser.add_argument("--gpu",     type=int, default=0)
    parser.add_argument("--w2",      type=float, default=0.5, help="V2 가중치")
    parser.add_argument("--w3",      type=float, default=0.5, help="V3 가중치")
    parser.add_argument("--hierarchical_inference", action="store_true")
    args = parser.parse_args()

    global DEVICE
    DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # 가중치 정규화
    total = args.w2 + args.w3
    w2, w3 = args.w2 / total, args.w3 / total
    print(f"앙상블 가중치: V2={w2:.2f}, V3={w3:.2f}")

    # 라벨 인코더
    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]

    # 모델 로드
    v2_ckpt = torch.load(ROOT / args.v2_ckpt, map_location=DEVICE)
    v2_model = FusionModelV2(n_classes,
                              esm_dim=CFG["model"]["esm2_dim"],
                              contact_dim=CFG["model"]["resnet_out_dim"],
                              fusion_dim=CFG["model"]["fusion_dim"],
                              dropout=0.0).to(DEVICE)
    v2_model.load_state_dict(v2_ckpt["model"])
    print(f"V2 로드: val_micro_f1={v2_ckpt.get('micro_f1', 'N/A')}")

    v3_ckpt = torch.load(ROOT / args.v3_ckpt, map_location=DEVICE)
    v3_model = FusionModelV3(n_classes,
                              esm_dim=CFG["model"]["esm2_dim"],
                              contact_dim=CFG["model"]["resnet_out_dim"],
                              fusion_dim=CFG["model"]["fusion_dim"],
                              dropout=0.0).to(DEVICE)
    v3_model.load_state_dict(v3_ckpt["model"])
    print(f"V3 로드: val_micro_f1={v3_ckpt.get('micro_f1', 'N/A')}")

    # 데이터 — ContactPairDataset (V2+V3 공통 사용)
    ds = ContactPairDataset(
        ids_file      = ROOT / CFG["paths"]["splits_dir"] / f"{args.split}_ids.txt",
        meta_csv      = ROOT / CFG["paths"]["meta_csv"],
        embed_dir     = ROOT / CFG["paths"]["embed_dir"],
        cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
        label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
        pair_emb_dir  = ROOT / "data" / "processed" / "contact_pair_embs",
        k_pairs       = 32,
    )
    loader = DataLoader(ds, batch_size=CFG["train"]["batch_size"],
                        shuffle=False, num_workers=4,
                        collate_fn=collate_fn_v3, pin_memory=True)
    print(f"평가 셋({args.split}): {len(ds):,}개")

    # Hierarchical constraint 매핑
    hier_maps = None
    if args.hierarchical_inference:
        M12 = build_parent_child_matrix(encoders, 1, 2)
        M23 = build_parent_child_matrix(encoders, 2, 3)
        M34 = build_parent_child_matrix(encoders, 3, 4)
        hier_maps = (M12, M23, M34)
        print("추론 방식: Hierarchical Constraint")
    else:
        print("추론 방식: 독립 argmax")

    # 추론
    print("\n추론 중...")
    preds, labels, masks = run_inference_both(v2_model, v3_model, loader,
                                               w2=w2, w3=w3, hier_maps=hier_maps)

    # 메트릭
    results = compute_metrics(preds, labels, masks)

    infer_mode = "hier" if args.hierarchical_inference else "indep"
    print(f"\n=== 앙상블(V2×{w2:.2f}+V3×{w3:.2f}) 평가 결과 [{infer_mode}] ===")
    print(f"  {'Level':<10} {'Acc':>7} {'Micro F1':>9} {'Macro F1':>9} "
          f"{'Weighted F1':>12} {'W-Prec':>8} {'W-Rec':>7}  n")
    print("  " + "-" * 72)
    for lvl, m in results.items():
        print(f"  {lvl:<10} {m['accuracy']:>7.4f} {m['micro_f1']:>9.4f} "
              f"{m['macro_f1']:>9.4f} {m.get('weighted_f1',0):>12.4f} "
              f"{m.get('weighted_precision',0):>8.4f} "
              f"{m.get('weighted_recall',0):>7.4f}  {m['n_samples']:,}")

    # HIT-EC 비교
    if "level4" in results:
        r4 = results["level4"]
        print(f"\n[비교 — Level 4 Micro F1]")
        print(f"  HIT-EC     : 0.9300")
        print(f"  앙상블(ours): {r4['micro_f1']:.4f}  ({'↑' if r4['micro_f1'] >= 0.9300 else '↓'} HIT-EC {'초과' if r4['micro_f1'] >= 0.9300 else '미달'})")
        print(f"  FusionV3   : 0.9226")
        print(f"  FusionV2   : 0.9209")

    # Underrepresented 평가
    rare_classes = get_rare_classes(
        str(ROOT / CFG["paths"]["meta_csv"]),
        str(ROOT / CFG["paths"]["splits_dir"]),
        threshold=25
    )
    if rare_classes:
        rare_p, rare_l = [], []
        for p, l, m in zip(preds, labels, masks):
            if m[3] == 1 and l[3] >= 0 and l[3] in rare_classes:
                rare_p.append(p[3])
                rare_l.append(l[3])
        if rare_p:
            rare_micro = f1_score(rare_l, rare_p, average="micro", zero_division=0)
            print(f"\n[Underrepresented EC (N≤25, Level4)]")
            print(f"  앙상블   : {rare_micro:.4f}")
            print(f"  FusionV2 : 0.6422  (V2 단독 최고)")
            print(f"  FusionV3 : 0.6118")
            print(f"  HIT-EC   : 0.77")
            print(f"  CLEAN    : 0.73")
            results["level4_underrepresented"] = {
                "n_samples": len(rare_p),
                "micro_f1":  round(rare_micro, 4),
            }

    # 저장
    out_dir = ROOT / CFG["paths"]["result_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    w2_str = str(int(w2 * 10))
    w3_str = str(int(w3 * 10))
    suffix = f"_w{w2_str}{w3_str}" + ("_hier" if args.hierarchical_inference else "")
    out_file = out_dir / f"ensemble_v2v3_{args.split}{suffix}_results.json"
    with open(out_file, "w") as f:
        json.dump({"weights": {"v2": w2, "v3": w3}, "metrics": results}, f, indent=2)
    print(f"\n결과 저장: {out_file}")


if __name__ == "__main__":
    main()
