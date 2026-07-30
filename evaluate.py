"""
평가 스크립트 — 체크포인트 로드 후 Test 셋 평가
사용법:
  python evaluate.py --model fusion --checkpoint outputs/checkpoints/fusion_phase2_best.pt
  python evaluate.py --model fusion --checkpoint ... --hierarchical_inference
  python evaluate.py --model b1_esm2_fc --checkpoint outputs/checkpoints/b1_esm2_fc_phase1_best.pt
"""
import argparse, pickle, yaml, json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, accuracy_score,
                              classification_report, precision_score,
                              recall_score)

ROOT = Path(__file__).resolve().parent

# config 파일은 --config 인자로 오버라이드 가능 (argparse 이전에 처리)
import sys as _sys
_cfg_file = "config.yaml"
for _i, _a in enumerate(_sys.argv):
    if _a == "--config" and _i + 1 < len(_sys.argv):
        _cfg_file = _sys.argv[_i + 1]
        break
with open(ROOT / "configs" / _cfg_file) as f:
    CFG = yaml.safe_load(f)

from models.dataset import ProteinDataset, collate_fn, ContactPairDataset, collate_fn_v3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_parent_child_matrix(encoders, parent_level: int, child_level: int) -> torch.Tensor:
    """(n_parent, n_child) binary matrix: M[i,j]=1 이면 child j가 parent i의 자식."""
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
    elif model_name == "fusion":
        from models.fusion_model import FusionModel
        return FusionModel(n_classes, esm_dim=CFG["model"]["esm2_dim"],
                           contact_dim=CFG["model"]["resnet_out_dim"],
                           fusion_dim=CFG["model"]["fusion_dim"], dropout=0.0)
    elif model_name == "fusion_esm_ft":
        from models.fusion_esm_ft import FusionESMFinetune
        unfreeze = CFG["model"].get("esm_unfreeze_layers", 4)
        return FusionESMFinetune(n_classes,
                                 contact_dim=CFG["model"]["resnet_out_dim"],
                                 fusion_dim=CFG["model"]["fusion_dim"],
                                 dropout=0.0,
                                 unfreeze_layers=unfreeze)

    elif model_name == "fusion_v2":
        from models.fusion_v2 import FusionModelV2
        return FusionModelV2(n_classes,
                             esm_dim=CFG["model"]["esm2_dim"],
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=0.0)

    elif model_name == "fusion_v3":
        from models.fusion_v3 import FusionModelV3
        return FusionModelV3(n_classes,
                             esm_dim=CFG["model"]["esm2_dim"],
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=0.0)

    elif model_name == "fusion_v3_esm_ft":
        from models.fusion_v3_esm_ft import FusionV3ESMFt
        unfreeze = CFG["model"].get("esm_unfreeze_layers", 2)
        return FusionV3ESMFt(n_classes,
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=0.0,
                             unfreeze_layers=unfreeze)

    elif model_name in ("fusion_v2_flatfc", "b4_flatfc"):
        from models.fusion_v2_flatfc import FusionV2FlatFC
        return FusionV2FlatFC(n_classes,
                              esm_dim=CFG["model"]["esm2_dim"],
                              contact_dim=CFG["model"].get("resnet_out_dim", 512),
                              fusion_dim=CFG["model"]["fusion_dim"],
                              dropout=0.0)

    elif model_name == "fusion_concat_flatfc":
        from models.fusion_simple_baselines import FusionConcatFlatFC
        return FusionConcatFlatFC(n_classes,
                                  esm_dim=CFG["model"]["esm2_dim"],
                                  contact_dim=CFG["model"].get("resnet_out_dim", 512),
                                  fusion_dim=CFG["model"]["fusion_dim"],
                                  dropout=0.0)

    elif model_name == "fusion_sum_flatfc":
        from models.fusion_simple_baselines import FusionSumFlatFC
        return FusionSumFlatFC(n_classes,
                               esm_dim=CFG["model"]["esm2_dim"],
                               contact_dim=CFG["model"].get("resnet_out_dim", 512),
                               fusion_dim=CFG["model"]["fusion_dim"],
                               dropout=0.0)

    elif model_name == "fusion_gated_mlp_flatfc":
        from models.fusion_simple_baselines import FusionGatedMLPFlatFC
        return FusionGatedMLPFlatFC(n_classes,
                                    esm_dim=CFG["model"]["esm2_dim"],
                                    contact_dim=CFG["model"].get("resnet_out_dim", 512),
                                    fusion_dim=CFG["model"]["fusion_dim"],
                                    dropout=0.0)


@torch.no_grad()
def run_inference(model, loader, model_name, hier_maps=None, threshold: float = 0.5):
    """
    반환:
      preds_l13  : list[(l1,l2,l3)] argmax 예측 (L1~L3)
      labels_all : list[(l1,l2,l3,l4)] 정수 레이블
      masks_all  : list[(m1,m2,m3,m4)]
      l4_probs   : np.ndarray (N, n_l4) L4 sigmoid 확률
      l4_mh_all  : np.ndarray (N, n_l4) L4 멀티핫 정답
    hier_maps: (M12, M23, M34) bool tensors (CPU). 지정 시 L4 가능 자식만 활성화.
    """
    model.eval()
    preds_l13  = []
    labels_all = []
    masks_all  = []
    l4_probs_list = []
    l4_mh_list    = []
    NEG_INF = -1e9

    for batch in loader:
        if model_name == "fusion_v3":
            esm_emb, cmap, sequences, labels, masks, l4_mh, _, pair_emb = batch
            pair_emb = pair_emb.to(DEVICE)
        else:
            esm_emb, cmap, sequences, labels, masks, l4_mh, _ = batch

        esm_emb = esm_emb.to(DEVICE)
        cmap    = cmap.to(DEVICE)

        if model_name == "b0_cnn":
            logits = model(sequences, device=DEVICE)
        elif model_name in ("fusion_esm_ft", "fusion_v3_esm_ft"):
            logits = model(sequences, cmap)
        elif model_name == "fusion_v3":
            logits = model(esm_emb, pair_emb)
        else:
            logits = model(esm_emb, cmap)

        lg = [l.cpu().float() for l in logits]

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

            # L4는 마스킹하지 않음 — multi-label sigmoid는 계층 제약 없이 평가
            # (L3 오류 cascade가 L4 recall을 낮추므로 free sigmoid 사용)

            preds_l13.extend(
                np.stack([l1_pred.numpy(), l2_pred.numpy(), l3_pred.numpy()], axis=1).tolist()
            )
        else:
            preds_l13.extend(
                np.stack([lg[i].argmax(dim=1).numpy() for i in range(3)], axis=1).tolist()
            )

        # L4: sigmoid 확률 저장 (멀티레이블 평가용, 마스킹 없음)
        l4_probs_list.append(torch.sigmoid(lg[3]).numpy())
        l4_mh_list.append(l4_mh.numpy())

        labels_all.extend(labels.numpy().tolist())
        masks_all.extend(masks.numpy().tolist())

    l4_probs  = np.concatenate(l4_probs_list, axis=0)   # (N, n_l4)
    l4_mh_all = np.concatenate(l4_mh_list,    axis=0)   # (N, n_l4)
    return preds_l13, labels_all, masks_all, l4_probs, l4_mh_all


def get_rare_classes(meta_csv: str, splits_dir: str, threshold: int = 25) -> set:
    """학습 셋에서 Level4 샘플 수 <= threshold인 EC 클래스 인덱스 집합."""
    import pandas as pd
    meta = pd.read_csv(meta_csv)
    train_ids = set(Path(splits_dir).joinpath("train_ids.txt").read_text().strip().split("\n"))
    train_meta = meta[meta["accession"].isin(train_ids)]
    counts = train_meta["l4_idx"].value_counts()
    return set(counts[counts <= threshold].index.tolist())


def compute_all_metrics(preds_l13, labels_all, masks_all,
                        l4_probs, l4_mh_all, encoders, threshold: float = 0.5):
    """
    L1~L3: 단일레이블 accuracy / F1
    L4   : 멀티레이블 precision / recall / micro-F1 / macro-F1
    """
    results = {}

    # ── L1~L3 단일레이블 ────────────────────────────────────────
    for lvl_idx, lvl_name in enumerate(["level1", "level2", "level3"]):
        valid_p, valid_l = [], []
        for p, l, m in zip(preds_l13, labels_all, masks_all):
            if m[lvl_idx] == 1 and l[lvl_idx] >= 0:
                valid_p.append(p[lvl_idx])
                valid_l.append(l[lvl_idx])
        if not valid_p:
            continue
        results[lvl_name] = {
            "n_samples":   len(valid_p),
            "accuracy":    round(accuracy_score(valid_l, valid_p), 4),
            "micro_f1":    round(f1_score(valid_l, valid_p, average="micro",     zero_division=0), 4),
            "macro_f1":    round(f1_score(valid_l, valid_p, average="macro",     zero_division=0), 4),
            "weighted_f1": round(f1_score(valid_l, valid_p, average="weighted",  zero_division=0), 4),
        }

    # ── L4 멀티레이블 ───────────────────────────────────────────
    # 유효한 샘플만 사용 (m4==1)
    valid_mask = np.array([m[3] == 1 for m in masks_all])
    lp = l4_probs[valid_mask]     # (N_valid, n_l4)
    lt = l4_mh_all[valid_mask]    # (N_valid, n_l4)

    pred_bin = (lp >= threshold).astype(np.int32)
    target   = lt.astype(np.int32)

    micro    = f1_score(target, pred_bin, average="micro", zero_division=0)
    macro    = f1_score(target, pred_bin, average="macro", zero_division=0)
    weighted = f1_score(target, pred_bin, average="weighted", zero_division=0)
    prec     = precision_score(target, pred_bin, average="micro", zero_division=0)
    rec      = recall_score(target, pred_bin, average="micro", zero_division=0)

    results["level4"] = {
        "n_samples":  int(valid_mask.sum()),
        "micro_f1":   round(micro, 4),
        "macro_f1":   round(macro, 4),
        "weighted_f1": round(weighted, 4),
        "precision":  round(prec,  4),
        "recall":     round(rec,   4),
        "threshold":  threshold,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml", help="configs/ 아래 yaml 파일명")
    parser.add_argument("--model",      required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      default="test",
                        help="Split name without '_ids.txt' suffix, or one of price149/new392.")
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--hierarchical_inference", action="store_true",
                        help="L1→L2→L3→L4 계층 constraint 추론 (독립 argmax 대신)")
    # Price-149 fair evaluation 전용 오버라이드
    parser.add_argument("--ids_file",   default=None, help="커스텀 IDs 파일")
    parser.add_argument("--meta_csv",   default=None, help="커스텀 meta CSV")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="배치 크기 오버라이드 (기본: config.yaml 값)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker 수 (sandbox/CPU 평가에서는 0 권장)")
    parser.add_argument("--cmap_dir",   default=None,
                        help="contact map 디렉토리 오버라이드 (채널 ablation 등)")
    args = parser.parse_args()

    global DEVICE
    DEVICE = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # 라벨 인코더
    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]

    # 체크포인트
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model = build_model(args.model, n_classes).to(DEVICE)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    print(f"체크포인트 로드: {args.checkpoint}")
    print(f"Val micro_f1 (학습 중 최고): {ckpt.get('micro_f1', 'N/A')}")

    # 데이터
    if args.model == "fusion_v3":
        # price149 전용 meta 처리 (V3도 동일하게 적용)
        if args.split == "price149" or args.ids_file:
            v3_ids_file = args.ids_file or str(ROOT / "data" / "splits" / "price149_ids.txt")
            v3_meta_csv = args.meta_csv or str(ROOT / "data" / "processed" / "price149_meta.csv")
        else:
            v3_ids_file = str(ROOT / CFG["paths"]["splits_dir"] / f"{args.split}_ids.txt")
            v3_meta_csv = str(ROOT / CFG["paths"]["meta_csv"])
        ds = ContactPairDataset(
            ids_file      = v3_ids_file,
            meta_csv      = v3_meta_csv,
            embed_dir     = ROOT / CFG["paths"]["embed_dir"],
            cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
            label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
            pair_emb_dir  = ROOT / "data" / "processed" / "contact_pair_embs",
            k_pairs       = 32,
        )
        bs = args.batch_size or CFG["train"]["batch_size"]
        loader = DataLoader(ds, batch_size=bs,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn_v3, pin_memory=True)
    else:
        # price149 / new392 split 또는 커스텀 경로 지원
        if args.split == "price149" or args.ids_file:
            ids_file = args.ids_file or str(ROOT / "data" / "splits" / "price149_ids.txt")
            meta_csv = args.meta_csv or str(ROOT / "data" / "processed" / "price149_meta.csv")
        elif args.split == "new392":
            ids_file = str(ROOT / "data" / "splits" / "new392_ids.txt")
            meta_csv = str(ROOT / "data" / "processed" / "new392_meta.csv")
        else:
            ids_file = str(ROOT / CFG["paths"]["splits_dir"] / f"{args.split}_ids.txt")
            meta_csv = str(ROOT / CFG["paths"]["meta_csv"])
        _cmap_dir = Path(args.cmap_dir) if args.cmap_dir else ROOT / CFG["paths"]["cmap_dir"]
        ds = ProteinDataset(
            ids_file      = ids_file,
            meta_csv      = meta_csv,
            embed_dir     = ROOT / CFG["paths"]["embed_dir"],
            cmap_dir      = _cmap_dir,
            label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
        )
        bs = args.batch_size or CFG["train"]["batch_size"]
        loader = DataLoader(ds, batch_size=bs,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn, pin_memory=True)
    print(f"평가 셋({args.split}): {len(ds):,}개")

    # Hierarchical constraint 매핑 빌드
    hier_maps = None
    if args.hierarchical_inference:
        M12 = build_parent_child_matrix(encoders, 1, 2)
        M23 = build_parent_child_matrix(encoders, 2, 3)
        M34 = build_parent_child_matrix(encoders, 3, 4)
        hier_maps = (M12, M23, M34)
        print("추론 방식: Hierarchical Constraint (L1→L2→L3 마스킹, L4는 자유 sigmoid)")
    else:
        print("추론 방식: 독립 argmax (기본)")

    # 추론
    preds_l13, labels_all, masks_all, l4_probs, l4_mh_all = run_inference(
        model, loader, args.model, hier_maps=hier_maps
    )

    # 메트릭
    results = compute_all_metrics(
        preds_l13, labels_all, masks_all, l4_probs, l4_mh_all, encoders
    )

    infer_mode = "hierarchical_constraint" if args.hierarchical_inference else "independent"
    print(f"\n=== 평가 결과 [{infer_mode}] ===")

    # L1~L3 단일레이블
    print(f"\n  [L1~L3 단일레이블]")
    print(f"  {'Level':<10} {'Acc':>7} {'Micro F1':>9} {'Macro F1':>9} {'W-F1':>9}  n")
    print("  " + "-" * 58)
    for lvl in ["level1", "level2", "level3"]:
        if lvl not in results:
            continue
        m = results[lvl]
        print(f"  {lvl:<10} {m['accuracy']:>7.4f} {m['micro_f1']:>9.4f} "
              f"{m['macro_f1']:>9.4f} {m['weighted_f1']:>9.4f}  {m['n_samples']:,}")

    # L4 멀티레이블
    if "level4" in results:
        r4 = results["level4"]
        print(f"\n  [L4 멀티레이블 F1  (threshold={r4['threshold']})]")
        print(f"  {'Micro F1':>10} {'Macro F1':>10} {'W-F1':>10} {'Precision':>10} {'Recall':>8}  n")
        print("  " + "-" * 61)
        print(f"  {r4['micro_f1']:>10.4f} {r4['macro_f1']:>10.4f} "
              f"{r4['weighted_f1']:>10.4f} {r4['precision']:>10.4f} "
              f"{r4['recall']:>8.4f}  {r4['n_samples']:,}")

        # CLEAN 비교
        is_price = (args.split == "price149")
        print(f"\n  [비교 — CLEAN (Yu et al., Science 2023)]")
        print(f"  {'모델':<30} {'Micro F1 (L4)':>14}  {'데이터셋':>12}")
        print(f"  {'-'*60}")
        dataset_tag = "Price-149 (fair, 0% overlap)" if is_price else "내부 Test set"
        print(f"  {'Ours':<30} {r4['micro_f1']:>14.4f}  {dataset_tag}")
        if is_price:
            print(f"  {'CLEAN (Price-149)':<30} {'0.499':>14}  Price-149 (fair, 0% overlap)")
            print(f"  {'CLEAN (New-392)':<30} {'0.533':>14}  New-392 (ref)")
        else:
            print(f"  {'CLEAN (New-392)':<30} {'0.533':>14}  New-392")
            print(f"  {'CLEAN (Price-149)':<30} {'0.499':>14}  Price-149")
        print(f"\n  ※ CLEAN F1 출처: Yu et al. Science 2023, max-sep threshold")

    # Underrepresented EC 평가 (멀티레이블)
    rare_classes = get_rare_classes(
        str(ROOT / CFG["paths"]["meta_csv"]),
        str(ROOT / CFG["paths"]["splits_dir"]),
        threshold=25
    )
    if rare_classes:
        rare_idx = np.array(sorted(rare_classes))
        valid_mask = np.array([m[3] == 1 for m in masks_all])
        lp_rare = l4_probs[valid_mask][:, rare_idx]
        lt_rare = l4_mh_all[valid_mask][:, rare_idx]
        pred_rare = (lp_rare >= 0.5).astype(np.int32)
        target_rare = lt_rare.astype(np.int32)
        rare_micro = f1_score(target_rare, pred_rare, average="micro", zero_division=0)
        print(f"\n  [Underrepresented EC (N≤25, Level4, 멀티레이블)]")
        print(f"  Micro F1 = {rare_micro:.4f}  ({len(rare_classes)} classes)")
        results["level4_underrepresented"] = {
            "n_classes": len(rare_classes),
            "micro_f1":  round(rare_micro, 4),
        }

    # 저장
    out_dir = ROOT / CFG["paths"]["result_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_hier" if args.hierarchical_inference else ""
    out_file = out_dir / f"{args.model}_{args.split}{suffix}_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n결과 저장: {out_file}")


if __name__ == "__main__":
    main()
