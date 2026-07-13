"""
EC-Bench Similarity-stratified Evaluation
CLEAN 논문 방식: 테스트 단백질을 훈련셋 대비 최고 유사도 기준으로 분류
Thresholds: ≤10%, ≤30%, ≤50%, ≤70%, ≤100% (누적)

사용법:
  python scripts/ecbench_eval_by_similarity.py \
    --checkpoints outputs/checkpoints/ecbench_b2_hard_val_best.pt outputs/checkpoints/ecbench_fv2_phase2_best.pt \
    --models b2_esm2_hier fusion_v2 \
    [--split swissprot|price149]
    [--threshold 0.5]
    [--recompute_sim]
"""
import sys, argparse, pickle, json, subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config_ecbench.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import ProteinDataset, collate_fn

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MMSEQS = "/home/user/anaconda3/bin/mmseqs"
TMP    = Path("/tmp/ecbench_sim_eval")

SIM_THRESHOLDS = [0.10, 0.30, 0.50, 0.70, 1.00]
SIM_LABELS     = ["≤10%", "≤30%", "≤50%", "≤70%", "≤100%"]


# ── 모델 빌더 ────────────────────────────────────────────────
def build_model(model_name, n_classes):
    if model_name == "b2_esm2_hier":
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
    else:
        raise ValueError(f"미지원 모델: {model_name}")


# ── MMSeqs2: test vs train similarity ────────────────────────
def compute_similarity(test_ids, train_ids, meta_df, cache_path: Path) -> dict:
    """각 테스트 단백질의 훈련셋 대비 최고 sequence identity 반환."""
    if cache_path.exists():
        print(f"유사도 캐시 로드: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    TMP.mkdir(exist_ok=True)
    import pandas as pd

    # FASTA 작성
    test_fasta  = TMP / "test.fasta"
    train_fasta = TMP / "train.fasta"
    result_tsv  = TMP / "sim_result.tsv"

    with open(test_fasta, "w") as f:
        for uid in test_ids:
            if uid in meta_df.index:
                f.write(f">{uid}\n{meta_df.loc[uid, 'sequence']}\n")

    train_meta = pd.read_csv(ROOT / CFG["paths"]["meta_csv"]).set_index("accession")
    with open(train_fasta, "w") as f:
        for uid in train_ids:
            if uid in train_meta.index:
                f.write(f">{uid}\n{train_meta.loc[uid, 'sequence']}\n")

    print(f"MMSeqs2 실행 중 (test={len(test_ids)}, train={len(train_ids)})...")
    cmd = [
        MMSEQS, "easy-search",
        str(test_fasta), str(train_fasta), str(result_tsv), str(TMP / "tmp"),
        "--min-seq-id", "0.0",
        "--alignment-mode", "3",
        "--format-output", "query,target,fident",
        "-c", "0.8", "--cov-mode", "0",
        "--threads", "16", "-s", "7.5",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[:300])
        raise RuntimeError(f"MMSeqs2 실패 (exit {result.returncode})")

    # max identity per test protein
    import pandas as _pd
    hits = _pd.read_csv(result_tsv, sep="\t", header=None,
                        names=["query", "target", "fident"])
    max_sim = hits.groupby("query")["fident"].max().to_dict()

    # 매칭 없음 → 0.0
    sim_dict = {uid: float(max_sim.get(uid, 0.0)) for uid in test_ids}

    with open(cache_path, "w") as f:
        json.dump(sim_dict, f)
    print(f"유사도 계산 완료, 저장: {cache_path}")
    return sim_dict


# ── 추론 ─────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, dataset, indices, batch_size):
    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=batch_size, shuffle=False,
                         num_workers=4, collate_fn=collate_fn, pin_memory=True)
    probs_list, labels_list = [], []
    for batch in loader:
        esm_emb, cmap, _, _, _, l4_mh, _ = batch
        esm_emb = esm_emb.to(DEVICE)
        cmap    = cmap.to(DEVICE)
        logits  = model(esm_emb, cmap)
        probs_list.append(torch.sigmoid(logits[3]).cpu().numpy())
        labels_list.append(l4_mh.numpy())
    if not probs_list:
        return np.array([]), np.array([])
    return np.concatenate(probs_list), np.concatenate(labels_list)


def f1_at_threshold(probs, labels, threshold):
    if len(probs) == 0:
        return float("nan"), float("nan")
    preds = (probs >= threshold).astype(np.int32)
    micro = f1_score(labels.astype(np.int32), preds, average="micro", zero_division=0)
    macro = f1_score(labels.astype(np.int32), preds, average="macro", zero_division=0)
    return micro, macro


# ── 메인 ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--models",      nargs="+", required=True)
    parser.add_argument("--split",       default="swissprot",
                        choices=["swissprot", "price149"])
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--recompute_sim", action="store_true",
                        help="기존 유사도 캐시 무시하고 재계산")
    args = parser.parse_args()

    assert len(args.checkpoints) == len(args.models), \
        "--checkpoints와 --models 개수가 같아야 합니다"

    # ── 라벨 인코더 ──
    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    print(f"클래스 수: {n_classes}")

    # ── 테스트셋 선택 ──
    if args.split == "swissprot":
        ids_file  = ROOT / CFG["paths"]["splits_dir"] / "test_ids.txt"
        meta_file = ROOT / "data/ecbench/processed/test_meta.csv"
        split_label = "Swiss-Prot 2023-01"
        sim_cache = ROOT / "data/ecbench/splits/test_vs_train_sim.json"
    else:
        ids_file  = ROOT / CFG["paths"]["splits_dir"] / "price149_ids.txt"
        meta_file = ROOT / "data/ecbench/processed/price149_meta.csv"
        split_label = "Price-149"
        sim_cache = ROOT / "data/ecbench/splits/price149_vs_train_sim.json"

    test_ids = ids_file.read_text().strip().split("\n")
    train_ids = (ROOT / CFG["paths"]["splits_dir"] / "train_ids.txt").read_text().strip().split("\n")

    import pandas as pd
    test_meta = pd.read_csv(meta_file).set_index("accession")

    # ── 데이터셋 ──
    dataset = ProteinDataset(
        ids_file      = str(ids_file),
        meta_csv      = str(meta_file),
        embed_dir     = ROOT / CFG["paths"]["embed_dir"],
        cmap_dir      = ROOT / CFG["paths"]["cmap_dir"],
        label_enc_pkl = ROOT / CFG["paths"]["label_enc"],
    )

    # ── MMSeqs2 유사도 계산 ──
    if args.recompute_sim and sim_cache.exists():
        sim_cache.unlink()
    sim_dict = compute_similarity(test_ids, train_ids, test_meta, sim_cache)

    # ── similarity threshold별 인덱스 ──
    # 각 threshold: max_sim <= threshold인 테스트 단백질 (누적)
    threshold_indices = {}
    for thr in SIM_THRESHOLDS:
        idxs = [i for i, uid in enumerate(test_ids)
                if sim_dict.get(uid, 0.0) <= thr]
        threshold_indices[thr] = idxs

    # ── 각 threshold별 분포 출력 ──
    print(f"\n[{split_label}] 유사도 분포")
    print(f"  {'Threshold':<12} {'N':>6}  {'비율':>7}")
    for thr, lbl in zip(SIM_THRESHOLDS, SIM_LABELS):
        n = len(threshold_indices[thr])
        print(f"  {lbl:<12} {n:>6}  ({n/len(test_ids)*100:.1f}%)")

    # ── 모델별 평가 ──
    all_results = {}

    for ckpt_path, model_name in zip(args.checkpoints, args.models):
        print(f"\n{'='*60}")
        print(f"모델: {model_name}  |  체크포인트: {ckpt_path}")

        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = build_model(model_name, n_classes).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()

        if "micro_f1" in ckpt:
            print(f"  (val micro_f1={ckpt['micro_f1']:.4f}, epoch={ckpt.get('epoch','?')})")

        results_per_thr = {}
        print(f"\n  {'Threshold':<10} {'N':>6} {'Micro F1':>10} {'Macro F1':>10}")
        print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*10}")

        for thr, lbl in zip(SIM_THRESHOLDS, SIM_LABELS):
            idxs = threshold_indices[thr]
            if len(idxs) == 0:
                print(f"  {lbl:<10} {0:>6} {'N/A':>10} {'N/A':>10}")
                results_per_thr[lbl] = {"n": 0, "micro_f1": None, "macro_f1": None}
                continue

            probs, labels = run_inference(model, dataset, idxs, args.batch_size)
            micro, macro  = f1_at_threshold(probs, labels, args.threshold)
            print(f"  {lbl:<10} {len(idxs):>6} {micro:>10.4f} {macro:>10.4f}")
            results_per_thr[lbl] = {
                "n": len(idxs),
                "micro_f1": round(float(micro), 4),
                "macro_f1": round(float(macro), 4),
            }

        all_results[model_name] = results_per_thr

    # ── 비교 테이블 출력 ──
    print(f"\n\n{'='*60}")
    print(f"  Similarity-stratified 비교 ({split_label})")
    print(f"{'='*60}")

    header = f"  {'Threshold':<10}"
    for model_name in all_results:
        header += f"  {model_name[:14]:>14}"
    print(header)
    print(f"  {'-'*10}" + f"  {'-'*14}" * len(all_results))

    for lbl in SIM_LABELS:
        row = f"  {lbl:<10}"
        for model_name in all_results:
            val = all_results[model_name].get(lbl, {})
            f1  = val.get("micro_f1")
            row += f"  {f1:>14.4f}" if f1 is not None else f"  {'N/A':>14}"
        print(row)

    # ── 결과 저장 ──
    out = {
        "split":      split_label,
        "threshold":  args.threshold,
        "models":     all_results,
        "sim_distribution": {
            lbl: len(threshold_indices[thr])
            for thr, lbl in zip(SIM_THRESHOLDS, SIM_LABELS)
        }
    }
    out_path = ROOT / CFG["paths"]["result_dir"] / f"sim_eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
