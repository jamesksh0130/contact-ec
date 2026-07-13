"""
Bootstrap 95% CI + McNemar Test (Bioinformatics 수준 통계 검증)

1. 4개 모델 전체에 대해 Bootstrap CI (N=1000 resampling, seed=42)
   - Swiss-Prot 2023-01 temporal test (N=124)
   - 95% CI: (2.5 percentile, 97.5 percentile)

2. McNemar test: B3 vs B1, Contact-EC vs B1 per-sample 비교
   - per-sample F1>0 여부로 '부분 정답' 판정
   - Edwards continuity correction 포함

사용법:
  CUDA_VISIBLE_DEVICES=1 python scripts/run_stats.py
"""
import sys, pickle, json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from scipy.stats import chi2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config_ecbench.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import ProteinDataset, collate_fn

DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"
LABEL_ENC = ROOT / CFG["paths"]["label_enc"]
SPLITS    = ROOT / CFG["paths"]["splits_dir"]
PROC_DIR  = ROOT / "data" / "ecbench" / "processed"

N_BOOTSTRAP = 1000
SEED        = 42

MODELS = {
    "B1 (ESM-2)":   ("b1_esm2_fc",       "outputs/checkpoints/ecbench_b1_best.pt"),
    "B2 (Hier.FC)": ("b2_esm2_hier",     "outputs/checkpoints/ecbench_b2_hard_val_best.pt"),
    "B3 (Contact)": ("b3_contact",       "outputs/checkpoints/ecbench_b3_phase1_best.pt"),
    "B4 (GCA)":     ("fusion_v2_flatfc", "outputs/checkpoints/ecbench_b4_flatfc_best.pt"),
    "Contact-EC":   ("fusion_v2",        "outputs/checkpoints/ecbench_fv2_phase2_best.pt"),
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
    elif name == "fusion_v2_flatfc":
        from models.fusion_v2_flatfc import FusionV2FlatFC
        return FusionV2FlatFC(n_classes,
                              esm_dim=CFG["model"]["esm2_dim"],
                              contact_dim=CFG["model"]["resnet_out_dim"],
                              fusion_dim=CFG["model"]["fusion_dim"],
                              dropout=0.0)
    elif name == "fusion_v2":
        from models.fusion_v2 import FusionModelV2
        return FusionModelV2(n_classes,
                             esm_dim=CFG["model"]["esm2_dim"],
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=0.0)


@torch.no_grad()
def run_inference(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="    추론", leave=False):
        esm_emb, cmap, _, _, _, l4_mh, _ = batch
        logits = model(esm_emb.to(DEVICE), cmap.to(DEVICE))
        all_probs.append(torch.sigmoid(logits[3]).cpu().numpy())
        all_labels.append(l4_mh.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def bootstrap_ci(probs, labels, n_boot=N_BOOTSTRAP, seed=SEED, thr=0.5):
    n         = len(probs)
    rng       = np.random.RandomState(seed)
    preds     = (probs >= thr).astype(np.int32)
    point_est = float(f1_score(labels, preds, average="micro", zero_division=0))
    scores    = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(float(f1_score(labels[idx].astype(np.int32),
                                     preds[idx],
                                     average="micro", zero_division=0)))
    scores = np.array(scores)
    return point_est, float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def sample_correct(probs, labels, thr=0.5):
    """샘플별 TP>0 여부 (부분 정답)."""
    preds = (probs >= thr).astype(np.float32)
    return np.array([(p * g).sum() > 0 for p, g in zip(preds, labels)], dtype=bool)


def mcnemar_test(corr_a, corr_b):
    b = int((~corr_a & corr_b).sum())
    c = int((corr_a & ~corr_b).sum())
    if b + c == 0:
        return b, c, 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p    = float(1 - chi2.cdf(stat, df=1))
    return b, c, float(stat), p


def main():
    with open(LABEL_ENC, "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    print(f"L4 클래스: {n_classes[3]}")

    ds = ProteinDataset(
        ids_file     = str(SPLITS  / "test_ids_full.txt"),
        meta_csv     = str(PROC_DIR / "test_meta_full.csv"),
        embed_dir    = ROOT / CFG["paths"]["embed_dir"],
        cmap_dir     = ROOT / CFG["paths"]["cmap_dir"],
        label_enc_pkl= LABEL_ENC,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)
    print(f"테스트셋: N={len(ds)}")

    probs_dict  = {}
    labels_ref  = None
    ci_results  = {}

    for label, (mname, ckpt_path) in MODELS.items():
        print(f"\n[{label}]")
        ckpt  = torch.load(ROOT / ckpt_path, map_location=DEVICE, weights_only=False)
        model = build_model(mname, n_classes).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        probs, labels = run_inference(model, loader)
        if labels_ref is None:
            labels_ref = labels
        probs_dict[label] = probs
        pt, lo, hi = bootstrap_ci(probs, labels)
        ci_results[label] = {"point_est": round(pt,4),
                             "ci_lower":  round(lo,4),
                             "ci_upper":  round(hi,4)}
        print(f"  Micro F1={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        del model; torch.cuda.empty_cache()

    # ── 테이블 출력 ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Bootstrap 95% CI  (N=1000, seed=42, N_test=124)")
    print(f"{'='*65}")
    print(f"  {'모델':<18} {'Micro F1':>10}  {'95% CI'}")
    print(f"  {'-'*50}")
    for lbl, r in ci_results.items():
        print(f"  {lbl:<18} {r['point_est']:>10.4f}  [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")

    # ── McNemar ─────────────────────────────────────────────────────────────
    mc_results = {}
    pairs = [("B3 (Contact)", "B1 (ESM-2)"),
             ("B4 (GCA)",    "B1 (ESM-2)"),
             ("B4 (GCA)",    "B3 (Contact)"),
             ("B4 (GCA)",    "Contact-EC"),
             ("Contact-EC",  "B1 (ESM-2)"),
             ("Contact-EC",  "B3 (Contact)")]

    print(f"\n{'='*65}")
    print("  McNemar Test (Edwards continuity correction)")
    print(f"{'='*65}")
    for name_a, name_b in pairs:
        corr_a = sample_correct(probs_dict[name_a], labels_ref)
        corr_b = sample_correct(probs_dict[name_b], labels_ref)
        b, c, stat, p = mcnemar_test(corr_a, corr_b)
        sig = "유의 *" if p < 0.05 else "비유의"
        print(f"  {name_a} vs {name_b}")
        print(f"    b(A틀 B맞)={b}  c(A맞 B틀)={c}  χ²={stat:.3f}  p={p:.4f}  {sig}")
        key = f"{name_a}_vs_{name_b}"
        mc_results[key] = {"b": b, "c": c, "chi2": round(stat,4),
                            "p_value": round(p,4), "significant": p < 0.05}

    # ── 저장 ─────────────────────────────────────────────────────────────────
    out = ROOT / "outputs" / "results" / "stats_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "bootstrap_ci": {"n_bootstrap": N_BOOTSTRAP, "seed": SEED,
                             "n_test": len(labels_ref), "results": ci_results},
            "mcnemar": mc_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
