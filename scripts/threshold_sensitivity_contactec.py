"""
Threshold sensitivity analysis for Contact-EC on the 124 known-label
Swiss-Prot 2023-01 temporal proteins.

This is a diagnostic analysis, not a test-set model-selection protocol:
it quantifies how much the reported fixed-threshold result depends on the
default 0.5 cutoff and whether missed proteins are mainly calibration misses.

Outputs:
  outputs/results/contactec_threshold_sensitivity.csv
  outputs/results/contactec_threshold_sensitivity_summary.json
  outputs/results/contactec_threshold_sensitivity_summary.md
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.dataset import ProteinDataset, collate_fn
from models.fusion_v2_flatfc import FusionV2FlatFC


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_cfg():
    with open(ROOT / "configs" / "config_ecbench.yaml") as f:
        return yaml.safe_load(f)


def build_model(n_classes, cfg):
    return FusionV2FlatFC(
        n_classes,
        esm_dim=cfg["model"]["esm2_dim"],
        contact_dim=cfg["model"]["resnet_out_dim"],
        fusion_dim=cfg["model"]["fusion_dim"],
        dropout=0.0,
    )


@torch.no_grad()
def infer(model, loader):
    model.eval()
    probs, labels, accs = [], [], []
    for batch in tqdm(loader, desc="Contact-EC inference"):
        esm_emb, cmap, _, _, _, l4_mh, batch_accs = batch
        logits = model(esm_emb.to(DEVICE), cmap.to(DEVICE))
        probs.append(torch.sigmoid(logits[3]).cpu().numpy())
        labels.append(l4_mh.numpy().astype(np.int32))
        accs.extend(batch_accs)
    return np.concatenate(probs), np.concatenate(labels), accs


def make_rare_mask(cfg, n_l4):
    proc_dir = ROOT / "data" / "ecbench" / "processed"
    splits_dir = ROOT / cfg["paths"]["splits_dir"]
    train_meta = proc_dir / "train_meta.csv"
    train_ids_f = splits_dir / "train_ids.txt"
    if not train_meta.exists() or not train_ids_f.exists():
        return None

    train_ids = set(train_ids_f.read_text().strip().splitlines())
    meta_df = pd.read_csv(train_meta)
    train_df = meta_df[meta_df["accession"].isin(train_ids)]
    counts = np.zeros(n_l4, dtype=np.int64)
    for idx in train_df["l4_idx"].dropna().astype(int):
        if 0 <= idx < n_l4:
            counts[idx] += 1
    return counts <= 25


def force_top1_for_empty(preds, probs):
    fixed = preds.copy()
    empty = fixed.sum(axis=1) == 0
    if empty.any():
        top1 = np.argmax(probs[empty], axis=1)
        fixed[np.flatnonzero(empty), top1] = 1
    return fixed


def metrics_from_preds(labels, preds, rare_mask=None):
    result = {
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="micro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="micro", zero_division=0)),
        "avg_predictions_per_protein": float(preds.sum(axis=1).mean()),
        "empty_prediction_count": int((preds.sum(axis=1) == 0).sum()),
    }
    if rare_mask is not None and rare_mask.sum() > 0:
        result["rare_ec_f1"] = float(
            f1_score(labels[:, rare_mask], preds[:, rare_mask], average="micro", zero_division=0)
        )
        result["rare_class_count"] = int(rare_mask.sum())
    else:
        result["rare_ec_f1"] = None
        result["rare_class_count"] = None
    return result


def eval_thresholds(probs, labels, rare_mask):
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.05), 2):
        preds = (probs >= threshold).astype(np.int32)
        row = {"method": "global_threshold", "threshold": float(threshold)}
        row.update(metrics_from_preds(labels, preds, rare_mask))
        rows.append(row)

    preds_05 = (probs >= 0.5).astype(np.int32)
    row = {"method": "threshold_0.5_or_top1_if_empty", "threshold": 0.5}
    row.update(metrics_from_preds(labels, force_top1_for_empty(preds_05, probs), rare_mask))
    rows.append(row)

    top1_preds = np.zeros_like(labels, dtype=np.int32)
    top1_preds[np.arange(probs.shape[0]), np.argmax(probs, axis=1)] = 1
    row = {"method": "top1_only", "threshold": None}
    row.update(metrics_from_preds(labels, top1_preds, rare_mask))
    rows.append(row)
    return rows


def main():
    cfg = load_cfg()
    with open(ROOT / cfg["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    n_l4 = n_classes[3]

    ds = ProteinDataset(
        ids_file=str(ROOT / cfg["paths"]["splits_dir"] / "test_ids_full.txt"),
        meta_csv=str(ROOT / "data" / "ecbench" / "processed" / "test_meta_full.csv"),
        embed_dir=str(ROOT / cfg["paths"]["embed_dir"]),
        cmap_dir=str(ROOT / cfg["paths"]["cmap_dir"]),
        label_enc_pkl=str(ROOT / cfg["paths"]["label_enc"]),
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn)

    ckpt_path = ROOT / "outputs" / "checkpoints" / "ecbench_b4_flatfc_best.pt"
    model = build_model(n_classes, cfg).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])

    probs, labels, accs = infer(model, loader)
    rare_mask = make_rare_mask(cfg, n_l4)
    rows = eval_thresholds(probs, labels, rare_mask)

    out_dir = ROOT / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "contactec_threshold_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    global_rows = [r for r in rows if r["method"] == "global_threshold"]
    fixed = next(r for r in global_rows if abs(r["threshold"] - 0.5) < 1e-9)
    best_micro = max(global_rows, key=lambda r: r["micro_f1"])
    best_weighted = max(global_rows, key=lambda r: r["weighted_f1"])
    fallback = next(r for r in rows if r["method"] == "threshold_0.5_or_top1_if_empty")
    top1 = next(r for r in rows if r["method"] == "top1_only")

    near_miss = []
    empty_05 = ((probs >= 0.5).astype(np.int32).sum(axis=1) == 0)
    top1_idx = np.argmax(probs, axis=1)
    for i, uid in enumerate(accs):
        if empty_05[i] and labels[i, top1_idx[i]] == 1:
            near_miss.append({
                "uid": uid,
                "top1_ec": str(encoders["level4"].classes_[top1_idx[i]]),
                "top1_prob": float(probs[i, top1_idx[i]]),
            })
    near_miss = sorted(near_miss, key=lambda x: x["top1_prob"], reverse=True)[:10]

    summary = {
        "n": int(labels.shape[0]),
        "n_l4_classes": int(n_l4),
        "checkpoint": str(ckpt_path),
        "fixed_threshold_0.5": fixed,
        "best_global_threshold_by_micro_f1": best_micro,
        "best_global_threshold_by_weighted_f1": best_weighted,
        "threshold_0.5_or_top1_if_empty": fallback,
        "top1_only": top1,
        "near_miss_empty_at_0.5_but_top1_true": near_miss,
    }
    out_json = out_dir / "contactec_threshold_sensitivity_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    out_md = out_dir / "contactec_threshold_sensitivity_summary.md"
    lines = [
        "# Contact-EC Threshold Sensitivity",
        "",
        f"N = {summary['n']} known-label Swiss-Prot 2023-01 temporal proteins.",
        f"Checkpoint = `{ckpt_path}`.",
        "",
        "## Key Results",
        "",
        "| Setting | Threshold | Micro F1 | Weighted F1 | Precision | Recall | Avg preds/protein | Empty predictions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in [
        ("Fixed global", fixed),
        ("Best global by micro F1", best_micro),
        ("0.5 or top-1 if empty", fallback),
        ("Top-1 only", top1),
    ]:
        thr = "-" if row["threshold"] is None else f"{row['threshold']:.2f}"
        lines.append(
            f"| {name} | {thr} | {row['micro_f1']:.4f} | {row['weighted_f1']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['avg_predictions_per_protein']:.2f} | {row['empty_prediction_count']} |"
        )
    lines.extend(["", "## Near Misses at Threshold 0.5", ""])
    if near_miss:
        lines.extend(["| UID | Top-1 EC | Probability |", "|---|---|---:|"])
        for item in near_miss:
            lines.append(f"| {item['uid']} | {item['top1_ec']} | {item['top1_prob']:.4f} |")
    else:
        lines.append("No empty-at-0.5 protein had a correct top-1 EC prediction.")
    out_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
