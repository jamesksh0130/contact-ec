#!/usr/bin/env python
"""Foldseek-disjoint failure-mode audit using cached seed-repeat probabilities.

This analysis is CPU-only once seed-repeat probability caches exist. It asks
which proteins are rescued by sequence--structure fusion and which failures are
explained by unseen or rare Level-4 labels in the Foldseek training partition.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "B1_ESM2_only": "B1 ESM-2 only",
    "B3_contact_only": "B3 contact only",
    "Fusion_ESM2_contact": "Fusion ESM-2 + contact",
}
VAL_THRESHOLDS = {
    "B1_ESM2_only": 0.13,
    "B3_contact_only": 0.07,
    "Fusion_ESM2_contact": 0.08,
}
FREQ_BINS = [
    ("unseen", 0, 0),
    ("1-5", 1, 5),
    ("6-25", 6, 25),
    ("26-100", 26, 100),
    (">100", 101, None),
]
L1_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}


def load_cfg(config: str) -> dict:
    with open(ROOT / "configs" / config) as f:
        return yaml.safe_load(f)


def parse_l4_indices(value) -> list[int]:
    text = str(value).strip().strip("[]")
    if not text or text == "nan":
        return []
    sep = "|" if "|" in text else ","
    out = []
    for part in text.split(sep):
        part = part.strip().strip("[]")
        if part:
            out.append(int(float(part)))
    return out


def per_sample_f1(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    labels = labels.astype(bool, copy=False)
    preds = preds.astype(bool, copy=False)
    tp = np.logical_and(labels, preds).sum(axis=1)
    fp = np.logical_and(~labels, preds).sum(axis=1)
    fn = np.logical_and(labels, ~preds).sum(axis=1)
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=np.float64), where=denom > 0)


def micro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    labels = labels.astype(bool, copy=False)
    preds = preds.astype(bool, copy=False)
    tp = float(np.logical_and(labels, preds).sum())
    fp = float(np.logical_and(~labels, preds).sum())
    fn = float(np.logical_and(labels, ~preds).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def frequency_bin(min_seen_count: int) -> str:
    for name, lo, hi in FREQ_BINS:
        if hi is None:
            if min_seen_count >= lo:
                return name
        elif lo <= min_seen_count <= hi:
            return name
    return "unknown"


def load_metadata(
    cfg: dict,
    split_prefix: str,
    split: str,
    n_l4: int,
    l1_classes: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_csv(ROOT / cfg["paths"]["meta_csv"]).set_index("accession")
    train_ids = (ROOT / cfg["paths"]["splits_dir"] / f"{split_prefix}train_ids.txt").read_text().splitlines()
    test_ids = (ROOT / cfg["paths"]["splits_dir"] / f"{split}_ids.txt").read_text().splitlines()

    train_counts = np.zeros(n_l4, dtype=np.int64)
    for acc in train_ids:
        if acc not in meta.index:
            continue
        for idx in parse_l4_indices(meta.at[acc, "l4_all_idxs"]):
            if 0 <= idx < n_l4:
                train_counts[idx] += 1

    rows = []
    for acc in test_ids:
        row = meta.loc[acc]
        if int(row.get("m4", 0)) != 1:
            continue
        idxs = [idx for idx in parse_l4_indices(row["l4_all_idxs"]) if 0 <= idx < n_l4]
        counts = train_counts[idxs] if idxs else np.array([], dtype=np.int64)
        min_count = int(counts.min()) if len(counts) else -1
        l1_idx = int(row["l1_idx"])
        l1 = str(l1_classes[l1_idx]) if 0 <= l1_idx < len(l1_classes) else str(l1_idx)
        rows.append(
            {
                "accession": acc,
                "l1": l1,
                "l1_name": L1_NAMES.get(l1, l1),
                "sequence_length": int(len(str(row["sequence"]))),
                "true_label_count": int(len(idxs)),
                "all_true_labels_seen": bool(len(counts) and np.all(counts > 0)),
                "any_unseen_true_label": bool(len(counts) and np.any(counts == 0)),
                "min_true_label_train_count": min_count,
                "difficulty_bin": frequency_bin(min_count),
            }
        )
    return pd.DataFrame(rows), train_counts


def load_runs(cache_dir: Path, model: str, split: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    paths = sorted(cache_dir.glob(f"{model}_seed*_{split}_l4_probs_labels.npz"))
    runs = []
    for path in paths:
        label = path.name.replace(f"{model}_", "").replace(f"_{split}_l4_probs_labels.npz", "")
        data = np.load(path)
        runs.append((label, data["probs"].astype(np.float32), data["labels"].astype(np.int32)))
    if not runs:
        raise FileNotFoundError(f"No cached probabilities found for {model} in {cache_dir}")
    return runs


def summarize_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col, dropna=False):
        row = {
            group_col: group,
            "n_samples": int(len(sub)),
            "mean_true_labels": float(sub["true_label_count"].mean()),
            "any_unseen_fraction": float(sub["any_unseen_true_label"].mean()),
        }
        for model in MODELS:
            row[f"{model}_hit_mean"] = float(sub[f"{model}_hit_rate"].mean())
            row[f"{model}_sample_f1_mean"] = float(sub[f"{model}_sample_f1_mean"].mean())
        row["fusion_minus_b1_hit"] = row["Fusion_ESM2_contact_hit_mean"] - row["B1_ESM2_only_hit_mean"]
        row["fusion_minus_b3_hit"] = row["Fusion_ESM2_contact_hit_mean"] - row["B3_contact_only_hit_mean"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_samples", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_foldseek.yaml")
    parser.add_argument("--split", default="foldseek_tmscore50_cc_test")
    parser.add_argument("--split-prefix", default="foldseek_tmscore50_cc_")
    parser.add_argument("--cache-dir", default="outputs/cache/foldseek_seed_repeats")
    parser.add_argument("--out-dir", default="outputs/audit")
    parser.add_argument("--output-prefix", default="foldseek_tmscore50_cc_failure_modes")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    with open(ROOT / cfg["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_l4 = len(encoders["level4"].classes_)
    l1_classes = list(encoders["level1"].classes_)
    sample_df, train_counts = load_metadata(cfg, args.split_prefix, args.split, n_l4, l1_classes)

    cache_dir = ROOT / args.cache_dir
    labels_ref = None
    run_metrics = []

    for model, display in MODELS.items():
        runs = load_runs(cache_dir, model, args.split)
        hit_cols = []
        f1_cols = []
        pred_cols = []
        top1_cols = []
        for run_label, probs, labels in runs:
            if labels_ref is None:
                labels_ref = labels
            elif not np.array_equal(labels_ref, labels):
                raise RuntimeError(f"Label matrix mismatch for {model} {run_label}")
            preds = (probs >= VAL_THRESHOLDS[model]).astype(np.int32)
            hit = ((labels * preds).sum(axis=1) > 0).astype(np.float32)
            f1 = per_sample_f1(labels, preds)
            pred_count = preds.sum(axis=1).astype(np.float32)
            top1 = np.zeros_like(preds)
            top1[np.arange(probs.shape[0]), np.argmax(probs, axis=1)] = 1
            top1_hit = ((labels * top1).sum(axis=1) > 0).astype(np.float32)

            hit_col = f"{model}_{run_label}_hit"
            f1_col = f"{model}_{run_label}_sample_f1"
            pred_col = f"{model}_{run_label}_pred_count"
            top1_col = f"{model}_{run_label}_top1_hit"
            sample_df[hit_col] = hit
            sample_df[f1_col] = f1
            sample_df[pred_col] = pred_count
            sample_df[top1_col] = top1_hit
            hit_cols.append(hit_col)
            f1_cols.append(f1_col)
            pred_cols.append(pred_col)
            top1_cols.append(top1_col)

            run_metrics.append(
                {
                    "model": model,
                    "display_model": display,
                    "run": run_label,
                    "threshold": VAL_THRESHOLDS[model],
                    "micro_f1": micro_f1(labels, preds),
                    "sample_hit_rate": float(hit.mean()),
                    "sample_f1_mean": float(f1.mean()),
                    "avg_pred_labels": float(pred_count.mean()),
                    "top1_hit_rate": float(top1_hit.mean()),
                }
            )

        sample_df[f"{model}_hit_rate"] = sample_df[hit_cols].mean(axis=1)
        sample_df[f"{model}_sample_f1_mean"] = sample_df[f1_cols].mean(axis=1)
        sample_df[f"{model}_pred_count_mean"] = sample_df[pred_cols].mean(axis=1)
        sample_df[f"{model}_top1_hit_rate"] = sample_df[top1_cols].mean(axis=1)

    sample_df["fusion_rescue_vs_b1"] = (
        (sample_df["Fusion_ESM2_contact_hit_rate"] > 0)
        & (sample_df["B1_ESM2_only_hit_rate"] == 0)
    )
    sample_df["fusion_rescue_vs_b3"] = (
        (sample_df["Fusion_ESM2_contact_hit_rate"] > 0)
        & (sample_df["B3_contact_only_hit_rate"] == 0)
    )
    sample_df["fusion_only_rescue"] = (
        (sample_df["Fusion_ESM2_contact_hit_rate"] > 0)
        & (sample_df["B1_ESM2_only_hit_rate"] == 0)
        & (sample_df["B3_contact_only_hit_rate"] == 0)
    )
    sample_df["all_models_fail"] = (
        (sample_df["Fusion_ESM2_contact_hit_rate"] == 0)
        & (sample_df["B1_ESM2_only_hit_rate"] == 0)
        & (sample_df["B3_contact_only_hit_rate"] == 0)
    )

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / f"{args.output_prefix}_per_sample.csv"
    run_path = out_dir / f"{args.output_prefix}_run_metrics.csv"
    bin_path = out_dir / f"{args.output_prefix}_by_difficulty_bin.csv"
    family_path = out_dir / f"{args.output_prefix}_by_l1_family.csv"
    json_path = out_dir / f"{args.output_prefix}.json"
    md_path = out_dir / f"{args.output_prefix}.md"

    by_bin = summarize_group(sample_df, "difficulty_bin")
    by_family = summarize_group(sample_df, "l1_name")
    run_df = pd.DataFrame(run_metrics)

    sample_df.to_csv(sample_path, index=False)
    run_df.to_csv(run_path, index=False)
    by_bin.to_csv(bin_path, index=False)
    by_family.to_csv(family_path, index=False)

    headline = {
        "n_samples": int(len(sample_df)),
        "fusion_rescue_vs_b1_fraction": float(sample_df["fusion_rescue_vs_b1"].mean()),
        "fusion_rescue_vs_b3_fraction": float(sample_df["fusion_rescue_vs_b3"].mean()),
        "fusion_only_rescue_fraction": float(sample_df["fusion_only_rescue"].mean()),
        "all_models_fail_fraction": float(sample_df["all_models_fail"].mean()),
        "all_models_fail_any_unseen_fraction": float(
            sample_df.loc[sample_df["all_models_fail"], "any_unseen_true_label"].mean()
        ),
        "fusion_only_rescue_any_unseen_fraction": float(
            sample_df.loc[sample_df["fusion_only_rescue"], "any_unseen_true_label"].mean()
        )
        if sample_df["fusion_only_rescue"].any()
        else 0.0,
    }
    json_path.write_text(json.dumps(headline, indent=2))

    md = [
        "# Foldseek Failure-Mode Audit",
        "",
        f"- Split: `{args.split}`",
        f"- Samples: {headline['n_samples']:,}",
        f"- Fusion rescues vs B1: {headline['fusion_rescue_vs_b1_fraction']:.3f}",
        f"- Fusion rescues vs B3: {headline['fusion_rescue_vs_b3_fraction']:.3f}",
        f"- Fusion-only rescues: {headline['fusion_only_rescue_fraction']:.3f}",
        f"- All-model failures: {headline['all_models_fail_fraction']:.3f}",
        f"- All-model failures with any unseen true label: {headline['all_models_fail_any_unseen_fraction']:.3f}",
        "",
        "## Run Metrics",
        "",
        run_df.to_markdown(index=False),
        "",
        "## By Difficulty Bin",
        "",
        by_bin.to_markdown(index=False),
        "",
        "## By EC Level-1 Family",
        "",
        by_family.to_markdown(index=False),
        "",
    ]
    md_path.write_text("\n".join(md))
    print(json.dumps(headline, indent=2))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
