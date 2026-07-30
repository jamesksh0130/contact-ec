#!/usr/bin/env python
"""Label coverage, frequency-bin, and EC-family audit for Foldseek split.

Uses cached Level-4 probabilities from `audit_foldseek_threshold_topk.py`.
No model training or inference is required once those caches exist.
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

MODELS = ["B1_ESM2_only", "B3_contact_only", "Fusion_ESM2_contact"]
MODEL_DISPLAY = {
    "B1_ESM2_only": "B1 ESM-2 only",
    "B3_contact_only": "B3 contact only",
    "Fusion_ESM2_contact": "Fusion ESM-2 + contact",
}
FREQ_BINS = [
    ("unseen", 0, 0),
    ("1-5", 1, 5),
    ("6-25", 6, 25),
    ("26-100", 26, 100),
    (">100", 101, None),
]
EC_L1_NAMES = {
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


def micro_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    labels = labels.astype(bool, copy=False)
    preds = preds.astype(bool, copy=False)
    tp_per_class = np.logical_and(labels, preds).sum(axis=0, dtype=np.float64)
    fp_per_class = np.logical_and(~labels, preds).sum(axis=0, dtype=np.float64)
    fn_per_class = np.logical_and(labels, ~preds).sum(axis=0, dtype=np.float64)
    tp = float(tp_per_class.sum())
    fp = float(fp_per_class.sum())
    fn = float(fn_per_class.sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    denom = 2 * tp_per_class + fp_per_class + fn_per_class
    per_class_f1 = np.divide(
        2 * tp_per_class,
        denom,
        out=np.zeros_like(tp_per_class, dtype=np.float64),
        where=denom > 0,
    )
    return {
        "micro_f1": micro_f1,
        "macro_f1": float(per_class_f1.mean()) if per_class_f1.size else 0.0,
        "precision": precision,
        "recall": recall,
        "support": int(labels.sum()),
        "predicted": int(preds.sum()),
    }


def per_sample_f1(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    labels = labels.astype(bool, copy=False)
    preds = preds.astype(bool, copy=False)
    tp = np.logical_and(labels, preds).sum(axis=1)
    fp = np.logical_and(~labels, preds).sum(axis=1)
    fn = np.logical_and(labels, ~preds).sum(axis=1)
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=np.float64), where=denom > 0)


def prepare_metadata(cfg: dict, split_prefix: str, split: str, n_l4: int) -> dict:
    meta = pd.read_csv(ROOT / cfg["paths"]["meta_csv"]).set_index("accession")
    train_ids = (ROOT / cfg["paths"]["splits_dir"] / f"{split_prefix}train_ids.txt").read_text().splitlines()
    test_ids_all = (ROOT / cfg["paths"]["splits_dir"] / f"{split}_ids.txt").read_text().splitlines()

    train_counts = np.zeros(n_l4, dtype=np.int64)
    for acc in train_ids:
        if acc not in meta.index:
            continue
        for idx in parse_l4_indices(meta.at[acc, "l4_all_idxs"]):
            if 0 <= idx < n_l4:
                train_counts[idx] += 1

    valid_test_ids = []
    l1_idx = []
    for acc in test_ids_all:
        if acc not in meta.index:
            continue
        row = meta.loc[acc]
        if int(row.get("m4", 0)) == 1:
            valid_test_ids.append(acc)
            l1_idx.append(int(row["l1_idx"]))

    return {
        "meta": meta,
        "train_counts": train_counts,
        "test_ids": valid_test_ids,
        "test_l1_idx": np.array(l1_idx, dtype=np.int64),
    }


def coverage_rows(labels: np.ndarray, train_counts: np.ndarray) -> list[dict]:
    rows = []
    total_pos = int(labels.sum())
    for name, lo, hi in FREQ_BINS:
        if hi is None:
            cls_mask = train_counts >= lo
        else:
            cls_mask = (train_counts >= lo) & (train_counts <= hi)
        pos = int(labels[:, cls_mask].sum())
        rows.append(
            {
                "bin": name,
                "class_count": int(cls_mask.sum()),
                "positive_labels": pos,
                "positive_label_fraction": pos / total_pos if total_pos else 0.0,
            }
        )

    per_sample_counts = labels @ (train_counts > 0).astype(np.int32)
    total_per_sample = labels.sum(axis=1)
    rows.append(
        {
            "bin": "sample_all_labels_seen",
            "class_count": "",
            "positive_labels": int((per_sample_counts == total_per_sample).sum()),
            "positive_label_fraction": float((per_sample_counts == total_per_sample).mean()),
        }
    )
    rows.append(
        {
            "bin": "sample_any_unseen_label",
            "class_count": "",
            "positive_labels": int((per_sample_counts < total_per_sample).sum()),
            "positive_label_fraction": float((per_sample_counts < total_per_sample).mean()),
        }
    )
    return rows


def run(args: argparse.Namespace) -> None:
    cfg = load_cfg(args.config)
    with open(ROOT / cfg["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_l4 = len(encoders["level4"].classes_)
    l1_classes = list(encoders["level1"].classes_)
    prepared = prepare_metadata(cfg, args.split_prefix, args.split, n_l4)
    train_counts = prepared["train_counts"]
    test_l1_idx = prepared["test_l1_idx"]

    best_threshold = pd.read_csv(ROOT / args.best_threshold_csv).set_index("model")["threshold"].to_dict()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    coverage_written = False
    freq_rows = []
    family_rows = []
    sample_rows = []

    for model in MODELS:
        cache = ROOT / args.cache_dir / f"{model}_{args.split}_l4_probs_labels.npz"
        data = np.load(cache)
        probs = data["probs"].astype(np.float32)
        labels = data["labels"].astype(np.int32)
        if len(test_l1_idx) != labels.shape[0]:
            raise RuntimeError(
                f"Metadata/test label length mismatch: {len(test_l1_idx)} vs {labels.shape[0]}"
            )
        threshold = float(best_threshold[model])
        preds = (probs >= threshold).astype(np.int32)

        if not coverage_written:
            pd.DataFrame(coverage_rows(labels, train_counts)).to_csv(
                out_dir / f"{args.output_prefix}_label_coverage.csv", index=False
            )
            coverage_written = True

        for name, lo, hi in FREQ_BINS:
            if hi is None:
                cls_mask = train_counts >= lo
            else:
                cls_mask = (train_counts >= lo) & (train_counts <= hi)
            if not cls_mask.any():
                continue
            metrics = micro_metrics(labels[:, cls_mask], preds[:, cls_mask])
            freq_rows.append(
                {
                    "model": model,
                    "display_model": MODEL_DISPLAY[model],
                    "threshold": threshold,
                    "frequency_bin": name,
                    "class_count": int(cls_mask.sum()),
                    **metrics,
                }
            )

        for l1_i, l1_name in enumerate(l1_classes):
            sample_mask = test_l1_idx == l1_i
            if not sample_mask.any():
                continue
            metrics = micro_metrics(labels[sample_mask], preds[sample_mask])
            sample_f1 = per_sample_f1(labels[sample_mask], preds[sample_mask])
            family_rows.append(
                {
                    "model": model,
                    "display_model": MODEL_DISPLAY[model],
                    "threshold": threshold,
                    "l1": l1_name,
                    "l1_name": EC_L1_NAMES.get(l1_name, l1_name),
                    "n_samples": int(sample_mask.sum()),
                    "mean_sample_f1": float(sample_f1.mean()),
                    "median_sample_f1": float(np.median(sample_f1)),
                    **metrics,
                }
            )

        sample_f1_all = per_sample_f1(labels, preds)
        true_label_counts = labels.sum(axis=1)
        seen_label_counts = labels @ (train_counts > 0).astype(np.int32)
        min_train_count = np.where(labels.astype(bool), train_counts[None, :], np.inf).min(axis=1)
        min_train_count[np.isinf(min_train_count)] = -1
        for i, acc in enumerate(prepared["test_ids"]):
            sample_rows.append(
                {
                    "model": model,
                    "accession": acc,
                    "threshold": threshold,
                    "l1_idx": int(test_l1_idx[i]),
                    "l1": l1_classes[int(test_l1_idx[i])],
                    "true_label_count": int(true_label_counts[i]),
                    "seen_label_count": int(seen_label_counts[i]),
                    "has_unseen_label": bool(seen_label_counts[i] < true_label_counts[i]),
                    "min_train_count_of_true_labels": int(min_train_count[i]),
                    "sample_f1": float(sample_f1_all[i]),
                    "predicted_label_count": int(preds[i].sum()),
                }
            )

    freq_df = pd.DataFrame(freq_rows)
    family_df = pd.DataFrame(family_rows)
    sample_df = pd.DataFrame(sample_rows)
    freq_df.to_csv(out_dir / f"{args.output_prefix}_frequency_bin_performance.csv", index=False)
    family_df.to_csv(out_dir / f"{args.output_prefix}_l1_family_performance.csv", index=False)
    sample_df.to_csv(out_dir / f"{args.output_prefix}_per_sample_difficulty.csv", index=False)

    payload = {
        "split": args.split,
        "split_prefix": args.split_prefix,
        "n_l4": n_l4,
        "train_l4_seen_classes": int((train_counts > 0).sum()),
        "train_l4_unseen_classes": int((train_counts == 0).sum()),
        "best_thresholds": {k: float(v) for k, v in best_threshold.items()},
        "frequency_best": freq_df.to_dict(orient="records"),
        "family_best": family_df.to_dict(orient="records"),
    }
    (out_dir / f"{args.output_prefix}_label_difficulty_audit.json").write_text(json.dumps(payload, indent=2))

    coverage = pd.read_csv(out_dir / f"{args.output_prefix}_label_coverage.csv")
    md = [
        "# Foldseek Label Difficulty Audit",
        "",
        f"- Split: `{args.split}`",
        f"- Train prefix: `{args.split_prefix}`",
        f"- L4 classes seen in foldseek train: {(train_counts > 0).sum():,} / {n_l4:,}",
        "",
        "## Label Coverage",
        "",
        "| Bin | Classes | Positive labels | Fraction |",
        "|---|---:|---:|---:|",
    ]
    for _, r in coverage.iterrows():
        cls = "" if pd.isna(r["class_count"]) else r["class_count"]
        md.append(
            f"| {r['bin']} | {cls} | {int(r['positive_labels'])} | "
            f"{float(r['positive_label_fraction']):.4f} |"
        )
    md.extend(
        [
            "",
            "## Frequency-bin Performance at Model-specific Best Thresholds",
            "",
            "| Model | Bin | Classes | Micro F1 | Precision | Recall | Support |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in freq_df.iterrows():
        md.append(
            f"| {r['display_model']} | {r['frequency_bin']} | {int(r['class_count'])} | "
            f"{r['micro_f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{int(r['support'])} |"
        )
    md.extend(
        [
            "",
            "## EC Level-1 Family Performance",
            "",
            "| Model | L1 | Family | n | Micro F1 | Precision | Recall | Mean sample F1 |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in family_df.iterrows():
        md.append(
            f"| {r['display_model']} | {r['l1']} | {r['l1_name']} | {int(r['n_samples'])} | "
            f"{r['micro_f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['mean_sample_f1']:.4f} |"
        )
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit separates fold-disjoint performance by label coverage, label frequency, "
            "and broad EC family. It should be used to explain whether the fold-disjoint "
            "drop is dominated by unseen/rare labels or is broadly present across enzyme families.",
        ]
    )
    (out_dir / f"{args.output_prefix}_label_difficulty_audit.md").write_text("\n".join(md) + "\n")
    print(out_dir / f"{args.output_prefix}_label_difficulty_audit.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_foldseek.yaml")
    parser.add_argument("--split", default="foldseek_tmscore50_cc_test")
    parser.add_argument("--split-prefix", default="foldseek_tmscore50_cc_")
    parser.add_argument("--cache-dir", default="outputs/cache/foldseek_threshold_topk")
    parser.add_argument("--best-threshold-csv", default="outputs/audit/foldseek_tmscore50_cc_best_threshold.csv")
    parser.add_argument("--out-dir", default="outputs/audit")
    parser.add_argument("--output-prefix", default="foldseek_tmscore50_cc")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
