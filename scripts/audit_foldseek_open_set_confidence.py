#!/usr/bin/env python
"""Open-vocabulary confidence audit for the Foldseek-disjoint split.

This CPU-only analysis asks whether model confidence can identify proteins whose
true Level-4 labels are absent from the Foldseek training label vocabulary.  It
does not turn Contact-EC into an open-set predictor; it quantifies whether
abstention or fallback-to-parent routing is plausible.
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

MODEL_FILES = {
    "B1_ESM2_only": [
        "B1_ESM2_only_seed42_base_foldseek_tmscore50_cc_test_l4_probs_labels.npz",
        "B1_ESM2_only_seed43_foldseek_tmscore50_cc_test_l4_probs_labels.npz",
        "B1_ESM2_only_seed44_foldseek_tmscore50_cc_test_l4_probs_labels.npz",
    ],
    "B3_contact_only": [
        "B3_contact_only_seed42_base_foldseek_tmscore50_cc_test_l4_probs_labels.npz",
        "B3_contact_only_seed43_foldseek_tmscore50_cc_test_l4_probs_labels.npz",
        "B3_contact_only_seed44_foldseek_tmscore50_cc_test_l4_probs_labels.npz",
    ],
    "Fusion_ESM2_contact": [
        "Fusion_ESM2_contact_seed42_base_foldseek_tmscore50_cc_test_l4_probs_labels.npz",
        "Fusion_ESM2_contact_seed43_foldseek_tmscore50_cc_test_l4_probs_labels.npz",
        "Fusion_ESM2_contact_seed44_foldseek_tmscore50_cc_test_l4_probs_labels.npz",
    ],
}


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


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            avg = (start + 1 + end) / 2.0
            ranks[order[start:end]] = avg
        start = end
    rank_sum_pos = ranks[y_true].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    hits = y_true[order].astype(np.float64)
    precision_at_k = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((precision_at_k * hits).sum() / n_pos)


def load_metadata(config: str, split_prefix: str, split: str, n_l4: int) -> pd.DataFrame:
    with open(ROOT / "configs" / config) as f:
        cfg = yaml.safe_load(f)
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
        true_idxs = [idx for idx in parse_l4_indices(row["l4_all_idxs"]) if 0 <= idx < n_l4]
        if not true_idxs:
            continue
        counts = train_counts[true_idxs]
        rows.append(
            {
                "accession": acc,
                "true_label_count": len(true_idxs),
                "any_unseen_true_label": bool((counts == 0).any()),
                "all_true_labels_unseen": bool((counts == 0).all()),
                "min_true_label_train_count": int(counts.min()) if len(counts) else 0,
            }
        )
    return pd.DataFrame(rows)


def confidence_features(probs: np.ndarray) -> dict[str, np.ndarray]:
    eps = 1e-8
    top1 = probs.max(axis=1)
    top5_sum = np.sort(probs, axis=1)[:, -5:].sum(axis=1)
    pred_count_005 = (probs >= 0.05).sum(axis=1)
    pred_count_010 = (probs >= 0.10).sum(axis=1)
    prob_sum = probs.sum(axis=1)
    entropy = -(probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps)).sum(axis=1)
    return {
        "low_top1": -top1,
        "low_top5_sum": -top5_sum,
        "low_prob_sum": -prob_sum,
        "high_entropy": entropy,
        "low_pred_count_005": -pred_count_005.astype(np.float64),
        "low_pred_count_010": -pred_count_010.astype(np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_foldseek.yaml")
    parser.add_argument("--split", default="foldseek_tmscore50_cc_test")
    parser.add_argument("--split-prefix", default="foldseek_tmscore50_cc_")
    parser.add_argument("--cache-dir", default="outputs/cache/foldseek_seed_repeats")
    parser.add_argument("--output-prefix", default="foldseek_tmscore50_cc_open_set_confidence")
    args = parser.parse_args()

    cache_dir = ROOT / args.cache_dir
    out_dir = ROOT / "outputs" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    first = np.load(cache_dir / MODEL_FILES["Fusion_ESM2_contact"][0])
    n_l4 = first["labels"].shape[1]
    meta = load_metadata(args.config, args.split_prefix, args.split, n_l4)
    y_unseen = meta["any_unseen_true_label"].to_numpy(dtype=bool)

    rows = []
    per_sample = meta.copy()
    for model, files in MODEL_FILES.items():
        prob_stack = [np.load(cache_dir / fname)["probs"].astype(np.float32) for fname in files]
        probs = np.mean(prob_stack, axis=0)
        if probs.shape[0] != len(meta):
            raise RuntimeError(f"{model}: probability rows {probs.shape[0]} != metadata rows {len(meta)}")
        feats = confidence_features(probs)
        for name, score in feats.items():
            rows.append(
                {
                    "model": model,
                    "score": name,
                    "target": "any_unseen_true_label",
                    "prevalence": float(y_unseen.mean()),
                    "auroc": auroc(y_unseen, score),
                    "average_precision": average_precision(y_unseen, score),
                }
            )
        per_sample[f"{model}_top1_prob"] = probs.max(axis=1)
        per_sample[f"{model}_prob_sum"] = probs.sum(axis=1)
        per_sample[f"{model}_entropy"] = feats["high_entropy"]

    result = pd.DataFrame(rows).sort_values(["model", "auroc"], ascending=[True, False])
    result.to_csv(out_dir / f"{args.output_prefix}.csv", index=False)
    per_sample.to_csv(out_dir / f"{args.output_prefix}_per_sample.csv", index=False)

    best = result.sort_values("auroc", ascending=False).groupby("model").head(1)
    payload = {
        "n_samples": int(len(meta)),
        "unseen_prevalence": float(y_unseen.mean()),
        "best_by_model": best.to_dict(orient="records"),
    }
    (out_dir / f"{args.output_prefix}.json").write_text(json.dumps(payload, indent=2))

    md = [
        "# Foldseek Open-Vocabulary Confidence Audit",
        "",
        f"- Samples: {len(meta):,}",
        f"- Proteins with any unseen true Level-4 label: {y_unseen.mean():.3f}",
        "",
        "## Best confidence score by model",
        "",
        best.to_markdown(index=False),
        "",
        "## All confidence scores",
        "",
        result.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- AUROC close to 0.5 means model confidence is weak for detecting unseen-label cases.",
        "- Strong AUROC/AP would support an abstention or hierarchical fallback mechanism.",
        "- This is a diagnostic audit, not a supervised open-set classifier.",
    ]
    (out_dir / f"{args.output_prefix}.md").write_text("\n".join(md) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
