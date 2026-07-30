#!/usr/bin/env python
"""Abstention and parent-level fallback audit for Foldseek-disjoint predictions."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from audit_foldseek_open_set_confidence import MODEL_FILES, load_metadata

ROOT = Path(__file__).resolve().parents[1]


def micro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    labels = labels.astype(bool, copy=False)
    preds = preds.astype(bool, copy=False)
    tp = float(np.logical_and(labels, preds).sum())
    fp = float(np.logical_and(~labels, preds).sum())
    fn = float(np.logical_and(labels, ~preds).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def sample_hit(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    return np.logical_and(labels.astype(bool), preds.astype(bool)).any(axis=1)


def build_l4_to_l3() -> tuple[np.ndarray, list[str]]:
    with open(ROOT / "data" / "label_encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    l4_classes = list(enc["level4"].classes_)
    l3_classes = list(enc["level3"].classes_)
    l3_to_idx = {label: i for i, label in enumerate(l3_classes)}
    l4_to_l3 = np.full(len(l4_classes), -1, dtype=np.int64)
    for i, label in enumerate(l4_classes):
        parent = ".".join(str(label).split(".")[:3])
        if parent in l3_to_idx:
            l4_to_l3[i] = l3_to_idx[parent]
    return l4_to_l3, l3_classes


def aggregate_l4_to_l3(probs: np.ndarray, l4_to_l3: np.ndarray, n_l3: int) -> np.ndarray:
    out = np.zeros((probs.shape[0], n_l3), dtype=np.float32)
    valid = np.where(l4_to_l3 >= 0)[0]
    # max aggregation is conservative: one confident child label supports its parent.
    np.maximum.at(out, (slice(None), l4_to_l3[valid]), probs[:, valid])
    return out


def true_l3_matrix(labels_l4: np.ndarray, l4_to_l3: np.ndarray, n_l3: int) -> np.ndarray:
    out = np.zeros((labels_l4.shape[0], n_l3), dtype=bool)
    rows, cols = np.where(labels_l4 > 0)
    parents = l4_to_l3[cols]
    ok = parents >= 0
    out[rows[ok], parents[ok]] = True
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_foldseek.yaml")
    parser.add_argument("--split", default="foldseek_tmscore50_cc_test")
    parser.add_argument("--split-prefix", default="foldseek_tmscore50_cc_")
    parser.add_argument("--cache-dir", default="outputs/cache/foldseek_seed_repeats")
    parser.add_argument("--model", default="Fusion_ESM2_contact", choices=sorted(MODEL_FILES))
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--output-prefix", default="foldseek_tmscore50_cc_fusion_abstention_fallback")
    args = parser.parse_args()

    cache_dir = ROOT / args.cache_dir
    out_dir = ROOT / "outputs" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    prob_stack = []
    labels = None
    for fname in MODEL_FILES[args.model]:
        z = np.load(cache_dir / fname)
        prob_stack.append(z["probs"].astype(np.float32))
        if labels is None:
            labels = z["labels"].astype(np.int32)
    probs = np.mean(prob_stack, axis=0)
    assert labels is not None

    meta = load_metadata(args.config, args.split_prefix, args.split, labels.shape[1])
    if len(meta) != probs.shape[0]:
        raise RuntimeError(f"metadata rows {len(meta)} != probability rows {probs.shape[0]}")

    l4_to_l3, l3_classes = build_l4_to_l3()
    l3_probs = aggregate_l4_to_l3(probs, l4_to_l3, len(l3_classes))
    l3_labels = true_l3_matrix(labels, l4_to_l3, len(l3_classes))

    top5_sum = np.sort(probs, axis=1)[:, -5:].sum(axis=1)
    order_low_conf = np.argsort(top5_sum)
    preds_l4 = (probs >= args.threshold).astype(np.int32)
    l4_hit = sample_hit(labels, preds_l4)
    l3_top1 = np.argmax(l3_probs, axis=1)
    l3_top1_hit = l3_labels[np.arange(len(l3_labels)), l3_top1]
    unseen = meta["any_unseen_true_label"].to_numpy(dtype=bool)

    rows = []
    for abstain_frac in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        n_abs = int(round(len(probs) * abstain_frac))
        abstain = np.zeros(len(probs), dtype=bool)
        if n_abs:
            abstain[order_low_conf[:n_abs]] = True
        retain = ~abstain
        retained_micro = micro_f1(labels[retain], preds_l4[retain]) if retain.any() else float("nan")
        retained_hit = float(l4_hit[retain].mean()) if retain.any() else float("nan")
        abstained_unseen = float(unseen[abstain].mean()) if abstain.any() else float("nan")
        abstained_l3_hit = float(l3_top1_hit[abstain].mean()) if abstain.any() else float("nan")
        hierarchical_success = np.where(abstain, l3_top1_hit, l4_hit)
        rows.append(
            {
                "model": args.model,
                "abstain_low_conf_fraction": abstain_frac,
                "coverage": float(retain.mean()),
                "retained_l4_micro_f1": retained_micro,
                "retained_l4_sample_hit": retained_hit,
                "abstained_unseen_prevalence": abstained_unseen,
                "abstained_l3_top1_hit": abstained_l3_hit,
                "overall_l4_or_fallback_l3_hit": float(hierarchical_success.mean()),
                "mean_top5_sum_retained": float(top5_sum[retain].mean()) if retain.any() else float("nan"),
                "mean_top5_sum_abstained": float(top5_sum[abstain].mean()) if abstain.any() else float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{args.output_prefix}.csv"
    json_path = out_dir / f"{args.output_prefix}.json"
    md_path = out_dir / f"{args.output_prefix}.md"
    df.to_csv(csv_path, index=False)
    payload = {
        "model": args.model,
        "threshold": args.threshold,
        "n_samples": int(len(probs)),
        "baseline": df.iloc[0].to_dict(),
        "best_overall_fallback_hit": df.loc[df["overall_l4_or_fallback_l3_hit"].idxmax()].to_dict(),
        "best_retained_l4_micro_f1": df.loc[df["retained_l4_micro_f1"].idxmax()].to_dict(),
    }
    json_path.write_text(json.dumps(payload, indent=2))
    md = [
        "# Foldseek Abstention and Parent-Level Fallback Audit",
        "",
        f"- Model: `{args.model}`",
        f"- Level-4 threshold: {args.threshold}",
        f"- Samples: {len(probs):,}",
        "",
        df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Abstention removes the lowest-confidence proteins according to the top-5 Level-4 probability sum.",
        "- Retained Level-4 micro F1 estimates performance on accepted closed-set predictions.",
        "- Abstained Level-3 top-1 hit estimates whether a conservative parent-level fallback remains useful.",
    ]
    md_path.write_text("\n".join(md) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
