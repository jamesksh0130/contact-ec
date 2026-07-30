#!/usr/bin/env python
"""Summarize label and cluster composition for Foldseek TM-score sweep splits."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


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


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def split_label_summary(cfg: dict, prefix: str, n_l4: int) -> dict:
    meta = pd.read_csv(ROOT / cfg["paths"]["meta_csv"]).set_index("accession")
    splits_dir = ROOT / cfg["paths"]["splits_dir"]
    train_ids = read_ids(splits_dir / f"{prefix}_train_ids.txt")
    test_ids = read_ids(splits_dir / f"{prefix}_test_ids.txt")

    train_counts = np.zeros(n_l4, dtype=np.int64)
    for acc in train_ids:
        if acc not in meta.index:
            continue
        for idx in parse_l4_indices(meta.at[acc, "l4_all_idxs"]):
            if 0 <= idx < n_l4:
                train_counts[idx] += 1

    test_positive = 0
    unseen_positive = 0
    rare_positive = 0
    sample_all_seen = 0
    sample_any_unseen = 0
    sample_all_unseen = 0
    valid_test = 0
    true_label_cardinality = []
    l1_counts: dict[str, int] = {}

    for acc in test_ids:
        if acc not in meta.index:
            continue
        row = meta.loc[acc]
        if int(row.get("m4", 0)) != 1:
            continue
        labels = [idx for idx in parse_l4_indices(row["l4_all_idxs"]) if 0 <= idx < n_l4]
        if not labels:
            continue
        valid_test += 1
        l1 = str(row.get("ec_l1", row.get("l1", row.get("l1_idx", ""))))
        l1_counts[l1] = l1_counts.get(l1, 0) + 1
        counts = train_counts[labels]
        test_positive += len(labels)
        unseen_positive += int((counts == 0).sum())
        rare_positive += int(((counts > 0) & (counts <= 25)).sum())
        sample_all_seen += int((counts > 0).all())
        sample_any_unseen += int((counts == 0).any())
        sample_all_unseen += int((counts == 0).all())
        true_label_cardinality.append(len(labels))

    audit_path = ROOT / "outputs" / "audit" / f"{prefix}_split_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    return {
        "prefix": prefix,
        "tm_score_threshold": audit.get("foldseek_tmscore_threshold"),
        "assignment_mode": audit.get("assignment_mode"),
        "clusters": audit.get("clusters"),
        "cluster_size_median": audit.get("cluster_size_median"),
        "cluster_size_mean": audit.get("cluster_size_mean"),
        "cluster_size_max": audit.get("cluster_size_max"),
        "train_proteins": len(train_ids),
        "test_proteins": len(test_ids),
        "valid_test_proteins": valid_test,
        "train_l4_seen_classes": int((train_counts > 0).sum()),
        "train_l4_unseen_classes": int((train_counts == 0).sum()),
        "test_positive_labels": test_positive,
        "unseen_positive_fraction": unseen_positive / test_positive if test_positive else 0.0,
        "rare_seen_positive_fraction": rare_positive / test_positive if test_positive else 0.0,
        "sample_all_labels_seen_fraction": sample_all_seen / valid_test if valid_test else 0.0,
        "sample_any_unseen_label_fraction": sample_any_unseen / valid_test if valid_test else 0.0,
        "sample_all_labels_unseen_fraction": sample_all_unseen / valid_test if valid_test else 0.0,
        "mean_true_label_cardinality": float(np.mean(true_label_cardinality)) if true_label_cardinality else 0.0,
        "largest_l1_fraction": max(l1_counts.values()) / valid_test if valid_test and l1_counts else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_foldseek.yaml")
    parser.add_argument(
        "--prefixes",
        nargs="+",
        default=["foldseek_tmscore40_cc", "foldseek_tmscore50_cc", "foldseek_tmscore60_cc"],
    )
    parser.add_argument("--out-dir", default="outputs/audit")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    with open(ROOT / cfg["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_l4 = len(encoders["level4"].classes_)

    rows = [split_label_summary(cfg, prefix, n_l4) for prefix in args.prefixes]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "foldseek_tmscore_sweep_split_composition.csv"
    md_path = out_dir / "foldseek_tmscore_sweep_split_composition.md"
    df.to_csv(csv_path, index=False)

    md = [
        "# Foldseek TM-score Sweep Split Composition",
        "",
        "| TM-score | Assignment | Clusters | Max cluster | Test proteins | Valid test | Seen L4 classes | Unseen positive labels | Any-unseen proteins | Rare seen positives |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        md.append(
            f"| {r['tm_score_threshold']:.2f} | {r['assignment_mode']} | "
            f"{int(r['clusters']):,} | {int(r['cluster_size_max']):,} | "
            f"{int(r['test_proteins']):,} | {int(r['valid_test_proteins']):,} | "
            f"{int(r['train_l4_seen_classes']):,} | "
            f"{r['unseen_positive_fraction']:.3f} | "
            f"{r['sample_any_unseen_label_fraction']:.3f} | "
            f"{r['rare_seen_positive_fraction']:.3f} |"
        )
    md.extend(
        [
            "",
            "Interpretation: the threshold sweep changes cluster granularity and, in the current generated splits, also changes the assignment mode for the TM-score 0.50 split. Therefore, the sweep should be interpreted as a robustness/composition audit rather than a strictly monotonic structural-stringency curve.",
        ]
    )
    md_path.write_text("\n".join(md) + "\n")
    print(csv_path)
    print(md_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
