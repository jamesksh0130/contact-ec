#!/usr/bin/env python
"""Collect TM-score strictness sweep results into paper-ready tables."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
AUDIT_DIR = ROOT / "outputs" / "audit"

MODELS = [
    ("b1_esm2_fc", "B1 ESM-2 only"),
    ("b3_contact", "B3 contact only"),
    ("fusion", "Contact-EC fusion"),
]
THRESHOLDS = [40, 50, 60]


def read_result(model: str, split: str) -> dict | None:
    path = RESULT_DIR / f"{model}_{split}_hier_results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_split(prefix: str) -> dict:
    path = AUDIT_DIR / f"{prefix}_split_audit.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    rows = []
    for thr in THRESHOLDS:
        prefix = f"foldseek_tmscore{thr}_cc"
        split = f"{prefix}_test"
        audit = read_split(prefix)
        for model_key, model_name in MODELS:
            result = read_result(model_key, split)
            row = {
                "tm_score_threshold": thr / 100.0,
                "split": split,
                "model": model_name,
                "test_proteins": audit.get("test_proteins"),
                "clusters": audit.get("clusters"),
                "cluster_size_median": audit.get("cluster_size_median"),
                "cluster_size_max": audit.get("cluster_size_max"),
                "l1_accuracy": None,
                "l2_accuracy": None,
                "l3_accuracy": None,
                "l4_micro_f1": None,
                "l4_precision": None,
                "l4_recall": None,
            }
            if result:
                row.update(
                    {
                        "l1_accuracy": result.get("level1", {}).get("accuracy"),
                        "l2_accuracy": result.get("level2", {}).get("accuracy"),
                        "l3_accuracy": result.get("level3", {}).get("accuracy"),
                        "l4_micro_f1": result.get("level4", {}).get("micro_f1"),
                        "l4_precision": result.get("level4", {}).get("precision"),
                        "l4_recall": result.get("level4", {}).get("recall"),
                    }
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = AUDIT_DIR / "foldseek_tmscore_sweep_summary.csv"
    out_md = AUDIT_DIR / "foldseek_tmscore_sweep_summary.md"
    df.to_csv(out_csv, index=False)

    complete = df[df["l4_micro_f1"].notna()].copy()
    md = [
        "# Foldseek TM-score Strictness Sweep",
        "",
        "## Available results",
        "",
        complete.to_markdown(index=False) if not complete.empty else "No completed model results yet.",
        "",
        "## Expected rows, including pending runs",
        "",
        df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- TM-score 0.40 is the looser structural clustering threshold; 0.60 is the stricter threshold.",
        "- The key question is whether Contact-EC fusion retains a relative advantage as structural disjointness becomes stricter.",
        "- Missing metric cells mean the corresponding long training/evaluation run has not finished yet.",
    ]
    out_md.write_text("\n".join(md) + "\n")
    print(df.to_string(index=False))
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
