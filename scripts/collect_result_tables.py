#!/usr/bin/env python3
"""Collect existing metric JSON files into paper-ready CSV tables."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
OUT = ROOT / "outputs" / "audit"


def infer_from_name(path: Path) -> dict:
    name = path.stem
    split = "unknown"
    for key in ["cluster_test", "price149", "new392", "test", "val"]:
        if key in name:
            split = key
            break

    eval_mode = "hierarchical" if "_hier" in name else "independent"
    if "knn" in name:
        eval_mode = "knn"
    if name.startswith("bootstrap_ci"):
        eval_mode = "bootstrap_ci"
    if name.startswith("wilcoxon"):
        eval_mode = "wilcoxon"

    model = name
    suffixes = [
        "_cluster_test_hier_results", "_cluster_test_results",
        "_price149_hier_results", "_price149_results",
        "_new392_hier_results", "_new392_results",
        "_test_hier_results", "_test_results",
        "_val_hier_results", "_val_results",
        "_hier_results", "_results",
    ]
    for suffix in suffixes:
        if model.endswith(suffix):
            model = model[: -len(suffix)]
            break
    if name.startswith("bootstrap_ci_"):
        model = name.replace("bootstrap_ci_", "")
    if name.startswith("wilcoxon_"):
        model = name

    return {"model": model, "split": split, "eval_mode": eval_mode}


def get_nested(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def metric_row(path: Path, data: dict) -> dict | None:
    info = infer_from_name(path)
    root = data.get("metrics", data)

    row = {
        "file": str(path.relative_to(ROOT)),
        **info,
        "threshold": None,
        "l1_micro_f1": get_nested(root, "level1.micro_f1"),
        "l1_macro_f1": get_nested(root, "level1.macro_f1"),
        "l2_micro_f1": get_nested(root, "level2.micro_f1"),
        "l2_macro_f1": get_nested(root, "level2.macro_f1"),
        "l3_micro_f1": get_nested(root, "level3.micro_f1"),
        "l3_macro_f1": get_nested(root, "level3.macro_f1"),
        "l4_micro_f1": get_nested(root, "level4.micro_f1"),
        "l4_macro_f1": get_nested(root, "level4.macro_f1"),
        "l4_precision": get_nested(root, "level4.precision"),
        "l4_recall": get_nested(root, "level4.recall"),
        "rare_l4_micro_f1": get_nested(root, "level4_underrepresented.micro_f1"),
        "n_l4_samples": get_nested(root, "level4.n_samples"),
    }
    row["threshold"] = get_nested(root, "level4.threshold")

    if "point_f1" in data:
        row.update({
            "split": "test" if "test" in path.stem else info["split"],
            "l4_micro_f1": data.get("point_f1"),
            "l4_precision": data.get("point_precision"),
            "l4_recall": data.get("point_recall"),
            "threshold": data.get("threshold"),
            "ci_low": data.get("ci_low"),
            "ci_high": data.get("ci_high"),
            "ci_mean": data.get("mean_f1"),
            "ci_std": data.get("std_f1"),
            "n_l4_samples": data.get("n_samples"),
        })

    if "V2_micro_f1" in data:
        row.update({
            "model": "fusion_v2_vs_b2_wilcoxon",
            "split": "test",
            "eval_mode": "wilcoxon",
            "l4_micro_f1": data.get("V2_micro_f1"),
            "b2_l4_micro_f1": data.get("B2_micro_f1"),
            "p_value": data.get("p_value"),
            "n_l4_samples": data.get("n_samples"),
        })

    if row["l4_micro_f1"] is None and "results_by_K" in data:
        rows = []
        for k, vals in data["results_by_K"].items():
            r = dict(row)
            r["eval_mode"] = f"knn_{k}"
            r["l4_micro_f1"] = vals.get("micro_f1")
            r["l4_precision"] = vals.get("precision")
            r["l4_recall"] = vals.get("recall")
            rows.append(r)
        return rows

    return row


def add_claim_safety(df: pd.DataFrame) -> pd.DataFrame:
    notes = []
    for _, r in df.iterrows():
        note = []
        split = str(r.get("split", ""))
        model = str(r.get("model", ""))
        if split == "new392":
            note.append("warning: overlaps current 2026 Swiss-Prot train; not clean external unless cutoff-controlled")
        if split == "cluster_test":
            note.append("valid OOD only if checkpoint trained on cluster_train")
        if split == "price149":
            note.append("no contact maps in current pipeline; structure path unavailable")
        if split == "test":
            note.append("random split has exact sequence duplicates; treat as in-distribution")
        if "hitec" in model:
            note.append("prior/pretrained comparison has limited class/domain comparability")
        notes.append(" | ".join(note))
    df["claim_safety_note"] = notes
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        row = metric_row(path, data)
        if row is None:
            continue
        if isinstance(row, list):
            rows.extend(row)
        else:
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No metric JSON files found")

    df = add_claim_safety(df)
    sort_cols = [c for c in ["split", "model", "eval_mode"] if c in df.columns]
    df = df.sort_values(sort_cols)
    df.to_csv(OUT / "all_result_metrics.csv", index=False)

    main_cols = [
        "model", "split", "eval_mode", "threshold", "l4_micro_f1",
        "l4_macro_f1", "l4_precision", "l4_recall", "rare_l4_micro_f1",
        "ci_low", "ci_high", "p_value", "claim_safety_note", "file",
    ]
    for c in main_cols:
        if c not in df:
            df[c] = None
    compact = df[main_cols].copy()
    compact = compact[compact["l4_micro_f1"].notna()]
    compact.to_csv(OUT / "paper_metric_candidates.csv", index=False)

    print(f"Wrote {OUT / 'all_result_metrics.csv'}")
    print(f"Wrote {OUT / 'paper_metric_candidates.csv'}")
    print("\nTop L4 micro F1 rows:")
    show = compact.sort_values("l4_micro_f1", ascending=False).head(20)
    print(show[["model", "split", "eval_mode", "l4_micro_f1", "claim_safety_note"]].to_string(index=False))


if __name__ == "__main__":
    main()
