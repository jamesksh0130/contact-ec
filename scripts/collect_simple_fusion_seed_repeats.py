#!/usr/bin/env python3
"""Collect temporal known-124 simple fusion baseline seed-repeat results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "outputs" / "results" / "simple_fusion_seed_repeats"
OUT_CSV = ROOT / "outputs" / "audit" / "simple_fusion_seed_repeats.csv"
OUT_MD = ROOT / "outputs" / "audit" / "simple_fusion_seed_repeats.md"

MODELS = [
    ("Concat flat FC", "fusion_concat_flatfc"),
    ("Sum flat FC", "fusion_sum_flatfc"),
    ("Gated MLP flat FC", "fusion_gated_mlp_flatfc"),
]
SEEDS = [42, 43, 44]


def load_row(label: str, model: str, seed: int) -> dict:
    path = RESULT_DIR / f"simple_{model}_seed{seed}_known124_hier_results.json"
    row = {
        "model_label": label,
        "model": model,
        "seed": seed,
        "status": "pending",
        "path": str(path.relative_to(ROOT)),
    }
    if not path.exists():
        return row

    data = json.loads(path.read_text())
    level4 = data.get("level4", {})
    rare = data.get("level4_underrepresented", {})
    row.update(
        {
            "status": "complete",
            "n_samples": level4.get("n_samples"),
            "micro_f1": level4.get("micro_f1"),
            "weighted_f1": level4.get("weighted_f1"),
            "macro_f1": level4.get("macro_f1"),
            "precision": level4.get("precision"),
            "recall": level4.get("recall"),
            "rare_ec_f1": rare.get("micro_f1"),
            "rare_ec_classes": rare.get("n_classes"),
        }
    )
    return row


def fmt_mean_sd(series: pd.Series) -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return "pending"
    if len(vals) == 1:
        return f"{vals.iloc[0]:.4f}"
    return f"{vals.mean():.4f} +/- {vals.std(ddof=1):.4f}"


def main() -> None:
    rows = [load_row(label, model, seed) for label, model in MODELS for seed in SEEDS]
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    summary_rows = []
    for label, group in df.groupby("model_label", sort=False):
        completed = group[group["status"] == "complete"]
        summary_rows.append(
            {
                "Model": label,
                "Completed seeds": f"{len(completed)}/{len(group)}",
                "Micro F1": fmt_mean_sd(completed.get("micro_f1", pd.Series(dtype=float))),
                "Weighted F1": fmt_mean_sd(completed.get("weighted_f1", pd.Series(dtype=float))),
                "Macro F1": fmt_mean_sd(completed.get("macro_f1", pd.Series(dtype=float))),
                "Precision": fmt_mean_sd(completed.get("precision", pd.Series(dtype=float))),
                "Recall": fmt_mean_sd(completed.get("recall", pd.Series(dtype=float))),
                "Rare EC F1": fmt_mean_sd(completed.get("rare_ec_f1", pd.Series(dtype=float))),
            }
        )
    summary = pd.DataFrame(summary_rows)

    lines = [
        "# Simple Fusion Baseline Seed Repeats",
        "",
        "Evaluation subset: complete Level-4 Swiss-Prot 2023-01 temporal proteins (N=124).",
        "Training corpus/config: SP-2018 EC-Bench setup (`config_ecbench.yaml`).",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Per-Seed Results",
        "",
        df.to_markdown(index=False),
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
