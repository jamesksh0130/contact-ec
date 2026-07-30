#!/usr/bin/env python3
"""Collect fair recency-intersection temporal evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs/results"
OUT_PREFIX = ROOT / "outputs/audit/recency_intersection_eval"


def load_l4(path: Path) -> dict[str, float]:
    with path.open() as f:
        data = json.load(f)
    return data["level4"] | {
        "rare_micro_f1": data.get("level4_underrepresented", {}).get("micro_f1")
    }


def main() -> None:
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for seed in [42, 43, 44]:
        metrics = load_l4(RESULT_DIR / f"fusion_v2_flatfc_recency_intersection_seed{seed}_results.json")
        rows.append(
            {
                "model": "Contact-EC 2018",
                "seed": seed,
                "n": metrics["n_samples"],
                "micro_f1": metrics["micro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "rare_micro_f1": metrics["rare_micro_f1"],
            }
        )

    metrics = load_l4(RESULT_DIR / "fusion_v2_flatfc_recency_intersection_expa_results.json")
    rows.append(
        {
            "model": "Contact-EC ExpA 2026",
            "seed": "single",
            "n": metrics["n_samples"],
            "micro_f1": metrics["micro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "rare_micro_f1": metrics["rare_micro_f1"],
        }
    )

    df = pd.DataFrame(rows)
    csv_path = OUT_PREFIX.with_suffix(".csv")
    json_path = OUT_PREFIX.with_suffix(".json")
    md_path = OUT_PREFIX.with_suffix(".md")
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    seed_df = df[df["model"] == "Contact-EC 2018"]
    summary = seed_df[["micro_f1", "weighted_f1", "precision", "recall", "rare_micro_f1"]].agg(["mean", "std"])
    expa = df[df["model"] == "Contact-EC ExpA 2026"].iloc[0]
    delta = float(expa["micro_f1"]) - float(summary.loc["mean", "micro_f1"])

    md = [
        "# Recency Intersection Evaluation",
        "",
        "All models are evaluated on the same temporal known subset that is Level-4 evaluable under the 2018, 2022, and ExpA/2026 encoders (N=99).",
        "This avoids comparing the ExpA result against the broader 2018-only N=124 denominator.",
        "",
        "## Per-run Results",
        "",
        df.to_markdown(index=False),
        "",
        "## Summary",
        "",
        f"- Contact-EC 2018 micro F1: {summary.loc['mean', 'micro_f1']:.4f} +/- {summary.loc['std', 'micro_f1']:.4f} (3 seeds).",
        f"- Contact-EC ExpA 2026 micro F1: {float(expa['micro_f1']):.4f} (single run).",
        f"- Fair-subset recent-corpus delta: {delta:+.4f} micro F1.",
        "",
        "Interpretation:",
        "- The previously reported ExpA value of 0.7209 is a 99-sample encoder-evaluable result, not a 124-sample known-set result.",
        "- The fair-subset gain remains positive but is smaller than the naive comparison against the 124-sample 2018 result.",
        "- ExpA should be described as a recent-corpus/vocabulary diagnostic unless repeated with multiple seeds and matched controls.",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(md[-6])
    print(md[-5])
    print(md[-4])


if __name__ == "__main__":
    main()
