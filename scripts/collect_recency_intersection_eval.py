#!/usr/bin/env python3
"""Collect fair recency-intersection temporal evaluation results.

The collector supports the original single ExpA run and, when available,
seed-specific ExpA repeats named with the recency_intersection_expa_seed*
split convention.
"""

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


def append_row(rows: list[dict[str, object]], model: str, seed: int | str, metrics: dict[str, float]) -> None:
    rows.append(
        {
            "model": model,
            "seed": seed,
            "n": metrics["n_samples"],
            "micro_f1": metrics["micro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "rare_micro_f1": metrics["rare_micro_f1"],
        }
    )


def load_expa_seed(seed: int) -> dict[str, float] | None:
    seed_path = RESULT_DIR / f"fusion_v2_flatfc_recency_intersection_expa_seed{seed}_results.json"
    if seed_path.exists():
        return load_l4(seed_path)
    if seed == 42:
        single_path = RESULT_DIR / "fusion_v2_flatfc_recency_intersection_expa_results.json"
        if single_path.exists():
            return load_l4(single_path)
    return None


def main() -> None:
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for seed in [42, 43, 44]:
        metrics = load_l4(RESULT_DIR / f"fusion_v2_flatfc_recency_intersection_seed{seed}_results.json")
        append_row(rows, "Contact-EC 2018", seed, metrics)

    for seed in [42, 43, 44]:
        metrics = load_expa_seed(seed)
        if metrics is not None:
            append_row(rows, "Contact-EC ExpA 2026", seed, metrics)

    df = pd.DataFrame(rows)
    csv_path = OUT_PREFIX.with_suffix(".csv")
    json_path = OUT_PREFIX.with_suffix(".json")
    md_path = OUT_PREFIX.with_suffix(".md")
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    seed_df = df[df["model"] == "Contact-EC 2018"]
    summary = seed_df[["micro_f1", "weighted_f1", "precision", "recall", "rare_micro_f1"]].agg(["mean", "std"])
    expa_df = df[df["model"] == "Contact-EC ExpA 2026"]
    expa_summary = expa_df[["micro_f1", "weighted_f1", "precision", "recall", "rare_micro_f1"]].agg(["mean", "std"])
    expa_n = int(expa_df["n"].iloc[0]) if len(expa_df) else 0
    delta = float(expa_summary.loc["mean", "micro_f1"]) - float(summary.loc["mean", "micro_f1"])
    expa_std = expa_summary.loc["std", "micro_f1"]
    expa_std_text = "n/a" if pd.isna(expa_std) else f"{float(expa_std):.4f}"

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
        f"- Contact-EC ExpA 2026 micro F1: {float(expa_summary.loc['mean', 'micro_f1']):.4f} +/- {expa_std_text} ({len(expa_df)} run(s), N={expa_n}).",
        f"- Fair-subset recent-corpus delta: {delta:+.4f} micro F1.",
        "",
        "Interpretation:",
        "- The original single-seed ExpA value of 0.7209 was a 99-sample encoder-evaluable result, not a 124-sample known-set result.",
        "- The three-seed ExpA mean remains positive on the fair subset and is smaller than the naive comparison against the 124-sample 2018 result.",
        "- ExpA should be described as a recent-corpus/vocabulary diagnostic because sample count, class frequencies, label coverage, and homolog coverage also change.",
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
