#!/usr/bin/env python3
"""Collect fixed-vocabulary recency-control evaluations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "audit"
RESULT_ROOT = ROOT / "outputs" / "results"
FIXED_DIR = RESULT_ROOT / "recency_fixed_vocab_seed_repeats"


def load_l4(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return data["level4"] | {
        "rare_micro_f1": data.get("level4_underrepresented", {}).get("micro_f1")
    }


def add_row(rows: list[dict[str, object]], model: str, subset: str, seed: int, path: Path) -> None:
    metrics = load_l4(path)
    if metrics is None:
        return
    rows.append(
        {
            "model": model,
            "subset": subset,
            "seed": seed,
            "n": metrics["n_samples"],
            "micro_f1": metrics["micro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "rare_micro_f1": metrics["rare_micro_f1"],
            "path": str(path.relative_to(ROOT)),
        }
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metrics = ["micro_f1", "weighted_f1", "precision", "recall", "rare_micro_f1"]
    rows = []
    for (model, subset), group in df.groupby(["model", "subset"], sort=False):
        row = {
            "model": model,
            "subset": subset,
            "runs": len(group),
            "n": int(group["n"].iloc[0]),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in [42, 43, 44]:
        add_row(
            rows,
            "Contact-EC SP-2018 old vocab",
            "known124",
            seed,
            RESULT_ROOT / "temporal_known_seed_repeats" / f"temporal_known_fusion_v2_flatfc_seed{seed}_known124_hier_results.json",
        )
        add_row(
            rows,
            "Contact-EC SP-2018 old vocab",
            "intersection99",
            seed,
            RESULT_ROOT / f"fusion_v2_flatfc_recency_intersection_seed{seed}_results.json",
        )
        add_row(
            rows,
            "Contact-EC SP-2026 fixed old vocab",
            "known124",
            seed,
            FIXED_DIR / f"seed{seed}_known124_results.json",
        )
        add_row(
            rows,
            "Contact-EC SP-2026 fixed old vocab",
            "intersection99",
            seed,
            FIXED_DIR / f"seed{seed}_intersection99_results.json",
        )
        add_row(
            rows,
            "Contact-EC SP-2026 ExpA full vocab",
            "intersection99",
            seed,
            RESULT_ROOT / f"fusion_v2_flatfc_recency_intersection_expa_seed{seed}_results.json",
        )

    df = pd.DataFrame(rows)
    summary = summarize(df)

    csv_path = OUT_DIR / "recency_fixed_vocab_eval.csv"
    summary_path = OUT_DIR / "recency_fixed_vocab_eval_summary.csv"
    json_path = OUT_DIR / "recency_fixed_vocab_eval_summary.json"
    md_path = OUT_DIR / "recency_fixed_vocab_eval.md"
    df.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(summary.to_dict(orient="records"), indent=2))

    md = [
        "# Fixed-Vocabulary Recency-Control Evaluation",
        "",
        "This table compares SP-2018 training, SP-2026 training under the fixed",
        "SP-2018 EC-Bench vocabulary, and full ExpA/SP-2026 vocabulary when available.",
        "",
        "## Per-run Results",
        "",
        df.to_markdown(index=False) if not df.empty else "No results collected yet.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f") if not summary.empty else "No summary available yet.",
        "",
        "Interpretation:",
        "- SP-2026 fixed-old-vocabulary vs SP-2018 estimates newer-sample/homolog-coverage effects without vocabulary expansion.",
        "- Full ExpA vs SP-2026 fixed-old-vocabulary estimates the remaining effect of encoder/vocabulary and class-space changes on the matched intersection.",
    ]
    md_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
