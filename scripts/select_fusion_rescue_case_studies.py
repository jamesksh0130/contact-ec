#!/usr/bin/env python
"""Select representative Fusion-only rescue cases for biological interpretation."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def first_ec(value: str) -> str:
    text = str(value).strip()
    if not text or text == "nan":
        return ""
    return text.split(";")[0].strip()


def ec_parent(ec: str, level: int) -> str:
    parts = str(ec).split(".")
    return ".".join(parts[:level]) if len(parts) >= level else str(ec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-csv", default="outputs/audit/foldseek_tmscore50_cc_failure_modes_per_sample.csv")
    parser.add_argument("--meta-csv", default="data/ecbench/processed/train_meta.csv")
    parser.add_argument("--n-per-family", type=int, default=4)
    parser.add_argument("--output-prefix", default="foldseek_tmscore50_cc_fusion_rescue_case_studies")
    args = parser.parse_args()

    out_dir = ROOT / "outputs" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    failure = pd.read_csv(ROOT / args.failure_csv)
    meta = pd.read_csv(ROOT / args.meta_csv)

    meta_small = meta[["accession", "sequence", "ec_raw", "l1_idx", "l2_idx", "l3_idx", "l4_idx"]].copy()
    merged = failure.merge(meta_small, on="accession", how="left", suffixes=("", "_meta"))
    merged["true_l4"] = merged["ec_raw"].apply(first_ec)
    merged["true_l3"] = merged["true_l4"].apply(lambda ec: ec_parent(ec, 3))
    merged["true_l2"] = merged["true_l4"].apply(lambda ec: ec_parent(ec, 2))
    merged["true_l1"] = merged["true_l4"].apply(lambda ec: ec_parent(ec, 1))

    candidates = merged[
        (merged["fusion_only_rescue"])
        & (~merged["any_unseen_true_label"])
        & (merged["Fusion_ESM2_contact_hit_rate"] > 0)
    ].copy()
    candidates["priority"] = 0
    candidates.loc[candidates["difficulty_bin"].isin(["6-25", "26-100"]), "priority"] += 3
    candidates.loc[candidates["difficulty_bin"].eq("1-5"), "priority"] += 2
    candidates.loc[candidates["Fusion_ESM2_contact_hit_rate"] >= 2 / 3, "priority"] += 2
    candidates.loc[candidates["Fusion_ESM2_contact_top1_hit_rate"] > 0, "priority"] += 1
    candidates = candidates.sort_values(
        ["priority", "Fusion_ESM2_contact_hit_rate", "Fusion_ESM2_contact_top1_hit_rate", "min_true_label_train_count"],
        ascending=[False, False, False, True],
    )

    selected = (
        candidates.groupby("l1_name", group_keys=False)
        .head(args.n_per_family)
        .sort_values(["priority", "l1_name", "Fusion_ESM2_contact_hit_rate"], ascending=[False, True, False])
    )

    cols = [
        "accession",
        "l1_name",
        "true_l4",
        "true_l3",
        "ec_raw",
        "sequence_length",
        "difficulty_bin",
        "min_true_label_train_count",
        "Fusion_ESM2_contact_hit_rate",
        "Fusion_ESM2_contact_top1_hit_rate",
        "Fusion_ESM2_contact_sample_f1_mean",
        "B1_ESM2_only_hit_rate",
        "B3_contact_only_hit_rate",
        "priority",
    ]
    selected[cols].to_csv(out_dir / f"{args.output_prefix}.csv", index=False)

    fasta_lines = []
    for row in selected.itertuples(index=False):
        fasta_lines.append(f">{row.accession}|{row.l1_name}|EC={row.true_l4}|bin={row.difficulty_bin}")
        seq = str(getattr(row, "sequence", ""))
        for i in range(0, len(seq), 80):
            fasta_lines.append(seq[i : i + 80])
    (out_dir / f"{args.output_prefix}.fasta").write_text("\n".join(fasta_lines) + "\n")

    family_summary = (
        candidates.groupby(["l1_name", "difficulty_bin"], observed=False)
        .agg(
            n_candidates=("accession", "count"),
            mean_fusion_hit=("Fusion_ESM2_contact_hit_rate", "mean"),
            median_train_count=("min_true_label_train_count", "median"),
        )
        .reset_index()
        .sort_values(["l1_name", "difficulty_bin"])
    )
    family_summary.to_csv(out_dir / f"{args.output_prefix}_family_summary.csv", index=False)

    md = [
        "# Fusion-Only Rescue Case Study Candidates",
        "",
        "Selection criteria: fusion-only rescue, no unseen true Level-4 label, and nonzero fusion hit rate across seeds.",
        "",
        "## Selected cases",
        "",
        selected[cols].head(30).to_markdown(index=False),
        "",
        "## Candidate distribution by EC family and training-frequency bin",
        "",
        family_summary.to_markdown(index=False),
        "",
        "## Recommended use",
        "",
        "- Pick 3-5 cases from different EC families.",
        "- Prefer seen-but-rare or moderately represented labels because they support structural generalisation rather than vocabulary expansion.",
        "- Link each selected accession to contact-map visualisation or Grad-CAM when available.",
    ]
    (out_dir / f"{args.output_prefix}.md").write_text("\n".join(md) + "\n")
    print(selected[cols].head(20).to_string(index=False))
    print(f"Wrote {out_dir / f'{args.output_prefix}.md'}")


if __name__ == "__main__":
    main()
