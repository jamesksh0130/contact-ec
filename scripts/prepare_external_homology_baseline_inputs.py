#!/usr/bin/env python
"""Prepare external temporal/Price FASTA inputs for MMseqs2 EC-transfer baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def ec_column(df: pd.DataFrame) -> str:
    for col in ("ec_raw", "ec_chosen", "EC number", "ec_number"):
        if col in df.columns:
            return col
    raise KeyError(f"No EC column found in {df.columns.tolist()}")


def accession_column(df: pd.DataFrame) -> str:
    for col in ("accession", "Entry", "id"):
        if col in df.columns:
            return col
    raise KeyError(f"No accession column found in {df.columns.tolist()}")


def sequence_column(df: pd.DataFrame) -> str:
    for col in ("sequence", "Sequence", "seq"):
        if col in df.columns:
            return col
    raise KeyError(f"No sequence column found in {df.columns.tolist()}")


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    acc_col = accession_column(df)
    seq_col = sequence_column(df)
    ec_col = ec_column(df)
    lines = []
    for row in df.itertuples(index=False):
        acc = str(getattr(row, acc_col.replace(" ", "_")))
        seq = str(getattr(row, seq_col.replace(" ", "_")))
        ec = str(getattr(row, ec_col.replace(" ", "_")))
        if not acc or not seq or seq == "nan" or not ec or ec == "nan":
            continue
        lines.append(f">{acc}|EC={ec}")
        for i in range(0, len(seq), 80):
            lines.append(seq[i : i + 80])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/baselines/homology")
    args = parser.parse_args()

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(ROOT / "data/ecbench/processed/train_meta.csv")
    temporal_known = pd.read_csv(ROOT / "data/ecbench/processed/test_meta_full.csv")
    temporal_current = pd.read_csv(ROOT / "data/ecbench/processed/test_meta.csv")
    price_ecbench = pd.read_csv(ROOT / "data/ecbench/processed/price149_meta.csv")
    price_raw = pd.read_csv(ROOT / "data/ecbench/raw/price149.csv")

    jobs = [
        ("sp2018_temporal_known124", train, temporal_known),
        ("sp2018_temporal_current101", train, temporal_current),
        ("sp2018_price149_encoded136", train, price_ecbench),
        ("sp2018_price149_raw149", train, price_raw),
    ]

    manifest = []
    for prefix, train_df, test_df in jobs:
        train_path = out / f"{prefix}_train.fasta"
        test_path = out / f"{prefix}_test.fasta"
        write_fasta(train_df, train_path)
        write_fasta(test_df, test_path)
        manifest.append(
            {
                "split": prefix,
                "train_n": len(train_df),
                "test_n": len(test_df),
                "train_fasta": str(train_path.relative_to(ROOT)),
                "test_fasta": str(test_path.relative_to(ROOT)),
            }
        )

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(out / "external_homology_baseline_manifest.csv", index=False)
    md = [
        "# External Homology Baseline Inputs",
        "",
        manifest_df.to_markdown(index=False),
        "",
        "All splits use SP-2018 EC-Bench training proteins as the search database.",
    ]
    (out / "external_homology_baseline_manifest.md").write_text("\n".join(md) + "\n")
    print(manifest_df.to_string(index=False))


if __name__ == "__main__":
    main()
