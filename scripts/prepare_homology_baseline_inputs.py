#!/usr/bin/env python
"""Prepare FASTA inputs for BLAST/DIAMOND homology baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    lines = []
    for row in df.itertuples(index=False):
        lines.append(f">{row.accession}|EC={row.ec_raw}")
        seq = str(row.sequence)
        for i in range(0, len(seq), 80):
            lines.append(seq[i : i + 80])
    path.write_text("\n".join(lines) + "\n")


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-csv", default="data/ecbench/processed/train_meta.csv")
    parser.add_argument("--splits-dir", default="data/ecbench/splits")
    parser.add_argument("--output-dir", default="outputs/baselines/homology")
    args = parser.parse_args()

    meta = pd.read_csv(ROOT / args.meta_csv)
    splits_dir = ROOT / args.splits_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split_prefixes = [
        "foldseek_tmscore40_cc",
        "foldseek_tmscore50_cc",
        "foldseek_tmscore60_cc",
    ]
    manifest = []
    for prefix in split_prefixes:
        train_ids = read_ids(splits_dir / f"{prefix}_train_ids.txt")
        test_ids = read_ids(splits_dir / f"{prefix}_test_ids.txt")
        train_df = meta[meta["accession"].isin(train_ids)].copy()
        test_df = meta[meta["accession"].isin(test_ids)].copy()
        train_fasta = out_dir / f"{prefix}_train.fasta"
        test_fasta = out_dir / f"{prefix}_test.fasta"
        write_fasta(train_df, train_fasta)
        write_fasta(test_df, test_fasta)
        manifest.append(
            {
                "split": prefix,
                "train_n": len(train_df),
                "test_n": len(test_df),
                "train_fasta": str(train_fasta.relative_to(ROOT)),
                "test_fasta": str(test_fasta.relative_to(ROOT)),
            }
        )

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(out_dir / "homology_baseline_manifest.csv", index=False)
    md = [
        "# Homology Baseline Inputs",
        "",
        manifest_df.to_markdown(index=False),
        "",
        "## DIAMOND commands",
        "",
        "```bash",
        "diamond makedb --in outputs/baselines/homology/foldseek_tmscore50_cc_train.fasta -d outputs/baselines/homology/foldseek_tmscore50_cc_train",
        "diamond blastp -d outputs/baselines/homology/foldseek_tmscore50_cc_train -q outputs/baselines/homology/foldseek_tmscore50_cc_test.fasta -o outputs/baselines/homology/foldseek_tmscore50_cc_diamond.tsv --outfmt 6 qseqid sseqid pident length evalue bitscore --max-target-seqs 10",
        "```",
        "",
        "DIAMOND/BLAST is not invoked by this preparation script; install the binary first, then run the command for each split.",
    ]
    (out_dir / "README.md").write_text("\n".join(md) + "\n")
    print(manifest_df.to_string(index=False))


if __name__ == "__main__":
    main()
