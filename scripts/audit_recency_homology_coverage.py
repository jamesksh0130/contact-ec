#!/usr/bin/env python3
"""Audit whether recency gains track sequence-neighbor and label coverage.

This script searches the complete-known temporal EC-Bench proteins against the
training split used by each cutoff/retraining condition. It reports the nearest
MMseqs2 hit identity and whether the temporal EC labels are present in the
training vocabulary. The audit is intentionally descriptive: it separates
possible recency, corpus-size, vocabulary, and homolog-coverage effects instead
of treating the ExpA gain as a pure time effect.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "audit"


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    meta_path: Path
    train_ids_path: Path
    ec_col: str


CORPORA = [
    CorpusSpec(
        "SP-2018",
        ROOT / "data" / "ecbench" / "processed" / "train_meta.csv",
        ROOT / "data" / "ecbench" / "splits" / "train_ids.txt",
        "ec_raw",
    ),
    CorpusSpec(
        "SP-2022",
        ROOT / "data" / "processed" / "dataset_meta_2022.csv",
        ROOT / "data" / "splits_2022" / "train_ids.txt",
        "ec_chosen",
    ),
    CorpusSpec(
        "SP-2026-ExpA",
        ROOT / "data" / "expa" / "dataset_meta_reenc.csv",
        ROOT / "data" / "expa" / "splits" / "train_ids.txt",
        "ec_chosen",
    ),
]


def read_ids(path: Path) -> set[str]:
    with path.open() as f:
        return {line.strip() for line in f if line.strip()}


def split_ecs(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    ecs: list[str] = []
    for token in str(value).replace(";", ",").split(","):
        ec = token.strip()
        if not ec or "-" in ec:
            continue
        if ec.count(".") == 3:
            ecs.append(ec)
    return sorted(set(ecs))


def ec_prefixes(ecs: Iterable[str], level: int) -> set[str]:
    return {".".join(ec.split(".")[:level]) for ec in ecs if ec.count(".") >= level - 1}


def load_test() -> pd.DataFrame:
    test = pd.read_csv(ROOT / "data" / "ecbench" / "processed" / "test_meta_full.csv")
    test = test[test["m4"].astype(int) == 1].copy()
    test["true_ecs"] = test["ec_raw"].map(split_ecs)
    test = test[test["true_ecs"].map(bool)].copy()
    return test[["accession", "sequence", "seq_len", "ec_raw", "true_ecs"]]


def load_train(spec: CorpusSpec) -> pd.DataFrame:
    meta = pd.read_csv(spec.meta_path)
    train_ids = read_ids(spec.train_ids_path)
    meta = meta[meta["accession"].astype(str).isin(train_ids)].copy()
    meta["train_ecs"] = meta[spec.ec_col].map(split_ecs)
    meta = meta[meta["train_ecs"].map(bool)].copy()
    return meta[["accession", "sequence", "seq_len", spec.ec_col, "train_ecs"]].rename(
        columns={spec.ec_col: "ec_raw"}
    )


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with path.open("w") as f:
        for row in df.itertuples(index=False):
            f.write(f">{row.accession}\n")
            seq = str(row.sequence)
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def run_mmseqs(
    mmseqs: str,
    query_fasta: Path,
    target_fasta: Path,
    out_tsv: Path,
    threads: int,
    force: bool,
) -> None:
    if out_tsv.exists() and not force:
        return
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mmseqs_recency_", dir=str(OUT_DIR)) as tmp:
        cmd = [
            mmseqs,
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(out_tsv),
            tmp,
            "--format-output",
            "query,target,pident,alnlen,qlen,tlen,evalue,bits",
            "--threads",
            str(threads),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)


def parse_top_hits(path: Path) -> pd.DataFrame:
    cols = ["query", "target", "pident", "alnlen", "qlen", "tlen", "evalue", "bits"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols + ["identity", "qcov", "tcov"])
    hits = pd.read_csv(path, sep="\t", names=cols)
    for col in ["pident", "alnlen", "qlen", "tlen", "evalue", "bits"]:
        hits[col] = pd.to_numeric(hits[col], errors="coerce")
    hits["identity"] = hits["pident"] / 100.0
    hits["qcov"] = hits["alnlen"] / hits["qlen"]
    hits["tcov"] = hits["alnlen"] / hits["tlen"]
    hits = hits.sort_values(["query", "bits", "evalue"], ascending=[True, False, True])
    return hits.drop_duplicates("query", keep="first")


def same_prefix(true_ecs: list[str], hit_ecs: list[str], level: int) -> bool:
    return bool(ec_prefixes(true_ecs, level) & ec_prefixes(hit_ecs, level))


def summarize(per_protein: pd.DataFrame, subset_name: str, subset_ids: set[str]) -> pd.DataFrame:
    rows = []
    for cutoff, group in per_protein.groupby("cutoff"):
        sub = group[group["accession"].isin(subset_ids)].copy()
        hit = sub["top_identity"].notna()
        rows.append(
            {
                "subset": subset_name,
                "cutoff": cutoff,
                "n": len(sub),
                "n_with_hit": int(hit.sum()),
                "median_top_identity": float(sub.loc[hit, "top_identity"].median()) if hit.any() else float("nan"),
                "mean_top_identity": float(sub.loc[hit, "top_identity"].mean()) if hit.any() else float("nan"),
                "n_identity_ge_30": int((sub["top_identity"] >= 0.30).sum()),
                "n_identity_ge_60": int((sub["top_identity"] >= 0.60).sum()),
                "n_identity_ge_90": int((sub["top_identity"] >= 0.90).sum()),
                "n_identity_ge_99": int((sub["top_identity"] >= 0.99).sum()),
                "proteins_all_l4_in_train": int(sub["true_all_l4_in_train"].sum()),
                "proteins_any_l4_absent_train": int(sub["true_any_l4_absent_train"].sum()),
                "top_hit_same_l1": int(sub["top_hit_same_l1"].sum()),
                "top_hit_same_l2": int(sub["top_hit_same_l2"].sum()),
                "top_hit_same_l3": int(sub["top_hit_same_l3"].sum()),
                "top_hit_same_l4": int(sub["top_hit_same_l4"].sum()),
            }
        )
    summary = pd.DataFrame(rows)
    for col in [
        "n_with_hit",
        "n_identity_ge_30",
        "n_identity_ge_60",
        "n_identity_ge_90",
        "n_identity_ge_99",
        "proteins_all_l4_in_train",
        "top_hit_same_l1",
        "top_hit_same_l2",
        "top_hit_same_l3",
        "top_hit_same_l4",
    ]:
        summary[col + "_frac"] = summary[col] / summary["n"]
    return summary


def write_markdown(summary: pd.DataFrame, per_protein: pd.DataFrame, path: Path) -> None:
    display_cols = [
        "subset",
        "cutoff",
        "n",
        "median_top_identity",
        "n_identity_ge_30",
        "n_identity_ge_60",
        "n_identity_ge_90",
        "n_identity_ge_99",
        "proteins_all_l4_in_train",
        "top_hit_same_l4",
    ]
    md = []
    md.append("# Recency Homology and Label-Coverage Audit\n")
    md.append(
        "This audit searches complete-known temporal EC-Bench proteins against each "
        "training split with MMseqs2 and reports nearest-neighbor identity, "
        "training label coverage, and top-hit EC agreement.\n"
    )
    md.append("## Summary\n")
    md.append(summary[display_cols].to_markdown(index=False, floatfmt=".3f"))
    md.append("\n## Interpretation\n")
    md.append(
        "- Increased ExpA performance should be interpreted together with any change in "
        "nearest-neighbor identity and training label coverage.\n"
    )
    md.append(
        "- If newer cutoffs show higher identity or more labels observed in training, "
        "the gain is not a pure architectural or pure calendar-time effect; it also "
        "reflects corpus expansion, vocabulary coverage, and homolog availability.\n"
    )
    md.append(
        "- This audit does not replace fold-disjoint evaluation. It is a sequence-level "
        "coverage control that should be reported alongside Foldseek/TM-align split "
        "results.\n"
    )
    md.append("\n## Files\n")
    md.append("- `recency_homology_coverage_per_protein.csv`\n")
    md.append("- `recency_homology_coverage_summary.csv`\n")
    md.append("- cached MMseqs2 result TSVs: `recency_homology_mmseqs_*.tsv`\n")
    md.append(f"\nGenerated rows: {len(per_protein)} per-protein cutoff records.\n")
    path.write_text("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="Rerun MMseqs2 even if cached TSVs exist.")
    parser.add_argument("--mmseqs", default=shutil.which("mmseqs") or "mmseqs")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test = load_test()
    intersection_ids = read_ids(ROOT / "data" / "ecbench" / "splits" / "test_ids_recency_intersection.txt")

    query_fasta = OUT_DIR / "recency_homology_temporal_complete_known.fasta"
    write_fasta(test, query_fasta)

    all_rows = []
    for spec in CORPORA:
        print(f"[audit] loading {spec.name}", flush=True)
        train = load_train(spec)
        train_fasta = OUT_DIR / f"recency_homology_train_{spec.name}.fasta"
        result_tsv = OUT_DIR / f"recency_homology_mmseqs_{spec.name}.tsv"
        write_fasta(train, train_fasta)
        print(f"[audit] mmseqs {spec.name}: {len(test)} queries vs {len(train)} train proteins", flush=True)
        run_mmseqs(args.mmseqs, query_fasta, train_fasta, result_tsv, args.threads, args.force)

        top = parse_top_hits(result_tsv)
        train_by_acc = train.set_index("accession")
        train_label_counts = {}
        for ecs in train["train_ecs"]:
            for ec in ecs:
                train_label_counts[ec] = train_label_counts.get(ec, 0) + 1
        train_labels = set(train_label_counts)

        for row in test.itertuples(index=False):
            hit = top[top["query"] == row.accession]
            true_ecs = list(row.true_ecs)
            freqs = [train_label_counts.get(ec, 0) for ec in true_ecs]
            rec = {
                "cutoff": spec.name,
                "accession": row.accession,
                "seq_len": row.seq_len,
                "true_ecs": ";".join(true_ecs),
                "true_all_l4_in_train": all(ec in train_labels for ec in true_ecs),
                "true_any_l4_absent_train": any(ec not in train_labels for ec in true_ecs),
                "true_l4_train_freq_min": min(freqs) if freqs else 0,
                "true_l4_train_freq_median": float(pd.Series(freqs).median()) if freqs else 0.0,
                "true_l4_train_freq_max": max(freqs) if freqs else 0,
                "in_recency_intersection_99": row.accession in intersection_ids,
            }
            if hit.empty:
                rec.update(
                    {
                        "top_hit": "",
                        "top_identity": float("nan"),
                        "top_qcov": float("nan"),
                        "top_tcov": float("nan"),
                        "top_bits": float("nan"),
                        "top_hit_ecs": "",
                        "top_hit_same_l1": False,
                        "top_hit_same_l2": False,
                        "top_hit_same_l3": False,
                        "top_hit_same_l4": False,
                    }
                )
            else:
                h = hit.iloc[0]
                hit_ecs = train_by_acc.loc[h["target"], "train_ecs"] if h["target"] in train_by_acc.index else []
                rec.update(
                    {
                        "top_hit": h["target"],
                        "top_identity": float(h["identity"]),
                        "top_qcov": float(h["qcov"]),
                        "top_tcov": float(h["tcov"]),
                        "top_bits": float(h["bits"]),
                        "top_hit_ecs": ";".join(hit_ecs),
                        "top_hit_same_l1": same_prefix(true_ecs, hit_ecs, 1),
                        "top_hit_same_l2": same_prefix(true_ecs, hit_ecs, 2),
                        "top_hit_same_l3": same_prefix(true_ecs, hit_ecs, 3),
                        "top_hit_same_l4": same_prefix(true_ecs, hit_ecs, 4),
                    }
                )
            all_rows.append(rec)

    per_protein = pd.DataFrame(all_rows)
    all_ids = set(test["accession"])
    summary = pd.concat(
        [
            summarize(per_protein, "complete-known-124", all_ids),
            summarize(per_protein, "recency-intersection-99", intersection_ids),
        ],
        ignore_index=True,
    )

    per_path = OUT_DIR / "recency_homology_coverage_per_protein.csv"
    summary_path = OUT_DIR / "recency_homology_coverage_summary.csv"
    json_path = OUT_DIR / "recency_homology_coverage_summary.json"
    md_path = OUT_DIR / "recency_homology_coverage.md"

    per_protein.to_csv(per_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(summary.to_dict(orient="records"), indent=2))
    write_markdown(summary, per_protein, md_path)
    print(f"[audit] wrote {summary_path.relative_to(ROOT)}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
