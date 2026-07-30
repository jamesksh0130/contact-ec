#!/usr/bin/env python
"""Evaluate a top-hit MMseqs2 EC-transfer baseline on Foldseek-disjoint splits."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EC_RE = re.compile(r"EC=([^\s]+)")


def parse_header_ec(value: str) -> tuple[str, tuple[str, ...]]:
    text = str(value)
    acc = text.split("|", 1)[0]
    match = EC_RE.search(text)
    if not match:
        return acc, tuple()
    ecs = tuple(ec.strip() for ec in re.split(r"[;,|]", match.group(1)) if ec.strip())
    return acc, ecs


def read_fasta_labels(path: Path) -> dict[str, tuple[str, ...]]:
    labels: dict[str, tuple[str, ...]] = {}
    with path.open() as f:
        for line in f:
            if not line.startswith(">"):
                continue
            acc, ecs = parse_header_ec(line[1:].strip())
            labels[acc] = ecs
    return labels


def prefix(ec: str, level: int) -> str:
    parts = ec.split(".")
    if len(parts) < level:
        return ec
    return ".".join(parts[:level])


def transfer_metrics(
    query_labels: dict[str, tuple[str, ...]],
    target_labels: dict[str, tuple[str, ...]],
    hits: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    best: dict[str, dict] = {}
    if not hits.empty:
        hits = hits.copy()
        hits["query_acc"] = hits["query"].astype(str).str.split("|").str[0]
        hits["target_acc"] = hits["target"].astype(str).str.split("|").str[0]
        hits = hits.sort_values(["query_acc", "evalue", "bits"], ascending=[True, True, False])
        for row in hits.itertuples(index=False):
            if row.query_acc not in best:
                best[row.query_acc] = {
                    "target": row.target_acc,
                    "pident": float(row.pident),
                    "alnlen": int(row.alnlen),
                    "evalue": float(row.evalue),
                    "bits": float(row.bits),
                }

    rows = []
    tp = fp = fn = 0
    level_hits = {1: 0, 2: 0, 3: 0, 4: 0}
    covered = 0

    for acc, true_ecs in query_labels.items():
        hit = best.get(acc)
        pred_ecs: tuple[str, ...] = tuple()
        if hit is not None:
            pred_ecs = target_labels.get(hit["target"], tuple())
        if pred_ecs:
            covered += 1

        true_set = set(true_ecs)
        pred_set = set(pred_ecs)
        tp_i = len(true_set & pred_set)
        fp_i = len(pred_set - true_set)
        fn_i = len(true_set - pred_set)
        tp += tp_i
        fp += fp_i
        fn += fn_i

        per_level = {}
        for level in (1, 2, 3, 4):
            true_pref = {prefix(ec, level) for ec in true_ecs}
            pred_pref = {prefix(ec, level) for ec in pred_ecs}
            ok = bool(true_pref and pred_pref and (true_pref & pred_pref))
            per_level[f"level{level}_hit"] = ok
            level_hits[level] += int(ok)

        rows.append(
            {
                "accession": acc,
                "true_ec": ";".join(true_ecs),
                "pred_ec": ";".join(pred_ecs),
                "covered": bool(pred_ecs),
                "exact_hit": bool(tp_i > 0),
                "sample_f1": (2 * tp_i / (2 * tp_i + fp_i + fn_i))
                if (2 * tp_i + fp_i + fn_i)
                else 0.0,
                "target": hit["target"] if hit else "",
                "pident": hit["pident"] if hit else math.nan,
                "alnlen": hit["alnlen"] if hit else math.nan,
                "evalue": hit["evalue"] if hit else math.nan,
                "bits": hit["bits"] if hit else math.nan,
                **per_level,
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = len(query_labels)
    per_sample = pd.DataFrame(rows)
    metrics = {
        "n_samples": n,
        "coverage": covered / n if n else 0.0,
        "l1_accuracy": level_hits[1] / n if n else 0.0,
        "l2_accuracy": level_hits[2] / n if n else 0.0,
        "l3_accuracy": level_hits[3] / n if n else 0.0,
        "l4_sample_hit": level_hits[4] / n if n else 0.0,
        "l4_micro_f1": micro_f1,
        "l4_precision": precision,
        "l4_recall": recall,
        "mean_sample_f1": float(per_sample["sample_f1"].mean()) if n else 0.0,
    }
    return metrics, per_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="outputs/baselines/homology")
    parser.add_argument("--splits", nargs="+", default=["foldseek_tmscore50_cc"])
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.is_absolute():
        base = ROOT / base
    summary = []
    for split in args.splits:
        train_fasta = base / f"{split}_train.fasta"
        test_fasta = base / f"{split}_test.fasta"
        hit_tsv = base / f"{split}_mmseqs_top10.tsv"
        if not hit_tsv.exists():
            print(f"[skip] missing {hit_tsv}")
            continue
        train_labels = read_fasta_labels(train_fasta)
        test_labels = read_fasta_labels(test_fasta)
        hits = pd.read_csv(
            hit_tsv,
            sep="\t",
            names=["query", "target", "pident", "alnlen", "evalue", "bits"],
        )
        metrics, per_sample = transfer_metrics(test_labels, train_labels, hits)
        metrics = {"split": split, **metrics}
        summary.append(metrics)
        per_sample.to_csv(base / f"{split}_mmseqs_top1_per_sample.csv", index=False)
        with (base / f"{split}_mmseqs_top1_metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2)

    if not summary:
        raise SystemExit("No MMseqs result files were evaluated.")
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(base / "mmseqs_homology_baseline_summary.csv", index=False)
    md = [
        "# MMseqs2 Homology Baseline",
        "",
        summary_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The baseline transfers all Level-4 EC labels from the single best MMseqs2 hit in the training partition.",
    ]
    (base / "mmseqs_homology_baseline_summary.md").write_text("\n".join(md) + "\n")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
