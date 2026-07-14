#!/usr/bin/env python3
"""Audit temporal-test overlap against SP-2018 and ExpA training corpora."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "audit"


def sha1_sequence(seq: str) -> str:
    return hashlib.sha1(str(seq).encode("utf-8")).hexdigest()


def load_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def summarize_train_vs_test(tag: str, train_meta: Path, train_ids: Path | None, test_meta: Path) -> dict:
    train = pd.read_csv(train_meta)
    test = pd.read_csv(test_meta)

    if train_ids is not None and train_ids.exists():
        ids = load_ids(train_ids)
        train = train[train["accession"].astype(str).isin(ids)].copy()

    train_acc = set(train["accession"].astype(str))
    test_acc = set(test["accession"].astype(str))

    train_hash = set(train["sequence"].astype(str).map(sha1_sequence))
    test_hash = set(test["sequence"].astype(str).map(sha1_sequence))

    acc_overlap = sorted(train_acc & test_acc)
    seq_overlap = sorted(train_hash & test_hash)

    return {
        "training_corpus": tag,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_unique_accessions": int(len(train_acc)),
        "test_unique_accessions": int(len(test_acc)),
        "accession_overlap_n": int(len(acc_overlap)),
        "accession_overlap_examples": acc_overlap[:10],
        "exact_sequence_overlap_n": int(len(seq_overlap)),
        "train_unique_l4_labels": int(train["l4_idx"].nunique()) if "l4_idx" in train.columns else None,
        "test_unique_l4_labels": int(test["l4_idx"].nunique()) if "l4_idx" in test.columns else None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    test_meta = ROOT / "data" / "ecbench" / "processed" / "test_meta_full.csv"

    rows = [
        summarize_train_vs_test(
            "SP-2018 EC-Bench train",
            ROOT / "data" / "ecbench" / "processed" / "train_meta.csv",
            ROOT / "data" / "ecbench" / "splits" / "train_ids.txt",
            test_meta,
        ),
        summarize_train_vs_test(
            "ExpA recent Swiss-Prot train",
            ROOT / "data" / "expa" / "dataset_meta_reenc.csv",
            ROOT / "data" / "expa" / "splits" / "train_ids.txt",
            test_meta,
        ),
    ]

    sim_path = ROOT / "data" / "ecbench" / "splits" / "test_vs_train_sim.json"
    if sim_path.exists():
        sim = json.loads(sim_path.read_text())
        sims = [float(v) for v in sim.values()]
        rows[0].update(
            {
                "max_recorded_train_similarity": round(max(sims), 4) if sims else None,
                "n_test_with_similarity_ge_0_30": int(sum(v >= 0.30 for v in sims)),
                "n_test_with_similarity_ge_0_90": int(sum(v >= 0.90 for v in sims)),
                "similarity_note": "Values are from the available EC-Bench test_vs_train_sim.json audit file.",
            }
        )

    pd.DataFrame(rows).to_csv(OUT / "temporal_leakage_audit_final.csv", index=False)
    (OUT / "temporal_leakage_audit_final.json").write_text(json.dumps(rows, indent=2))

    lines = [
        "# Temporal Leakage Audit",
        "",
        "Audit against the full evaluable Swiss-Prot 2023-01 temporal subset.",
        "",
        "| Training corpus | Train rows | Test rows | Accession overlap | Exact sequence overlap | Train L4 labels | Test L4 labels |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['training_corpus']} | {row['train_rows']} | {row['test_rows']} | "
            f"{row['accession_overlap_n']} | {row['exact_sequence_overlap_n']} | "
            f"{row['train_unique_l4_labels']} | {row['test_unique_l4_labels']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: accession and exact-sequence overlap are necessary but not sufficient leakage checks.",
            "They do not replace homolog- or fold-disjoint evaluation.",
        ]
    )
    (OUT / "temporal_leakage_audit_final.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
