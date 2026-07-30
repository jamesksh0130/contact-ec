#!/usr/bin/env python3
"""Audit dataset/vocabulary components behind the temporal recency effect.

This script does not train models. It summarizes how much changes across
available Swiss-Prot cutoff corpora: corpus size, train-split size, Level-4
vocabulary size, direct temporal overlap, exact-sequence overlap, and temporal
known-124 label coverage. These quantities separate a reported "recency" gain
from vocabulary and coverage effects.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def split_ecs(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    out: list[str] = []
    for part in text.replace(",", ";").split(";"):
        ec = part.strip()
        if ec and "-" not in ec:
            out.append(ec)
    return out


def exact_sequence_set(df: pd.DataFrame) -> set[str]:
    if "sequence" not in df.columns:
        return set()
    return set(df["sequence"].dropna().astype(str))


def accession_set(df: pd.DataFrame) -> set[str]:
    return set(df["accession"].dropna().astype(str))


def load_encoder_classes(path: Path) -> set[str]:
    with path.open("rb") as f:
        enc = pickle.load(f)
    return set(map(str, enc["level4"].classes_))


def corpus_l4_classes(df: pd.DataFrame) -> set[str]:
    col = "ec_raw" if "ec_raw" in df.columns else "ec_chosen"
    classes: set[str] = set()
    for value in df[col].dropna():
        classes.update(split_ecs(value))
    return classes


def training_subset(df: pd.DataFrame, split_path: Path | None) -> pd.DataFrame:
    if split_path is None or not split_path.exists():
        return df
    ids = {line.strip() for line in split_path.read_text().splitlines() if line.strip()}
    return df[df["accession"].astype(str).isin(ids)].copy()


def temporal_label_coverage(test_df: pd.DataFrame, vocab: set[str], train_classes: set[str]) -> dict[str, object]:
    all_labels: list[str] = []
    per_protein_all_seen_vocab = 0
    per_protein_all_seen_train = 0
    per_protein_any_unseen_vocab = 0
    per_protein_any_unseen_train = 0
    for value in test_df["ec_raw"]:
        labels = split_ecs(value)
        all_labels.extend(labels)
        if labels and all(ec in vocab for ec in labels):
            per_protein_all_seen_vocab += 1
        if labels and all(ec in train_classes for ec in labels):
            per_protein_all_seen_train += 1
        if any(ec not in vocab for ec in labels):
            per_protein_any_unseen_vocab += 1
        if any(ec not in train_classes for ec in labels):
            per_protein_any_unseen_train += 1

    label_total = len(all_labels)
    vocab_seen = sum(ec in vocab for ec in all_labels)
    train_seen = sum(ec in train_classes for ec in all_labels)
    return {
        "temporal_positive_l4_labels": label_total,
        "label_vocab_coverage": vocab_seen / label_total if label_total else 0.0,
        "label_train_coverage": train_seen / label_total if label_total else 0.0,
        "proteins_all_labels_in_vocab": per_protein_all_seen_vocab,
        "proteins_all_labels_in_train": per_protein_all_seen_train,
        "proteins_any_label_oov_vocab": per_protein_any_unseen_vocab,
        "proteins_any_label_absent_train": per_protein_any_unseen_train,
    }


def class_frequency_stats(train_df: pd.DataFrame, temporal_labels: Iterable[str]) -> dict[str, object]:
    col = "ec_raw" if "ec_raw" in train_df.columns else "ec_chosen"
    counts: dict[str, int] = {}
    for value in train_df[col].dropna():
        for ec in split_ecs(value):
            counts[ec] = counts.get(ec, 0) + 1
    freqs = [counts.get(ec, 0) for ec in temporal_labels]
    if not freqs:
        return {}
    s = pd.Series(freqs)
    return {
        "temporal_label_train_freq_min": int(s.min()),
        "temporal_label_train_freq_median": float(s.median()),
        "temporal_label_train_freq_mean": float(s.mean()),
        "temporal_label_train_freq_max": int(s.max()),
        "temporal_labels_absent_train": int((s == 0).sum()),
        "temporal_labels_rare_le_5": int((s <= 5).sum()),
        "temporal_labels_rare_le_25": int((s <= 25).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-prefix", default="outputs/audit/recency_cutoff_decomposition")
    args = parser.parse_args()

    out_prefix = ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(ROOT / "data/ecbench/processed/test_meta_full.csv")
    test_acc = accession_set(test_df)
    test_seq = exact_sequence_set(test_df)
    temporal_labels = [ec for value in test_df["ec_raw"] for ec in split_ecs(value)]

    corpora = [
        {
            "cutoff": "SP-2018",
            "corpus_path": ROOT / "data/ecbench/processed/train_meta.csv",
            "encoder_path": ROOT / "data/ecbench/label_encoders.pkl",
            "encoded_test_path": ROOT / "data/ecbench/processed/test_meta_full.csv",
            "train_ids": ROOT / "data/ecbench/splits/train_ids.txt",
            "notes": "EC-Bench February 2018 controlled corpus",
        },
        {
            "cutoff": "SP-2022",
            "corpus_path": ROOT / "data/processed/dataset_meta_2022.csv",
            "encoder_path": ROOT / "data/label_encoders_2022.pkl",
            "encoded_test_path": ROOT / "data/ecbench/processed/test_meta_full_2022enc.csv",
            "train_ids": ROOT / "data/splits_2022/train_ids.txt",
            "notes": "Local 2022 Swiss-Prot/HIT-EC-derived corpus",
        },
        {
            "cutoff": "SP-2026/ExpA",
            "corpus_path": ROOT / "data/expa/dataset_meta_reenc.csv",
            "encoder_path": ROOT / "data/expa/label_encoders.pkl",
            "encoded_test_path": ROOT / "data/ecbench/processed/test_meta_full_expa_enc.csv",
            "train_ids": ROOT / "data/expa/splits/train_ids.txt",
            "notes": "Recent-data ExpA corpus used for recency control",
        },
    ]

    rows: list[dict[str, object]] = []
    for spec in corpora:
        df = pd.read_csv(spec["corpus_path"])
        train_df = training_subset(df, spec["train_ids"])
        encoded_test_df = pd.read_csv(spec["encoded_test_path"])
        vocab = load_encoder_classes(spec["encoder_path"])
        train_classes = corpus_l4_classes(train_df)
        cov = temporal_label_coverage(test_df, vocab, train_classes)
        freq = class_frequency_stats(train_df, temporal_labels)
        rows.append(
            {
                "cutoff": spec["cutoff"],
                "corpus_proteins": len(df),
                "train_split_proteins": len(train_df),
                "encoder_l4_classes": len(vocab),
                "train_l4_classes": len(train_classes),
                "temporal_accession_overlap_corpus": len(test_acc & accession_set(df)),
                "temporal_accession_overlap_train": len(test_acc & accession_set(train_df)),
                "temporal_exact_sequence_overlap_corpus": len(test_seq & exact_sequence_set(df)),
                "temporal_exact_sequence_overlap_train": len(test_seq & exact_sequence_set(train_df)),
                "encoded_test_proteins": len(encoded_test_df),
                "encoded_test_m4_evaluable": int(pd.to_numeric(encoded_test_df["m4"], errors="coerce").fillna(0).sum()),
                "encoded_test_l4_idx_missing": int((pd.to_numeric(encoded_test_df["l4_idx"], errors="coerce").fillna(-1) < 0).sum()),
                **cov,
                **freq,
                "notes": spec["notes"],
            }
        )

    result = pd.DataFrame(rows)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    result.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    display = result.copy()
    for col in ["label_vocab_coverage", "label_train_coverage"]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    md = [
        "# Recency Cutoff Decomposition Audit",
        "",
        "Temporal subset: complete Level-4 Swiss-Prot 2023-01 proteins (N=124).",
        "This audit separates corpus recency from sample size, label vocabulary, direct overlap, and temporal-label train coverage.",
        "",
        display.to_markdown(index=False),
        "",
        "Interpretation:",
        "- Accession/exact-sequence overlap should be zero for a leakage-proof temporal evaluation.",
        "- Encoded-test m4 coverage is the model's actual Level-4 scoring denominator for that label encoder.",
        "- Vocabulary and train-label coverage quantify whether temporal gains can be explained by label-space expansion.",
        "- Frequency statistics show whether temporal labels become less rare as newer corpora are used.",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
