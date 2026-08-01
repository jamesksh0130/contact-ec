#!/usr/bin/env python3
"""Build recent-corpus controls under the fixed SP-2018 EC-Bench vocabulary.

The resulting metadata re-encodes newer Swiss-Prot rows with the EC-Bench
SP-2018 label encoders and keeps only complete Level-4 labels already present in
that old vocabulary. This separates vocabulary expansion from newer-sample and
homolog-coverage effects.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    meta_csv: Path
    train_ids: Path
    val_ids: Path
    ec_col: str


SOURCES = {
    "sp2022": SourceSpec(
        "sp2022",
        ROOT / "data" / "processed" / "dataset_meta_2022.csv",
        ROOT / "data" / "splits_2022" / "train_ids.txt",
        ROOT / "data" / "splits_2022" / "val_ids.txt",
        "ec_chosen",
    ),
    "sp2026": SourceSpec(
        "sp2026",
        ROOT / "data" / "expa" / "dataset_meta_reenc.csv",
        ROOT / "data" / "expa" / "splits" / "train_ids.txt",
        ROOT / "data" / "expa" / "splits" / "val_ids.txt",
        "ec_chosen",
    ),
}


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def split_ecs(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    ecs = []
    for token in str(value).replace(";", ",").split(","):
        ec = token.strip()
        if ec and "-" not in ec and ec.count(".") == 3:
            ecs.append(ec)
    return sorted(set(ecs))


def encode_row(
    row: pd.Series,
    class_sets: dict[int, set[str]],
    class_to_idx: dict[int, dict[str, int]],
    ec_col: str,
) -> dict[str, object] | None:
    ecs = split_ecs(row.get(ec_col))
    if not ecs:
        return None

    level_sets = {}
    for level in range(1, 5):
        level_sets[level] = {".".join(ec.split(".")[:level]) for ec in ecs}

    encoded_levels = {}
    for level in range(1, 5):
        if not level_sets[level].issubset(class_sets[level]):
            return None
        encoded_levels[level] = sorted(level_sets[level])

    l4_indices = sorted(class_to_idx[4][ec] for ec in encoded_levels[4])
    primary_l4 = encoded_levels[4][0]
    primary_levels = {
        1: primary_l4.split(".")[0],
        2: ".".join(primary_l4.split(".")[:2]),
        3: ".".join(primary_l4.split(".")[:3]),
        4: primary_l4,
    }

    out = {
        "accession": str(row["accession"]),
        "sequence": str(row["sequence"]),
        "seq_len": int(row.get("seq_len", len(str(row["sequence"])))),
        "ec_raw": ";".join(ecs),
        "l4_all_idxs": "|".join(str(i) for i in l4_indices),
    }
    for level in range(1, 5):
        out[f"l{level}_idx"] = class_to_idx[level][primary_levels[level]]
        out[f"m{level}"] = 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCES), default="sp2026")
    parser.add_argument("--label-enc", default="data/ecbench/label_encoders.pkl")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    spec = SOURCES[args.source]
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data" / "recency_fixed_vocab" / args.source
    splits_dir = out_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    with (ROOT / args.label_enc).open("rb") as f:
        encoders = pickle.load(f)
    class_sets = {
        level: set(map(str, encoders[f"level{level}"].classes_)) for level in range(1, 5)
    }
    class_to_idx = {
        level: {str(c): int(i) for i, c in enumerate(encoders[f"level{level}"].classes_)}
        for level in range(1, 5)
    }

    temporal_ids = set(read_ids(ROOT / "data" / "ecbench" / "splits" / "test_ids_full.txt"))
    source = pd.read_csv(spec.meta_csv)
    source["accession"] = source["accession"].astype(str)
    source_by_acc = source.set_index("accession", drop=False)

    rows = []
    accepted = set()
    rejected_old_vocab = 0
    rejected_temporal = 0
    rejected_missing = 0
    for row in source.itertuples(index=False):
        acc = str(row.accession)
        if acc in temporal_ids:
            rejected_temporal += 1
            continue
        encoded = encode_row(pd.Series(row._asdict()), class_sets, class_to_idx, spec.ec_col)
        if encoded is None:
            if not split_ecs(getattr(row, spec.ec_col)):
                rejected_missing += 1
            else:
                rejected_old_vocab += 1
            continue
        rows.append(encoded)
        accepted.add(acc)

    meta = pd.DataFrame(rows).sort_values("accession")
    meta_csv = out_dir / "dataset_meta_oldvocab.csv"
    meta.to_csv(meta_csv, index=False)

    def write_split(src_ids_path: Path, dst_name: str) -> int:
        ids = [acc for acc in read_ids(src_ids_path) if acc in accepted and acc in source_by_acc.index]
        (splits_dir / f"{dst_name}_ids.txt").write_text("\n".join(ids) + "\n")
        return len(ids)

    train_n = write_split(spec.train_ids, "train")
    val_n = write_split(spec.val_ids, "val")

    audit = {
        "source": spec.name,
        "source_meta": str(spec.meta_csv.relative_to(ROOT)),
        "old_label_encoder": args.label_enc,
        "n_source_rows": int(len(source)),
        "n_encoded_old_vocab": int(len(meta)),
        "n_train": train_n,
        "n_val": val_n,
        "n_temporal_accessions_excluded": rejected_temporal,
        "n_rejected_missing_complete_ec": rejected_missing,
        "n_rejected_outside_old_vocab": rejected_old_vocab,
        "n_l4_old_vocab": int(len(encoders["level4"].classes_)),
        "meta_csv": str(meta_csv.relative_to(ROOT)),
        "splits_dir": str(splits_dir.relative_to(ROOT)),
    }
    (out_dir / "fixed_vocab_control_audit.json").write_text(json.dumps(audit, indent=2))
    md = [
        f"# Fixed-vocabulary recency control: {spec.name}",
        "",
        "This dataset keeps the SP-2018 EC-Bench label encoder fixed and retains only",
        "newer-corpus proteins whose complete Level-4 EC labels are already present",
        "in the old vocabulary. Temporal test accessions are excluded from training",
        "and validation candidate rows.",
        "",
        f"- Source rows: {audit['n_source_rows']:,}",
        f"- Encoded old-vocabulary rows: {audit['n_encoded_old_vocab']:,}",
        f"- Train rows: {train_n:,}",
        f"- Val rows: {val_n:,}",
        f"- Temporal accessions excluded: {rejected_temporal:,}",
        f"- Complete EC rows outside old vocabulary: {rejected_old_vocab:,}",
        f"- Old L4 vocabulary size: {audit['n_l4_old_vocab']:,}",
    ]
    (out_dir / "fixed_vocab_control_audit.md").write_text("\n".join(md) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
