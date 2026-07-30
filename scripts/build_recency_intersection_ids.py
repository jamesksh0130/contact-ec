#!/usr/bin/env python3
"""Build temporal known subset IDs evaluable by all recency label encoders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def evaluable_accessions(path: Path) -> set[str]:
    df = pd.read_csv(path)
    m4 = pd.to_numeric(df["m4"], errors="coerce").fillna(0)
    l4 = pd.to_numeric(df["l4_idx"], errors="coerce").fillna(-1)
    return set(df.loc[(m4 == 1) & (l4 >= 0), "accession"].astype(str))


def main() -> None:
    test_ids_path = ROOT / "data/ecbench/splits/test_ids_full.txt"
    ordered_ids = [x.strip() for x in test_ids_path.read_text().splitlines() if x.strip()]

    paths = [
        ROOT / "data/ecbench/processed/test_meta_full.csv",
        ROOT / "data/ecbench/processed/test_meta_full_2022enc.csv",
        ROOT / "data/ecbench/processed/test_meta_full_expa_enc.csv",
    ]
    sets = [evaluable_accessions(path) for path in paths]
    intersection = set.intersection(*sets)
    ordered_intersection = [acc for acc in ordered_ids if acc in intersection]

    out_path = ROOT / "data/ecbench/splits/test_ids_recency_intersection.txt"
    out_path.write_text("\n".join(ordered_intersection) + "\n", encoding="utf-8")

    audit_path = ROOT / "outputs/audit/recency_intersection_ids.md"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Recency Intersection Temporal IDs",
        "",
        f"Original temporal known IDs: {len(ordered_ids)}",
        f"SP-2018 evaluable: {len(sets[0])}",
        f"SP-2022 evaluable: {len(sets[1])}",
        f"SP-2026/ExpA evaluable: {len(sets[2])}",
        f"Intersection evaluable by all encoders: {len(ordered_intersection)}",
        "",
        f"IDs written to `{out_path.relative_to(ROOT)}`.",
    ]
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {audit_path}")
    print(f"N={len(ordered_intersection)}")


if __name__ == "__main__":
    main()
