#!/usr/bin/env python
"""Break down MMseqs2 top-hit EC-transfer performance by hit identity/e-value."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


IDENTITY_BINS = [
    ("no_hit", None, None),
    ("0-20", 0.0, 20.0),
    ("20-30", 20.0, 30.0),
    ("30-50", 30.0, 50.0),
    ("50-70", 50.0, 70.0),
    (">=70", 70.0, None),
]


EVALUE_BINS = [
    ("no_hit", None, None),
    (">1e-3", 1e-3, None),
    ("1e-10..1e-3", 1e-10, 1e-3),
    ("1e-50..1e-10", 1e-50, 1e-10),
    ("<=1e-50", None, 1e-50),
]


def micro_f1_from_rows(df: pd.DataFrame) -> float:
    tp = fp = fn = 0
    for row in df.itertuples(index=False):
        true = {x for x in str(row.true_ec).split(";") if x}
        pred = {x for x in str(row.pred_ec).split(";") if x}
        tp += len(true & pred)
        fp += len(pred - true)
        fn += len(true - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def identity_bin(value) -> str:
    if pd.isna(value):
        return "no_hit"
    x = float(value)
    for name, lo, hi in IDENTITY_BINS[1:]:
        if hi is None and x >= lo:
            return name
        if lo <= x < hi:
            return name
    return "unknown"


def evalue_bin(value) -> str:
    if pd.isna(value):
        return "no_hit"
    x = float(value)
    for name, lo, hi in EVALUE_BINS[1:]:
        if lo is None and x <= hi:
            return name
        if hi is None and x > lo:
            return name
        if lo is not None and hi is not None and lo < x <= hi:
            return name
    return "unknown"


def summarize(df: pd.DataFrame, split: str, col: str, order: list[str]) -> pd.DataFrame:
    rows = []
    for name in order:
        sub = df[df[col] == name]
        if sub.empty:
            rows.append(
                {
                    "split": split,
                    "bin_type": col,
                    "bin": name,
                    "n": 0,
                    "fraction": 0.0,
                    "l4_sample_hit": 0.0,
                    "l4_micro_f1": 0.0,
                    "mean_sample_f1": 0.0,
                    "median_pident": float("nan"),
                    "median_evalue": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "split": split,
                "bin_type": col,
                "bin": name,
                "n": int(len(sub)),
                "fraction": float(len(sub) / len(df)),
                "l4_sample_hit": float(sub["exact_hit"].mean()),
                "l4_micro_f1": float(micro_f1_from_rows(sub)),
                "mean_sample_f1": float(sub["sample_f1"].mean()),
                "median_pident": float(sub["pident"].median()) if sub["pident"].notna().any() else float("nan"),
                "median_evalue": float(sub["evalue"].median()) if sub["evalue"].notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="outputs/baselines/homology")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["foldseek_tmscore40_cc", "foldseek_tmscore50_cc", "foldseek_tmscore60_cc"],
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.is_absolute():
        base = ROOT / base

    all_rows = []
    for split in args.splits:
        path = base / f"{split}_mmseqs_top1_per_sample.csv"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        df = pd.read_csv(path)
        df["identity_bin"] = df["pident"].map(identity_bin)
        df["evalue_bin"] = df["evalue"].map(evalue_bin)
        df.to_csv(base / f"{split}_mmseqs_top1_per_sample_binned.csv", index=False)
        all_rows.append(summarize(df, split, "identity_bin", [x[0] for x in IDENTITY_BINS]))
        all_rows.append(summarize(df, split, "evalue_bin", [x[0] for x in EVALUE_BINS]))

    if not all_rows:
        raise SystemExit("No per-sample MMseqs2 files found.")
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(base / "mmseqs_homology_bin_breakdown.csv", index=False)

    md = ["# MMseqs2 Homology Bin Breakdown", ""]
    for split in args.splits:
        sub = out[(out["split"] == split) & (out["bin_type"] == "identity_bin")]
        if not sub.empty:
            md.extend([f"## {split}: percent identity", "", sub.to_markdown(index=False, floatfmt=".4f"), ""])
        sub = out[(out["split"] == split) & (out["bin_type"] == "evalue_bin")]
        if not sub.empty:
            md.extend([f"## {split}: e-value", "", sub.to_markdown(index=False, floatfmt=".4g"), ""])
    (base / "mmseqs_homology_bin_breakdown.md").write_text("\n".join(md) + "\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
