#!/usr/bin/env python3
"""Audit contact-map availability and coarse structural-input statistics.

This is a reviewer-facing diagnostic: it checks whether reported splits differ
substantially in sequence length, contact-map availability, zero maps, density,
or long-range-contact ratio. It does not require model inference.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SplitSpec:
    name: str
    ids_file: Path
    meta_csv: Path


DEFAULT_SPLITS = [
    SplitSpec(
        "ecbench_train",
        ROOT / "data/ecbench/splits/train_ids.txt",
        ROOT / "data/ecbench/processed/train_meta.csv",
    ),
    SplitSpec(
        "ecbench_val",
        ROOT / "data/ecbench/splits/val_ids.txt",
        ROOT / "data/ecbench/processed/train_meta.csv",
    ),
    SplitSpec(
        "temporal_known124",
        ROOT / "data/ecbench/splits/test_ids_full.txt",
        ROOT / "data/ecbench/processed/test_meta_full.csv",
    ),
    SplitSpec(
        "temporal_intersection99",
        ROOT / "data/ecbench/splits/test_ids_recency_intersection.txt",
        ROOT / "data/ecbench/processed/test_meta_full.csv",
    ),
    SplitSpec(
        "foldseek50_train",
        ROOT / "data/ecbench/splits/foldseek_tmscore50_cc_train_ids.txt",
        ROOT / "data/ecbench/processed/train_meta.csv",
    ),
    SplitSpec(
        "foldseek50_val",
        ROOT / "data/ecbench/splits/foldseek_tmscore50_cc_val_ids.txt",
        ROOT / "data/ecbench/processed/train_meta.csv",
    ),
    SplitSpec(
        "foldseek50_test",
        ROOT / "data/ecbench/splits/foldseek_tmscore50_cc_test_ids.txt",
        ROOT / "data/ecbench/processed/train_meta.csv",
    ),
    SplitSpec(
        "price149",
        ROOT / "data/ecbench/splits/price149_ids.txt",
        ROOT / "data/ecbench/processed/price149_meta.csv",
    ),
]


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def as_binary_contact(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 3:
        raw = raw[0]
    if raw.shape != (256, 256):
        return np.zeros((256, 256), dtype=bool)
    return raw > 0.5


def contact_stats(acc: str, cmap_dir: Path) -> dict:
    path = cmap_dir / f"{acc}.npy"
    if not path.exists():
        return {
            "accession": acc,
            "cmap_found": False,
            "cmap_valid_shape": False,
            "zero_map": True,
            "contact_density": np.nan,
            "short_contact_density": np.nan,
            "long_contact_density": np.nan,
            "long_contact_fraction": np.nan,
        }

    raw = np.load(path, mmap_mode="r")
    valid_shape = raw.shape == (256, 256) or raw.shape == (3, 256, 256)
    binary = as_binary_contact(np.asarray(raw))
    zero_map = not bool(binary.any())

    idx = np.arange(256)
    diff = np.abs(idx[:, None] - idx[None, :])
    upper = np.triu(np.ones((256, 256), dtype=bool), k=1)
    short_mask = upper & (diff < 12)
    long_mask = upper & (diff >= 12)
    upper_contacts = binary & upper

    n_upper = int(upper.sum())
    n_short = int(short_mask.sum())
    n_long = int(long_mask.sum())
    c_upper = int(upper_contacts.sum())
    c_short = int((binary & short_mask).sum())
    c_long = int((binary & long_mask).sum())

    return {
        "accession": acc,
        "cmap_found": True,
        "cmap_valid_shape": valid_shape,
        "zero_map": zero_map,
        "contact_density": c_upper / n_upper,
        "short_contact_density": c_short / n_short,
        "long_contact_density": c_long / n_long,
        "long_contact_fraction": c_long / c_upper if c_upper else 0.0,
    }


def summarize_numeric(values: Iterable[float]) -> dict:
    arr = pd.Series(values).dropna()
    if arr.empty:
        return {"mean": np.nan, "median": np.nan, "p10": np.nan, "p90": np.nan}
    return {
        "mean": float(arr.mean()),
        "median": float(arr.median()),
        "p10": float(arr.quantile(0.10)),
        "p90": float(arr.quantile(0.90)),
    }


def audit_split(spec: SplitSpec, cmap_dir: Path, cache_dir: Path) -> tuple[pd.DataFrame, dict]:
    ids = read_ids(spec.ids_file)
    meta = pd.read_csv(spec.meta_csv) if spec.meta_csv.exists() else pd.DataFrame()
    if "accession" in meta.columns:
        meta = meta.drop_duplicates("accession").set_index("accession")

    cache_path = cache_dir / f"contact_map_quality_{spec.name}.csv"
    if cache_path.exists():
        stats = pd.read_csv(cache_path)
    else:
        rows = [contact_stats(acc, cmap_dir) for acc in ids]
        stats = pd.DataFrame(rows)
        stats.to_csv(cache_path, index=False)

    stats.insert(0, "split", spec.name)
    if not meta.empty and "seq_len" in meta.columns:
        stats["seq_len"] = stats["accession"].map(meta["seq_len"])
    else:
        stats["seq_len"] = np.nan

    n = len(stats)
    found = int(stats["cmap_found"].sum()) if n else 0
    valid = int(stats["cmap_valid_shape"].sum()) if n else 0
    zero = int(stats["zero_map"].sum()) if n else 0
    seq = summarize_numeric(stats["seq_len"])
    dens = summarize_numeric(stats["contact_density"])
    long_frac = summarize_numeric(stats["long_contact_fraction"])

    summary = {
        "split": spec.name,
        "n": n,
        "cmap_found": found,
        "cmap_found_rate": found / n if n else np.nan,
        "cmap_valid_shape": valid,
        "zero_map": zero,
        "zero_map_rate": zero / n if n else np.nan,
        "seq_len_median": seq["median"],
        "seq_len_p10": seq["p10"],
        "seq_len_p90": seq["p90"],
        "contact_density_mean": dens["mean"],
        "contact_density_median": dens["median"],
        "long_contact_fraction_mean": long_frac["mean"],
        "long_contact_fraction_median": long_frac["median"],
    }
    return stats, summary


def write_markdown(summary: pd.DataFrame, out: Path) -> None:
    cols = [
        "split",
        "n",
        "cmap_found_rate",
        "zero_map_rate",
        "seq_len_median",
        "seq_len_p10",
        "seq_len_p90",
        "contact_density_median",
        "long_contact_fraction_median",
    ]
    table = summary[cols].copy()
    for col in ["cmap_found_rate", "zero_map_rate", "contact_density_median", "long_contact_fraction_median"]:
        table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    for col in ["seq_len_median", "seq_len_p10", "seq_len_p90"]:
        table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")

    md = [
        "# Contact-map input quality audit",
        "",
        "This audit summarizes contact-map availability and coarse contact statistics for key evaluation splits.",
        "It is intended to identify obvious structure-input artifacts such as missing maps, all-zero maps, or large density shifts.",
        "",
        table.to_markdown(index=False),
        "",
        "Notes: density uses the upper triangle of channel 0 after thresholding at 0.5; long-range contacts use |i-j| >= 12 on the 256 x 256 grid.",
    ]
    out.write_text("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmap-dir", type=Path, default=ROOT / "data/processed/contact_maps")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/audit")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        help="Optional deterministic random cap for quick audits; omit for exact split-wide statistics.",
    )
    parser.add_argument("--sample-seed", type=int, default=2026)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "contact_map_quality_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.refresh_cache:
        for path in cache_dir.glob("contact_map_quality_*.csv"):
            path.unlink()

    all_rows = []
    summaries = []
    for spec in DEFAULT_SPLITS:
        if not spec.ids_file.exists() or not spec.meta_csv.exists():
            continue
        if args.max_per_split is not None:
            ids_all = read_ids(spec.ids_file)
            if len(ids_all) > args.max_per_split:
                rng = np.random.default_rng(args.sample_seed)
                ids = list(rng.choice(ids_all, size=args.max_per_split, replace=False))
            else:
                ids = ids_all
            tmp_ids = cache_dir / f"tmp_{spec.name}_sample{len(ids)}_seed{args.sample_seed}_ids.txt"
            tmp_ids.write_text("\n".join(ids) + ("\n" if ids else ""))
            spec = SplitSpec(f"{spec.name}_sample{len(ids)}", tmp_ids, spec.meta_csv)
        rows, summary = audit_split(spec, args.cmap_dir, cache_dir)
        all_rows.append(rows)
        summaries.append(summary)

    per_sample = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary = pd.DataFrame(summaries)

    per_sample.to_csv(args.out_dir / "contact_map_quality_per_sample.csv", index=False)
    summary.to_csv(args.out_dir / "contact_map_quality_summary.csv", index=False)
    (args.out_dir / "contact_map_quality_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n"
    )
    write_markdown(summary, args.out_dir / "contact_map_quality_audit.md")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
