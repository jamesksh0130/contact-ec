#!/usr/bin/env python3
"""Build a Foldseek-based structure-disjoint split.

The script clusters available AlphaFold/PDB structures for a selected metadata
table and writes split ID files in the same style as the existing sequence
cluster split.  It also writes an audit report with cluster sizes and verifies
that no Foldseek cluster is shared across train/validation/test.

Typical use:

  python scripts/foldseek_structure_split.py \
    --meta data/ecbench/processed/train_meta.csv \
    --prefix foldseek \
    --min-seq-id 0.30 \
    --coverage 0.80

This is a split-construction utility.  A full paper claim still requires
retraining on foldseek_train and evaluation on foldseek_test or on a temporally
held-out fold-disjoint subset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PDB_DIR = ROOT / "data" / "raw" / "pdb"
DEFAULT_SPLIT_DIR = ROOT / "data" / "ecbench" / "splits"
DEFAULT_OUT_DIR = ROOT / "outputs" / "audit"
DEFAULT_TMP_DIR = ROOT / "tmp_foldseek_split"
DEFAULT_FOLDSEEK_BIN = os.environ.get("FOLDSEEK_BIN", "foldseek")


def run(cmd: list[str], *, quiet: bool = False) -> None:
    if not quiet:
        print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_tsv(path: Path) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rep, member = line.rstrip("\n").split("\t")[:2]
            clusters[Path(rep).stem].append(Path(member).stem)
    return dict(clusters)


def assign_clusters_by_count(clusters: list[list[str]], seed: int, val_ratio: float, test_ratio: float) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    clusters = [list(c) for c in clusters]
    rng.shuffle(clusters)
    n_clusters = len(clusters)
    n_val = max(1, int(n_clusters * val_ratio))
    n_test = max(1, int(n_clusters * test_ratio))
    val_clusters = clusters[:n_val]
    test_clusters = clusters[n_val : n_val + n_test]
    train_clusters = clusters[n_val + n_test :]
    train_ids = [acc for c in train_clusters for acc in c]
    val_ids = [acc for c in val_clusters for acc in c]
    test_ids = [acc for c in test_clusters for acc in c]
    return train_ids, val_ids, test_ids


def assign_clusters_by_proteins(clusters: list[list[str]], seed: int, val_ratio: float, test_ratio: float) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = [list(c) for c in clusters]
    rng.shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)

    total = sum(len(c) for c in shuffled)
    targets = {
        "test": int(round(total * test_ratio)),
        "val": int(round(total * val_ratio)),
    }
    buckets: dict[str, list[list[str]]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for cluster in shuffled:
        candidates = []
        for split in ("test", "val"):
            before = abs(targets[split] - counts[split])
            after = abs(targets[split] - (counts[split] + len(cluster)))
            candidates.append((after - before, counts[split], split))
        candidates.sort()
        best_delta, _, best_split = candidates[0]
        if counts[best_split] < targets[best_split] or best_delta <= 0:
            split = best_split
        else:
            split = "train"
        buckets[split].append(cluster)
        counts[split] += len(cluster)

    train_ids = [acc for c in buckets["train"] for acc in c]
    val_ids = [acc for c in buckets["val"] for acc in c]
    test_ids = [acc for c in buckets["test"] for acc in c]
    return train_ids, val_ids, test_ids


def write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + ("\n" if ids else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=str(ROOT / "data" / "ecbench" / "processed" / "train_meta.csv"))
    parser.add_argument("--pdb-dir", default=str(PDB_DIR))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tmp-dir", default=str(DEFAULT_TMP_DIR))
    parser.add_argument("--prefix", default="foldseek")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--assignment-mode", choices=["protein_balanced", "cluster_count"], default="protein_balanced")
    parser.add_argument("--min-seq-id", type=float, default=0.30)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--tmscore-threshold", type=float, default=0.50)
    parser.add_argument("--lddt-threshold", type=float, default=0.00)
    parser.add_argument("--alignment-type", type=int, default=1, help="Foldseek alignment type: 1=TM alignment, 2=3Di+AA")
    parser.add_argument("--cluster-mode", type=int, default=1, help="Foldseek cluster mode: 1=connected component")
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--foldseek-verbosity", type=int, default=1)
    parser.add_argument("--foldseek-bin", default=DEFAULT_FOLDSEEK_BIN)
    parser.add_argument("--reuse-tsv", default=None, help="Existing Foldseek cluster TSV to reuse")
    parser.add_argument("--keep-tmp", action="store_true")
    args = parser.parse_args()

    meta_path = Path(args.meta)
    pdb_dir = Path(args.pdb_dir)
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    tmp_dir = Path(args.tmp_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(meta_path)
    meta = meta[meta["accession"].notna()].copy()
    valid_accs = set(meta["accession"].astype(str))

    structure_dir = tmp_dir / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)

    available = []
    missing = []
    for acc in sorted(valid_accs):
        src = pdb_dir / f"{acc}.pdb"
        if src.exists():
            available.append(acc)
            dst = structure_dir / f"{acc}.pdb"
            if dst.is_symlink() and not dst.exists():
                dst.unlink()
            if not dst.exists():
                dst.symlink_to(src.resolve())
        else:
            missing.append(acc)

    if not available:
        raise RuntimeError(f"No PDB files found for metadata accessions in {pdb_dir}")
    print(f"Metadata proteins: {len(valid_accs):,}")
    print(f"Structures available: {len(available):,}")
    print(f"Structures missing: {len(missing):,}")

    if args.reuse_tsv:
        tsv_path = Path(args.reuse_tsv)
    else:
        db_path = tmp_dir / "struct_db"
        cluster_path = tmp_dir / "struct_cluster"
        foldseek_tmp = tmp_dir / "foldseek_tmp"
        tsv_path = tmp_dir / f"{args.prefix}_cluster.tsv"

        run(
            [
                args.foldseek_bin,
                "createdb",
                str(structure_dir),
                str(db_path),
                "--threads",
                str(args.threads),
                "-v",
                str(args.foldseek_verbosity),
            ]
        )
        run(
            [
                args.foldseek_bin,
                "cluster",
                str(db_path),
                str(cluster_path),
                str(foldseek_tmp),
                "--min-seq-id",
                str(args.min_seq_id),
                "-c",
                str(args.coverage),
                "--cov-mode",
                "0",
                "--tmscore-threshold",
                str(args.tmscore_threshold),
                "--lddt-threshold",
                str(args.lddt_threshold),
                "--alignment-type",
                str(args.alignment_type),
                "--cluster-mode",
                str(args.cluster_mode),
                "-s",
                str(args.sensitivity),
                "--threads",
                str(args.threads),
                "-v",
                str(args.foldseek_verbosity),
            ]
        )
        run(
            [
                args.foldseek_bin,
                "createtsv",
                str(db_path),
                str(db_path),
                str(cluster_path),
                str(tsv_path),
                "-v",
                str(args.foldseek_verbosity),
            ]
        )

    saved_tsv = out_dir / f"{args.prefix}_cluster.tsv"
    if Path(tsv_path).resolve() != saved_tsv.resolve():
        shutil.copy2(tsv_path, saved_tsv)

    cluster_map = parse_tsv(tsv_path)
    clusters = [sorted([m for m in members if m in valid_accs]) for members in cluster_map.values()]
    clusters = [c for c in clusters if c]
    clustered = {acc for c in clusters for acc in c}
    unclustered = sorted(set(available) - clustered)
    clusters.extend([[acc] for acc in unclustered])

    if args.assignment_mode == "cluster_count":
        train_ids, val_ids, test_ids = assign_clusters_by_count(clusters, args.seed, args.val_ratio, args.test_ratio)
    else:
        train_ids, val_ids, test_ids = assign_clusters_by_proteins(clusters, args.seed, args.val_ratio, args.test_ratio)

    # Coverage sanity.
    all_split = set(train_ids) | set(val_ids) | set(test_ids)
    overlap_tv = set(train_ids) & set(val_ids)
    overlap_tt = set(train_ids) & set(test_ids)
    overlap_vt = set(val_ids) & set(test_ids)
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError("Split ID overlap detected")

    write_ids(split_dir / f"{args.prefix}_train_ids.txt", train_ids)
    write_ids(split_dir / f"{args.prefix}_val_ids.txt", val_ids)
    write_ids(split_dir / f"{args.prefix}_test_ids.txt", test_ids)

    cluster_rows = []
    membership = {}
    for i, members in enumerate(clusters):
        split = "train"
        if any(m in val_ids for m in members):
            split = "val"
        if any(m in test_ids for m in members):
            split = "test"
        for acc in members:
            membership[acc] = {"cluster_id": i, "split": split, "cluster_size": len(members)}
        cluster_rows.append({"cluster_id": i, "size": len(members), "split": split, "representative": members[0]})

    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_csv(out_dir / f"{args.prefix}_cluster_summary.csv", index=False)
    pd.DataFrame(
        [{"accession": acc, **membership[acc]} for acc in sorted(membership)]
    ).to_csv(out_dir / f"{args.prefix}_membership.csv", index=False)
    pd.DataFrame({"accession": missing}).to_csv(out_dir / f"{args.prefix}_missing_structures.csv", index=False)

    sizes = cluster_df["size"].to_numpy()
    audit = {
        "meta": str(meta_path.relative_to(ROOT) if meta_path.is_relative_to(ROOT) else meta_path),
        "pdb_dir": str(pdb_dir),
        "prefix": args.prefix,
        "foldseek_min_seq_id": args.min_seq_id,
        "foldseek_coverage": args.coverage,
        "foldseek_tmscore_threshold": args.tmscore_threshold,
        "foldseek_lddt_threshold": args.lddt_threshold,
        "foldseek_alignment_type": args.alignment_type,
        "foldseek_cluster_mode": args.cluster_mode,
        "foldseek_sensitivity": args.sensitivity,
        "foldseek_verbosity": args.foldseek_verbosity,
        "assignment_mode": args.assignment_mode,
        "cluster_tsv": str(saved_tsv.relative_to(ROOT) if saved_tsv.is_relative_to(ROOT) else saved_tsv),
        "seed": args.seed,
        "metadata_proteins": int(len(valid_accs)),
        "structures_available": int(len(available)),
        "structures_missing": int(len(missing)),
        "clusters": int(len(clusters)),
        "cluster_size_min": int(sizes.min()),
        "cluster_size_median": float(np.median(sizes)),
        "cluster_size_mean": float(sizes.mean()),
        "cluster_size_max": int(sizes.max()),
        "train_proteins": int(len(train_ids)),
        "val_proteins": int(len(val_ids)),
        "test_proteins": int(len(test_ids)),
        "train_clusters": int((cluster_df["split"] == "train").sum()),
        "val_clusters": int((cluster_df["split"] == "val").sum()),
        "test_clusters": int((cluster_df["split"] == "test").sum()),
        "split_protein_total": int(len(all_split)),
        "id_overlap_train_val": int(len(overlap_tv)),
        "id_overlap_train_test": int(len(overlap_tt)),
        "id_overlap_val_test": int(len(overlap_vt)),
    }
    (out_dir / f"{args.prefix}_split_audit.json").write_text(json.dumps(audit, indent=2))

    lines = [
        "# Foldseek structure-disjoint split audit",
        "",
        pd.DataFrame([audit]).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Split files are assigned at the Foldseek cluster level, so no cluster is shared across train/validation/test.",
        "- This creates a structure-disjoint training protocol for future retraining.",
        "- It does not by itself evaluate a model; report performance only after training on the generated train split.",
    ]
    (out_dir / f"{args.prefix}_split_audit.md").write_text("\n".join(lines) + "\n")

    print(json.dumps(audit, indent=2))
    if not args.keep_tmp:
        # Preserve reusable TSV and summaries in outputs/audit, remove bulky DB files.
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        raise
