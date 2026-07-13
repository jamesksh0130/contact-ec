#!/usr/bin/env python3
"""Generate paper reliability audit tables.

This script is intentionally CPU-only. It checks split overlap, external-set
overlap, label distribution, multi-label statistics, and asset availability.
"""
from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "audit"


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def parse_l4_idxs(value) -> list[int]:
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    if s.startswith("["):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple, set)):
                return [int(float(x)) for x in obj if str(x).strip()]
            return [int(float(obj))]
        except Exception:
            s = s.strip("[]")
    sep = "|" if "|" in s else ","
    out = []
    for part in s.split(sep):
        part = part.strip().strip("[]")
        if part:
            out.append(int(float(part)))
    return sorted(set(out))


def sequence_hash(seq: str) -> str:
    return hashlib.sha1(str(seq).encode("utf-8")).hexdigest()


def split_df(meta: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    if not ids:
        return meta.iloc[0:0].copy()
    idx = pd.Index(ids, name="accession")
    return meta.set_index("accession").reindex(idx).reset_index()


def asset_coverage(ids: list[str], embed_dir: Path, cmap_dir: Path) -> dict:
    n = len(ids)
    emb = sum((embed_dir / f"{acc}.npy").exists() for acc in ids)
    cmap = sum((cmap_dir / f"{acc}.npy").exists() for acc in ids)
    return {
        "n": n,
        "embedding_found": emb,
        "embedding_missing": n - emb,
        "embedding_found_rate": round(emb / n, 6) if n else 0.0,
        "contact_map_found": cmap,
        "contact_map_missing": n - cmap,
        "contact_map_found_rate": round(cmap / n, 6) if n else 0.0,
    }


def label_stats(name: str, df: pd.DataFrame, train_l4_counts: Counter | None) -> dict:
    valid = df[df["m4"].fillna(0).astype(float).eq(1)].copy()
    parsed = df["l4_all_idxs"].map(parse_l4_idxs) if "l4_all_idxs" in df else pd.Series([[]] * len(df))
    cardinalities = parsed.map(len)
    all_l4 = [x for xs in parsed for x in xs]
    unique_l4 = set(all_l4)
    multi_label_n = int((cardinalities > 1).sum())

    rare_sample_n = 0
    seen_l4_n = 0
    unseen_l4_n = 0
    if train_l4_counts is not None:
        train_labels = set(train_l4_counts)
        rare_labels = {k for k, v in train_l4_counts.items() if v <= 25}
        rare_sample_n = sum(any(x in rare_labels for x in xs) for xs in parsed)
        seen_l4_n = sum(1 for x in unique_l4 if x in train_labels)
        unseen_l4_n = sum(1 for x in unique_l4 if x not in train_labels)

    return {
        "split": name,
        "n_rows": int(len(df)),
        "valid_l4_rows": int(len(valid)),
        "unique_l1": int(df["l1_idx"].dropna().nunique()) if "l1_idx" in df else 0,
        "unique_l2": int(df["l2_idx"].dropna().nunique()) if "l2_idx" in df else 0,
        "unique_l3": int(df["l3_idx"].dropna().nunique()) if "l3_idx" in df else 0,
        "unique_l4": int(len(unique_l4)),
        "avg_l4_labels_per_protein": round(float(cardinalities.mean()), 6) if len(df) else 0.0,
        "max_l4_labels_per_protein": int(cardinalities.max()) if len(df) else 0,
        "multi_l4_protein_n": multi_label_n,
        "multi_l4_protein_rate": round(multi_label_n / len(df), 6) if len(df) else 0.0,
        "partial_l1_rows": int((df.get("m1", pd.Series(dtype=float)).fillna(0).astype(float) == 0).sum()),
        "partial_l2_rows": int((df.get("m2", pd.Series(dtype=float)).fillna(0).astype(float) == 0).sum()),
        "partial_l3_rows": int((df.get("m3", pd.Series(dtype=float)).fillna(0).astype(float) == 0).sum()),
        "partial_l4_rows": int((df.get("m4", pd.Series(dtype=float)).fillna(0).astype(float) == 0).sum()),
        "rare_l4_sample_n_train_le25": int(rare_sample_n),
        "seen_l4_classes_vs_train": int(seen_l4_n),
        "unseen_l4_classes_vs_train": int(unseen_l4_n),
    }


def top_l4_rows(name: str, df: pd.DataFrame, top_k: int = 20) -> list[dict]:
    parsed = df["l4_all_idxs"].map(parse_l4_idxs) if "l4_all_idxs" in df else pd.Series([[]] * len(df))
    counts = Counter(x for xs in parsed for x in xs)
    return [{"split": name, "l4_idx": k, "count": v} for k, v in counts.most_common(top_k)]


def overlap_rows(split_ids: dict[str, list[str]]) -> list[dict]:
    names = sorted(split_ids)
    rows = []
    for i, a in enumerate(names):
        sa = set(split_ids[a])
        for b in names[i + 1 :]:
            sb = set(split_ids[b])
            inter = sa & sb
            rows.append({
                "split_a": a,
                "split_b": b,
                "n_a": len(sa),
                "n_b": len(sb),
                "overlap_n": len(inter),
                "overlap_examples": ";".join(sorted(inter)[:10]),
            })
    return rows


def sequence_overlap_rows(meta_by_split: dict[str, pd.DataFrame]) -> list[dict]:
    hashes = {}
    for name, df in meta_by_split.items():
        if "sequence" not in df:
            hashes[name] = set()
        else:
            hashes[name] = set(df["sequence"].dropna().map(sequence_hash))
    rows = []
    names = sorted(hashes)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rows.append({
                "split_a": a,
                "split_b": b,
                "exact_sequence_overlap_n": len(hashes[a] & hashes[b]),
            })
    return rows


def audit_dataset(tag: str, meta_csv: Path, splits_dir: Path, split_names: list[str],
                  embed_dir: Path, cmap_dir: Path) -> dict:
    meta = pd.read_csv(meta_csv)
    split_ids = {name: read_ids(splits_dir / f"{name}_ids.txt") for name in split_names}
    meta_by_split = {name: split_df(meta, ids) for name, ids in split_ids.items()}

    train_parsed = meta_by_split.get("train", meta.iloc[0:0])["l4_all_idxs"].map(parse_l4_idxs)
    train_counts = Counter(x for xs in train_parsed for x in xs)

    split_stats = []
    asset_rows = []
    top_rows = []
    for name, df in meta_by_split.items():
        split_stats.append(label_stats(name, df, train_counts))
        asset_rows.append({"split": name, **asset_coverage(split_ids[name], embed_dir, cmap_dir)})
        top_rows.extend(top_l4_rows(name, df))

    prefix = OUT / tag
    pd.DataFrame(split_stats).to_csv(prefix.with_name(prefix.name + "_split_label_stats.csv"), index=False)
    pd.DataFrame(asset_rows).to_csv(prefix.with_name(prefix.name + "_asset_coverage.csv"), index=False)
    pd.DataFrame(top_rows).to_csv(prefix.with_name(prefix.name + "_top_l4.csv"), index=False)
    pd.DataFrame(overlap_rows(split_ids)).to_csv(prefix.with_name(prefix.name + "_accession_overlap.csv"), index=False)
    pd.DataFrame(sequence_overlap_rows(meta_by_split)).to_csv(prefix.with_name(prefix.name + "_exact_sequence_overlap.csv"), index=False)

    return {
        "tag": tag,
        "meta_csv": str(meta_csv.relative_to(ROOT)),
        "splits_dir": str(splits_dir.relative_to(ROOT)),
        "n_meta_rows": int(len(meta)),
        "split_sizes": {k: len(v) for k, v in split_ids.items()},
        "train_unique_l4": int(len(train_counts)),
        "train_rare_l4_classes_le25": int(sum(v <= 25 for v in train_counts.values())),
    }


def external_overlap_main() -> None:
    meta = pd.read_csv(ROOT / "data" / "processed" / "dataset_meta.csv")
    train_ids = set(read_ids(ROOT / "data" / "splits" / "train_ids.txt"))
    full_ids = set(meta["accession"].astype(str))
    rows = []
    for name, p in [
        ("new392", ROOT / "data" / "processed" / "new392_meta.csv"),
        ("price149", ROOT / "data" / "processed" / "price149_meta.csv"),
    ]:
        if not p.exists():
            continue
        ext = pd.read_csv(p)
        ids = set(ext["accession"].astype(str))
        seq_hashes_ext = set(ext["sequence"].dropna().map(sequence_hash))
        seq_hashes_train = set(meta[meta["accession"].isin(train_ids)]["sequence"].dropna().map(sequence_hash))
        rows.append({
            "external_set": name,
            "n": len(ids),
            "accession_overlap_with_train_n": len(ids & train_ids),
            "accession_overlap_with_full_meta_n": len(ids & full_ids),
            "exact_sequence_overlap_with_train_n": len(seq_hashes_ext & seq_hashes_train),
            "contact_map_found_n": sum((ROOT / "data" / "processed" / "contact_maps" / f"{x}.npy").exists() for x in ids),
            "embedding_found_n": sum((ROOT / "data" / "processed" / "embeddings" / f"{x}.npy").exists() for x in ids),
            "overlap_examples_train": ";".join(sorted(ids & train_ids)[:20]),
        })
    pd.DataFrame(rows).to_csv(OUT / "main_external_overlap.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []

    summaries.append(audit_dataset(
        tag="main",
        meta_csv=ROOT / "data" / "processed" / "dataset_meta.csv",
        splits_dir=ROOT / "data" / "splits",
        split_names=["train", "val", "test", "cluster_train", "cluster_val", "cluster_test", "new392", "price149"],
        embed_dir=ROOT / "data" / "processed" / "embeddings",
        cmap_dir=ROOT / "data" / "processed" / "contact_maps",
    ))

    if (ROOT / "data" / "ecbench" / "processed" / "train_meta.csv").exists():
        ec_train = pd.read_csv(ROOT / "data" / "ecbench" / "processed" / "train_meta.csv")
        ec_test = pd.read_csv(ROOT / "data" / "ecbench" / "processed" / "test_meta.csv")
        ec_price = pd.read_csv(ROOT / "data" / "ecbench" / "processed" / "price149_meta.csv")
        ec_meta = pd.concat([ec_train, ec_test, ec_price], ignore_index=True).drop_duplicates("accession")
        ec_meta_path = OUT / "_ecbench_combined_meta.csv"
        ec_meta.to_csv(ec_meta_path, index=False)
        summaries.append(audit_dataset(
            tag="ecbench",
            meta_csv=ec_meta_path,
            splits_dir=ROOT / "data" / "ecbench" / "splits",
            split_names=["train", "val", "val_easy", "val_hard", "test", "price149"],
            embed_dir=ROOT / "data" / "processed" / "embeddings",
            cmap_dir=ROOT / "data" / "processed" / "contact_maps",
        ))

    external_overlap_main()
    (OUT / "audit_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote audit outputs to {OUT}")


if __name__ == "__main__":
    main()
