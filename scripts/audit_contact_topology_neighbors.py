#!/usr/bin/env python3
"""Nearest-neighbor audit using raw contact-map topology fingerprints.

This audit is deliberately independent of the learned contact-pair ESM tensor
used in FusionV3.  It summarizes each 256x256 contact map with a coarse
downsampled topology grid, sequence-separation contact densities, and node-degree
statistics, then asks whether the complete temporal proteins have close
neighbours in the SP-2018 training set.

It is not a Foldseek/TM-align fold-disjoint benchmark.  Instead, it is a
structure-input relatedness audit that is cheap to reproduce from the released
contact maps and highlights whether temporal performance should be interpreted
conservatively.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CMAP_DIR = ROOT / "data" / "processed" / "contact_maps"
OUT_DIR = ROOT / "outputs" / "audit"
FIG_DIR = ROOT / "outputs" / "figures"

EC_RE = re.compile(r"\d+\.\d+\.\d+\.\d+")


def parse_ecs(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return EC_RE.findall(str(value))


def ec_prefixes(ecs: list[str], depth: int) -> set[str]:
    out = set()
    for ec in ecs:
        parts = ec.split(".")
        if len(parts) >= depth:
            out.add(".".join(parts[:depth]))
    return out


def load_contact_map(acc: str) -> np.ndarray | None:
    path = CMAP_DIR / f"{acc}.npy"
    if not path.exists():
        return None
    arr = np.load(path, mmap_mode="r")
    if arr.shape != (256, 256):
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all() or float(arr.max()) <= 0:
        return None
    # Enforce symmetry softly because contact maps are expected to be symmetric,
    # while some preprocessing paths can introduce tiny interpolation asymmetry.
    arr = 0.5 * (arr + arr.T)
    return arr


def topology_fingerprint(acc: str) -> np.ndarray | None:
    cm = load_contact_map(acc)
    if cm is None:
        return None

    n = cm.shape[0]
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    valid = sep > 2
    binary = (cm > 0.5).astype(np.float32)

    # Coarse topology grid: 256x256 -> 32x32 block means.  This retains global
    # fold/contact layout while keeping nearest-neighbour search lightweight.
    coarse = cm.reshape(32, 8, 32, 8).mean(axis=(1, 3)).astype(np.float32)
    coarse = coarse.reshape(-1)

    # Sequence-separation contact densities.
    sep_bins = [(3, 6), (6, 12), (12, 24), (24, 48), (48, 96), (96, 160), (160, 256)]
    sep_feats = []
    for lo, hi in sep_bins:
        m = (sep >= lo) & (sep < hi)
        denom = int(m.sum())
        sep_feats.append(float(binary[m].sum() / denom) if denom else 0.0)

    # Degree distribution summarizes local/topological contact concentration.
    deg = (binary * valid.astype(np.float32)).sum(axis=1) / max(1, n - 3)
    quantiles = np.quantile(deg, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    deg_feats = np.array(
        [
            float(binary[valid].mean()),
            float(deg.mean()),
            float(deg.std()),
            *[float(q) for q in quantiles],
        ],
        dtype=np.float32,
    )

    vec = np.concatenate([coarse, np.array(sep_feats, dtype=np.float32), deg_feats])
    norm = float(np.linalg.norm(vec))
    if norm == 0 or not np.isfinite(norm):
        return None
    return (vec / norm).astype(np.float32)


def metadata_with_ec_sets(path: Path, accession_col: str = "accession") -> pd.DataFrame:
    df = pd.read_csv(path)
    if accession_col != "accession":
        df = df.rename(columns={accession_col: "accession"})
    df = df[df["accession"].notna() & df["ec_raw"].notna()].copy()
    df["ec_list"] = df["ec_raw"].map(parse_ecs)
    df = df[df["ec_list"].map(bool)].copy()
    for depth in [1, 2, 3, 4]:
        df[f"l{depth}_set"] = df["ec_list"].map(lambda xs, d=depth: ec_prefixes(xs, d))
    return df


def overlap(a: set[str], b: set[str]) -> bool:
    return bool(a & b)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train = metadata_with_ec_sets(ROOT / "data" / "ecbench" / "processed" / "train_meta.csv")
    temporal = metadata_with_ec_sets(ROOT / "data" / "ecbench" / "processed" / "test_meta_full.csv")

    q_vecs: list[np.ndarray] = []
    q_rows: list[dict] = []
    for _, row in temporal.iterrows():
        vec = topology_fingerprint(str(row["accession"]))
        if vec is None:
            continue
        q_vecs.append(vec)
        q_rows.append(row.to_dict())
    if not q_vecs:
        raise RuntimeError("No temporal contact maps available for topology audit")

    q_mat = np.vstack(q_vecs)
    q_df = pd.DataFrame(q_rows)

    best = []
    for _, row in q_df.iterrows():
        best.append(
            {
                "query_accession": row["accession"],
                "query_ec": ";".join(row["ec_list"]),
                "nearest_train_accession": None,
                "nearest_train_ec": None,
                "nearest_cosine": -2.0,
                "same_l1": False,
                "same_l2": False,
                "same_l3": False,
                "same_l4": False,
            }
        )

    chunk_vecs: list[np.ndarray] = []
    chunk_rows: list[dict] = []
    processed = 0
    usable = 0
    chunk_size = 2048

    train_rows = train.to_dict(orient="records")

    def flush() -> None:
        nonlocal chunk_vecs, chunk_rows, usable
        if not chunk_vecs:
            return
        mat = np.vstack(chunk_vecs)
        sims = q_mat @ mat.T
        argmax = sims.argmax(axis=1)
        maxval = sims[np.arange(sims.shape[0]), argmax]
        for qi, sim in enumerate(maxval):
            if float(sim) > best[qi]["nearest_cosine"]:
                tr = chunk_rows[int(argmax[qi])]
                q = q_df.iloc[qi]
                best[qi].update(
                    {
                        "nearest_train_accession": tr["accession"],
                        "nearest_train_ec": ";".join(tr["ec_list"]),
                        "nearest_cosine": round(float(sim), 6),
                        "same_l1": overlap(q["l1_set"], tr["l1_set"]),
                        "same_l2": overlap(q["l2_set"], tr["l2_set"]),
                        "same_l3": overlap(q["l3_set"], tr["l3_set"]),
                        "same_l4": overlap(q["l4_set"], tr["l4_set"]),
                    }
                )
        usable += len(chunk_rows)
        chunk_vecs = []
        chunk_rows = []

    for tr in train_rows:
        processed += 1
        vec = topology_fingerprint(str(tr["accession"]))
        if vec is None:
            continue
        chunk_vecs.append(vec)
        chunk_rows.append(tr)
        if len(chunk_vecs) >= chunk_size:
            flush()
            print(f"processed={processed:,} usable={usable:,}", end="\r")
            sys.stdout.flush()
    flush()
    print(f"\nprocessed={processed:,} usable={usable:,} temporal={len(best):,}")

    nn = pd.DataFrame(best)
    nn.to_csv(OUT_DIR / "contact_topology_neighbor_audit.csv", index=False)

    bins = [-1, 0.80, 0.90, 0.95, 0.98, 1.0]
    labels = ["<=0.80", "0.80-0.90", "0.90-0.95", "0.95-0.98", ">0.98"]
    nn["cosine_bin"] = pd.cut(nn["nearest_cosine"], bins=bins, labels=labels, include_lowest=True)
    summary = (
        nn.groupby("cosine_bin", observed=False)
        .agg(
            n=("query_accession", "count"),
            mean_cosine=("nearest_cosine", "mean"),
            same_l1_rate=("same_l1", "mean"),
            same_l2_rate=("same_l2", "mean"),
            same_l3_rate=("same_l3", "mean"),
            same_l4_rate=("same_l4", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "contact_topology_neighbor_summary.csv", index=False)

    overall = {
        "temporal_known_with_contact_maps": int(len(nn)),
        "train_processed": int(processed),
        "train_with_contact_maps": int(usable),
        "feature_dimension": int(q_mat.shape[1]),
        "median_nearest_cosine": float(nn["nearest_cosine"].median()),
        "mean_nearest_cosine": float(nn["nearest_cosine"].mean()),
        "n_cosine_gt_0_90": int((nn["nearest_cosine"] > 0.90).sum()),
        "n_cosine_gt_0_95": int((nn["nearest_cosine"] > 0.95).sum()),
        "n_cosine_gt_0_98": int((nn["nearest_cosine"] > 0.98).sum()),
        "same_l1_rate": float(nn["same_l1"].mean()),
        "same_l2_rate": float(nn["same_l2"].mean()),
        "same_l3_rate": float(nn["same_l3"].mean()),
        "same_l4_rate": float(nn["same_l4"].mean()),
    }
    (OUT_DIR / "contact_topology_neighbor_audit.json").write_text(
        json.dumps({"overall": overall, "bins": summary.to_dict(orient="records")}, indent=2)
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].hist(nn["nearest_cosine"], bins=20, color="#386641", edgecolor="white")
    axes[0].axvline(overall["median_nearest_cosine"], color="#bc4749", linestyle="--", label="median")
    axes[0].set_xlabel("Nearest train contact-topology cosine")
    axes[0].set_ylabel("Temporal proteins")
    axes[0].legend(frameon=False)

    rates = [overall["same_l1_rate"], overall["same_l2_rate"], overall["same_l3_rate"], overall["same_l4_rate"]]
    axes[1].bar(["L1", "L2", "L3", "L4"], rates, color="#6a994e")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Nearest-neighbor EC overlap rate")
    axes[1].set_xlabel("EC hierarchy level")
    for i, v in enumerate(rates):
        axes[1].text(i, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Raw contact-map topology nearest-neighbor audit")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "contact_topology_neighbor_audit.png", dpi=300)
    plt.close(fig)

    lines = [
        "# Raw contact-map topology nearest-neighbor audit",
        "",
        "This is not a Foldseek/TM-align fold-disjoint benchmark. It uses hand-crafted fingerprints derived from the raw 256x256 contact maps: a 32x32 coarse contact grid, sequence-separation contact densities, and node-degree statistics.",
        "",
        "## Overall",
        "",
        pd.DataFrame([overall]).to_markdown(index=False),
        "",
        "## Bins",
        "",
        summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- This audit is independent of the learned contact-pair ESM representation.",
        "- High nearest-neighbor topology similarity indicates that temporal performance should still be interpreted conservatively.",
        "- EC overlap rates among nearest topology neighbours quantify whether contact-map proximity also recovers enzyme hierarchy.",
        "- A publication-grade fold-disjoint claim still requires Foldseek/TM-align or CATH/SCOP clustering followed by retraining/evaluation.",
    ]
    (OUT_DIR / "contact_topology_neighbor_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
