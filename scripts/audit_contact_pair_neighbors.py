#!/usr/bin/env python3
"""Nearest-neighbor audit using contact-pair representations.

This is not a fold-disjoint test. It asks whether temporal proteins have close
neighbours in the SP-2018 training set under the already extracted contact-pair
ESM representation used by the FusionV3 experiments. The result is a structural
relatedness warning: high nearest-neighbour similarity would mean that temporal
performance should not be interpreted as fold-disjoint generalization.
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
PAIR_DIR = ROOT / "data" / "processed" / "contact_pair_embs"
OUT_DIR = ROOT / "outputs" / "audit"
FIG_DIR = ROOT / "outputs" / "figures"

EC_RE = re.compile(r"\d+\.\d+\.\d+\.\d+")


def parse_ecs(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return EC_RE.findall(str(value))


def pair_vector(acc: str) -> np.ndarray | None:
    path = PAIR_DIR / f"{acc}.npy"
    if not path.exists():
        return None
    arr = np.load(path, mmap_mode="r").astype(np.float32)
    if arr.size == 0:
        return None
    flat = arr.reshape(-1, arr.shape[-1])
    # Mean and standard deviation retain coarse contact-conditioned residue context
    # without storing the full 32x2x1280 tensor for all training proteins.
    vec = np.concatenate([flat.mean(axis=0), flat.std(axis=0)], axis=0)
    norm = float(np.linalg.norm(vec))
    if norm == 0 or not np.isfinite(norm):
        return None
    return (vec / norm).astype(np.float32)


def build_temporal_known(train_l4: set[str]) -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "data" / "ecbench" / "raw" / "test_ec.csv")
    rows = []
    for _, row in raw.iterrows():
        ecs = [ec for ec in parse_ecs(row["ec_number"]) if ec in train_l4]
        if not ecs:
            continue
        rows.append(
            {
                "accession": row["id"],
                "ec_raw": ";".join(ecs),
                "seq_len": len(str(row["seq"])),
                "l1": ecs[0].split(".")[0],
                "l2": ".".join(ecs[0].split(".")[:2]),
                "l3": ".".join(ecs[0].split(".")[:3]),
                "l4": ecs[0],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(ROOT / "data" / "ecbench" / "processed" / "train_meta.csv")
    train = train[train["accession"].notna() & train["ec_raw"].notna()].copy()
    train["l4_list"] = train["ec_raw"].map(parse_ecs)
    train["l4"] = train["l4_list"].map(lambda xs: xs[0] if xs else None)
    train = train[train["l4"].notna()].copy()
    train["l1"] = train["l4"].str.split(".").str[0]
    train["l2"] = train["l4"].str.split(".").str[:2].str.join(".")
    train["l3"] = train["l4"].str.split(".").str[:3].str.join(".")
    train_l4 = set(train["l4"])

    temporal = build_temporal_known(train_l4)
    temporal["pair_emb_exists"] = temporal["accession"].map(lambda x: (PAIR_DIR / f"{x}.npy").exists())
    temporal = temporal[temporal["pair_emb_exists"]].copy()

    q_vecs = []
    q_rows = []
    for _, row in temporal.iterrows():
        vec = pair_vector(row["accession"])
        if vec is not None:
            q_vecs.append(vec)
            q_rows.append(row.to_dict())
    if not q_vecs:
        raise RuntimeError("No temporal contact-pair embeddings found")
    q_mat = np.vstack(q_vecs)
    q_df = pd.DataFrame(q_rows)

    best = [
        {
            "query_accession": row["accession"],
            "query_ec": row["ec_raw"],
            "query_l1": row["l1"],
            "query_l2": row["l2"],
            "query_l3": row["l3"],
            "query_l4": row["l4"],
            "nearest_train_accession": None,
            "nearest_train_ec": None,
            "nearest_cosine": -2.0,
            "same_l1": False,
            "same_l2": False,
            "same_l3": False,
            "same_l4": False,
        }
        for _, row in q_df.iterrows()
    ]

    train_rows = train.to_dict(orient="records")
    chunk_vecs = []
    chunk_rows = []
    processed = 0
    usable = 0
    chunk_size = 4096

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
                best[qi].update(
                    {
                        "nearest_train_accession": tr["accession"],
                        "nearest_train_ec": tr["ec_raw"],
                        "nearest_cosine": round(float(sim), 6),
                        "same_l1": q_df.iloc[qi]["l1"] == tr["l1"],
                        "same_l2": q_df.iloc[qi]["l2"] == tr["l2"],
                        "same_l3": q_df.iloc[qi]["l3"] == tr["l3"],
                        "same_l4": q_df.iloc[qi]["l4"] == tr["l4"],
                    }
                )
        usable += len(chunk_rows)
        chunk_vecs = []
        chunk_rows = []

    for tr in train_rows:
        processed += 1
        vec = pair_vector(str(tr["accession"]))
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
    nn.to_csv(OUT_DIR / "contact_pair_neighbor_audit.csv", index=False)

    bins = [-1, 0.4, 0.6, 0.8, 0.9, 1.0]
    labels = ["<=0.40", "0.40-0.60", "0.60-0.80", "0.80-0.90", ">0.90"]
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
    summary.to_csv(OUT_DIR / "contact_pair_neighbor_summary.csv", index=False)

    overall = {
        "temporal_known_with_pair_embeddings": int(len(nn)),
        "train_processed": int(processed),
        "train_with_pair_embeddings": int(usable),
        "median_nearest_cosine": float(nn["nearest_cosine"].median()),
        "mean_nearest_cosine": float(nn["nearest_cosine"].mean()),
        "n_cosine_gt_0_80": int((nn["nearest_cosine"] > 0.80).sum()),
        "n_cosine_gt_0_90": int((nn["nearest_cosine"] > 0.90).sum()),
        "same_l1_rate": float(nn["same_l1"].mean()),
        "same_l2_rate": float(nn["same_l2"].mean()),
        "same_l3_rate": float(nn["same_l3"].mean()),
        "same_l4_rate": float(nn["same_l4"].mean()),
    }
    (OUT_DIR / "contact_pair_neighbor_audit.json").write_text(
        json.dumps({"overall": overall, "bins": summary.to_dict(orient="records")}, indent=2)
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(nn["nearest_cosine"], bins=20, color="#32746d", edgecolor="white")
    ax.axvline(overall["median_nearest_cosine"], color="#b85c38", linestyle="--", label="median")
    ax.set_xlabel("Nearest train contact-pair cosine similarity")
    ax.set_ylabel("Temporal proteins")
    ax.set_title("Contact-pair nearest-neighbor audit")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "contact_pair_neighbor_hist.png", dpi=300)
    plt.close(fig)

    lines = [
        "# Contact-pair nearest-neighbor audit",
        "",
        "This is not a Foldseek/TM-align fold-disjoint benchmark. It uses the existing contact-pair ESM representation as a model-proximal relatedness audit.",
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
        "- High nearest-neighbor similarity indicates that temporal performance should not be interpreted as fold-disjoint structural generalization.",
        "- Same-EC rates among nearest neighbours quantify how often contact-conditioned representation proximity also recovers EC hierarchy.",
        "- A true fold-disjoint claim still requires Foldseek/TM-align or CATH/SCOP clustering and retraining/evaluation on the resulting split.",
    ]
    (OUT_DIR / "contact_pair_neighbor_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
