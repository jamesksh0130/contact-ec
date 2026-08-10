#!/usr/bin/env python3
"""Audit the Price-149 external benchmark failure mode.

This script does not run model inference. It summarizes already generated
metadata and result JSON files so the manuscript can report why Price-149 is a
stress test rather than only listing a low Level-4 score.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "audit"
FIG_DIR = ROOT / "outputs" / "figures"


EC_RE = re.compile(r"\d+\.\d+\.\d+\.\d+")


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def parse_ecs(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return EC_RE.findall(str(value))


def existing(path: Path) -> bool:
    return path.exists()


def cmap_stats(path: Path) -> tuple[bool, float | None, int | None]:
    if not path.exists():
        return False, None, None
    arr = np.load(path, mmap_mode="r")
    total = float(np.asarray(arr).sum())
    nonzero = int(np.count_nonzero(np.asarray(arr)))
    return total > 0, total, nonzero


def extract_price_metrics() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    b1 = read_json(ROOT / "outputs" / "results" / "ecbench_eval_ecbench_b1_best.json")
    b3 = read_json(ROOT / "outputs" / "results" / "ecbench_eval_ecbench_b3_phase1_best.json")
    b4 = read_json(ROOT / "outputs" / "results" / "ecbench_eval_ecbench_b4_flatfc_best.json")
    esmfold = read_json(ROOT / "outputs" / "results" / "price149_esmfold_eval.json")
    hier = read_json(ROOT / "outputs" / "results" / "fusion_v2_price149_hier_results.json")

    for model_name, source in [
        ("B1 (ESM-2, SP-2018)", b1["results"]["price149"]),
        ("B3 (Contact, SP-2018)", b3["results"]["price149"]),
        ("Contact-EC flat (SP-2018)", b4["results"]["price149"]),
    ]:
        rows.append(
            {
                "model": model_name,
                "setting": "canonical flat Level-4",
                "n": source["n_samples"],
                "level": "L4",
                "micro_f1": source["micro_f1"],
                "weighted_f1": source["weighted_f1"],
                "precision": source["precision"],
                "recall": source["recall"],
            }
        )

    rows.append(
        {
            "model": "Contact-EC ESMFold-map",
            "setting": "ESMFold contact maps",
            "n": esmfold["n"],
            "level": "L4",
            "micro_f1": esmfold["results"]["Contact-EC"]["micro_f1"],
            "weighted_f1": esmfold["results"]["Contact-EC"]["weighted_f1"],
            "precision": None,
            "recall": None,
        }
    )

    for level in ["level1", "level2", "level3", "level4"]:
        source = hier[level]
        rows.append(
            {
                "model": "Contact-EC hierarchical",
                "setting": "hierarchical diagnostic",
                "n": source["n_samples"],
                "level": level.replace("level", "L"),
                "micro_f1": source["micro_f1"],
                "weighted_f1": source.get("weighted_f1"),
                "precision": source.get("precision"),
                "recall": source.get("recall"),
            }
        )

    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(ROOT / "data" / "new392" / "price149_labels.csv", sep="\t")
    meta = pd.read_csv(ROOT / "data" / "ecbench" / "processed" / "price149_meta.csv")
    with (ROOT / "data" / "ecbench" / "label_encoders.pkl").open("rb") as f:
        encoders = pickle.load(f)
    l4_vocab = set(encoders["level4"].classes_)

    raw["ecs"] = raw["EC number"].map(parse_ecs)
    raw["n_ec"] = raw["ecs"].map(len)
    raw["all_ec_in_l4_vocab"] = raw["ecs"].map(lambda xs: bool(xs) and all(x in l4_vocab for x in xs))
    raw["any_ec_in_l4_vocab"] = raw["ecs"].map(lambda xs: any(x in l4_vocab for x in xs))
    raw["in_encoded_meta"] = raw["Entry"].isin(set(meta["accession"]))
    raw["embedding_exists"] = raw["Entry"].map(lambda x: existing(ROOT / "data" / "processed" / "embeddings" / f"{x}.npy"))
    raw["contact_map_exists"] = raw["Entry"].map(lambda x: existing(ROOT / "data" / "processed" / "contact_maps" / f"{x}.npy"))

    cmap_rows = []
    for entry in raw["Entry"]:
        nonzero, total, nnz = cmap_stats(ROOT / "data" / "processed" / "contact_maps" / f"{entry}.npy")
        cmap_rows.append({"Entry": entry, "contact_map_nonzero": nonzero, "contact_sum": total, "contact_nnz": nnz})
    raw = raw.merge(pd.DataFrame(cmap_rows), on="Entry", how="left")

    meta["l1"] = meta["ec_raw"].astype(str).str.extract(r"^(\d+)")[0]
    l1_summary = (
        meta.groupby("l1", dropna=False)
        .agg(n=("accession", "count"), median_len=("seq_len", "median"), mean_len=("seq_len", "mean"))
        .reset_index()
    )

    metrics = pd.DataFrame(extract_price_metrics())

    summary_rows = [
        {"category": "raw_price149", "measure": "proteins", "value": int(len(raw))},
        {"category": "encoded_subset", "measure": "proteins", "value": int(len(meta))},
        {"category": "raw_price149", "measure": "encoded_in_sp2018_l4_vocab_all_ec", "value": int(raw["all_ec_in_l4_vocab"].sum())},
        {"category": "raw_price149", "measure": "encoded_in_sp2018_l4_vocab_any_ec", "value": int(raw["any_ec_in_l4_vocab"].sum())},
        {"category": "assets", "measure": "embedding_exists", "value": int(raw["embedding_exists"].sum())},
        {"category": "assets", "measure": "contact_map_exists", "value": int(raw["contact_map_exists"].sum())},
        {"category": "assets", "measure": "contact_map_nonzero", "value": int(raw["contact_map_nonzero"].sum())},
        {"category": "sequence_length", "measure": "median_raw", "value": float(raw["Sequence"].str.len().median())},
        {"category": "sequence_length", "measure": "max_raw", "value": int(raw["Sequence"].str.len().max())},
        {"category": "sequence_length", "measure": "raw_len_gt_1024", "value": int((raw["Sequence"].str.len() > 1024).sum())},
        {"category": "label_space", "measure": "unique_l4_raw", "value": int(len(set(x for xs in raw["ecs"] for x in xs)))},
        {"category": "label_space", "measure": "unique_l4_in_sp2018_vocab", "value": int(len(set(x for xs in raw["ecs"] for x in xs if x in l4_vocab)))},
        {"category": "label_space", "measure": "unique_l4_out_of_sp2018_vocab", "value": int(len(set(x for xs in raw["ecs"] for x in xs if x not in l4_vocab)))},
    ]
    summary = pd.DataFrame(summary_rows)

    raw_out = raw.drop(columns=["ecs"]).copy()
    raw_out.to_csv(OUT_DIR / "price149_failure_audit.csv", index=False)
    summary.to_csv(OUT_DIR / "price149_failure_summary.csv", index=False)
    metrics.to_csv(OUT_DIR / "price149_failure_metrics.csv", index=False)
    l1_summary.to_csv(OUT_DIR / "price149_failure_l1_distribution.csv", index=False)

    # Figure: coarse hierarchy survives, exact Level-4 specificity collapses.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    canonical = metrics[metrics["setting"].eq("canonical flat Level-4")]
    axes[0].bar(canonical["model"], canonical["micro_f1"], color=["#7a8ca1", "#b87f5a", "#32746d"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Micro F1")
    axes[0].set_title("Canonical Price-149 Level-4")
    axes[0].tick_params(axis="x", labelsize=8, pad=4)
    for label in axes[0].get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")

    hier_plot = metrics[(metrics["model"].eq("Contact-EC hierarchical")) & (metrics["level"].isin(["L1", "L2", "L3", "L4"]))]
    axes[1].plot(hier_plot["level"], hier_plot["micro_f1"], marker="o", color="#32746d", linewidth=2)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Micro F1")
    axes[1].set_title("Hierarchy Diagnostic")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(FIG_DIR / "price149_failure_breakdown.png", dpi=300)
    plt.close(fig)

    payload = {
        "summary": summary_rows,
        "metrics": metrics.to_dict(orient="records"),
        "l1_distribution": l1_summary.to_dict(orient="records"),
        "interpretation": [
            "The canonical SP-2018 fusion model improves over ESM-2 alone on Price-149 but remains far below CLEAN-Contact.",
            "Contact-only transfer collapses on Price-149, indicating that the current contact-map branch does not generalize robustly to this external bacterial RefSeq set.",
            "The hierarchical diagnostic shows high Level-1 to Level-3 scores but very low Level-4 micro F1, consistent with a specificity/calibration failure rather than complete functional-family failure.",
            "ESMFold contact maps do not close the gap, suggesting structure-source and preprocessing shift rather than simply missing structure inputs.",
        ],
    }
    with (OUT_DIR / "price149_failure_audit.json").open("w") as f:
        json.dump(payload, f, indent=2)

    md_lines = [
        "# Price-149 failure audit",
        "",
        "## Coverage",
        "",
        summary.to_markdown(index=False),
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Level-1 distribution",
        "",
        l1_summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Contact-EC improves over the SP-2018 ESM-2 baseline on canonical Price-149 Level-4 evaluation (0.2500 vs. 0.0324 micro F1), but remains below CLEAN-Contact.",
        "- Contact-only transfer is 0.0000, so the current structural branch is not robust on this external bacterial RefSeq benchmark.",
        "- Hierarchical diagnostics show strong coarse-family signal (L1 0.9379; L2 0.7862; L3 0.7655) but weak exact Level-4 specificity (0.0508).",
        "- ESMFold-derived contact maps do not recover the gap, supporting the interpretation that external structure-source/preprocessing shift requires retraining or calibration.",
    ]
    (OUT_DIR / "price149_failure_audit.md").write_text("\n".join(md_lines) + "\n")

    print("Wrote Price-149 audit outputs:")
    print(f"  {OUT_DIR / 'price149_failure_summary.csv'}")
    print(f"  {OUT_DIR / 'price149_failure_metrics.csv'}")
    print(f"  {FIG_DIR / 'price149_failure_breakdown.png'}")


if __name__ == "__main__":
    main()
