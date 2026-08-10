#!/usr/bin/env python3
"""Stratify temporal-set performance by recorded train-set sequence similarity.

This audit joins the available EC-Bench test-vs-train similarity file with the
case-wise HIT-EC/Contact-EC comparison table.  The similarity file covers the
101-protein EC-Bench temporal subset; the case-wise table covers the 124
fully evaluable temporal proteins used in the final manuscript.

Outputs:
  outputs/audit/temporal_similarity_per_protein.csv
  outputs/audit/temporal_similarity_bin_summary.csv
  outputs/audit/temporal_similarity_audit.json
  outputs/audit/temporal_similarity_audit.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CASEWISE_CSV = ROOT / "outputs" / "results" / "casewise_hitec_contactec.csv"
SIM_JSON = ROOT / "data" / "ecbench" / "splits" / "test_vs_train_sim.json"
OUT_DIR = ROOT / "outputs" / "audit"
FIG_DIR = ROOT / "outputs" / "figures"


def split_ecs(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {x.strip() for x in text.split(";") if x.strip()}


def micro_f1_from_sets(rows: Iterable[dict], pred_col: str) -> float:
    tp = 0
    pred_total = 0
    true_total = 0
    for row in rows:
        true = split_ecs(row.get("true_ecs"))
        pred = split_ecs(row.get(pred_col))
        tp += len(true & pred)
        pred_total += len(pred)
        true_total += len(true)
    denom = pred_total + true_total
    if denom == 0:
        return 0.0
    return 2.0 * tp / denom


def sim_bin(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "not_available"
    value = float(value)
    if value == 0:
        return "0.00"
    if value <= 0.30:
        return "(0,0.30]"
    if value <= 0.60:
        return "(0.30,0.60]"
    if value <= 0.90:
        return "(0.60,0.90]"
    return ">0.90"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    order = ["0.00", "(0,0.30]", "(0.30,0.60]", "(0.60,0.90]", ">0.90"]
    rows = []
    for bin_name in order:
        sub = df[df["similarity_bin"] == bin_name].copy()
        if sub.empty:
            continue
        records = sub.to_dict(orient="records")
        rows.append(
            {
                "similarity_bin": bin_name,
                "n": int(len(sub)),
                "mean_max_train_seq_identity": round(float(sub["max_train_seq_identity"].mean()), 4),
                "median_max_train_seq_identity": round(float(sub["max_train_seq_identity"].median()), 4),
                "mean_seq_len": round(float(sub["seq_len"].mean()), 1),
                "contactec_micro_f1": round(micro_f1_from_sets(records, "contactec_pred_ecs"), 4),
                "hitec_micro_f1": round(micro_f1_from_sets(records, "hitec_pred_ecs"), 4),
                "contactec_hit_rate": round(float(sub["contactec_hit"].mean()), 4),
                "hitec_hit_rate": round(float(sub["hitec_hit"].mean()), 4),
                "contactec_mean_per_protein_f1": round(float(sub["contactec_per_protein_f1"].mean()), 4),
                "hitec_mean_per_protein_f1": round(float(sub["hitec_per_protein_f1"].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    casewise = pd.read_csv(CASEWISE_CSV)
    with open(SIM_JSON) as f:
        sim = {str(k): float(v) for k, v in json.load(f).items()}

    casewise["max_train_seq_identity"] = casewise["uid"].map(sim)
    casewise["similarity_available"] = casewise["max_train_seq_identity"].notna()
    casewise["similarity_bin"] = casewise["max_train_seq_identity"].map(sim_bin)
    casewise["contactec_hit"] = casewise["contactec_hit"].astype(bool)
    casewise["hitec_hit"] = casewise["hitec_hit"].astype(bool)

    per_protein_out = OUT_DIR / "temporal_similarity_per_protein.csv"
    cols = [
        "uid",
        "seq_len",
        "true_ecs",
        "true_l1",
        "max_train_seq_identity",
        "similarity_available",
        "similarity_bin",
        "contactec_pred_ecs",
        "contactec_per_protein_f1",
        "contactec_hit",
        "hitec_pred_ecs",
        "hitec_per_protein_f1",
        "hitec_hit",
        "case_status",
    ]
    casewise[cols].to_csv(per_protein_out, index=False)

    with_sim = casewise[casewise["similarity_available"]].copy()
    summary = summarize(with_sim)
    summary_out = OUT_DIR / "temporal_similarity_bin_summary.csv"
    summary.to_csv(summary_out, index=False)
    fig_out = FIG_DIR / "temporal_similarity_strata.png"
    try:
        import matplotlib.pyplot as plt

        if not summary.empty:
            x = range(len(summary))
            width = 0.36
            fig, ax = plt.subplots(figsize=(7.4, 3.9))
            ax.bar(
                [i - width / 2 for i in x],
                summary["contactec_micro_f1"],
                width=width,
                label="Contact-EC",
                color="#3b82f6",
            )
            ax.bar(
                [i + width / 2 for i in x],
                summary["hitec_micro_f1"],
                width=width,
                label="HIT-EC",
                color="#ef4444",
            )
            ax.set_xticks(list(x))
            ax.set_xticklabels(summary["similarity_bin"], rotation=0, ha="center")
            ax.tick_params(axis="x", pad=5)
            ax.set_ylim(0, 1.02)
            ax.set_ylabel("Level-4 micro F1")
            ax.set_xlabel("Maximum recorded train sequence identity")
            ax.legend(frameon=False, ncol=2, loc="upper left")
            ax.grid(axis="y", alpha=0.25, linewidth=0.8)
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.18)
            fig.savefig(fig_out, dpi=300)
            plt.close(fig)
    except Exception as exc:  # pragma: no cover - figure is a convenience output
        fig_out = None
        print(f"Figure generation skipped: {exc}")

    overall_records = with_sim.to_dict(orient="records")
    overall = {
        "casewise_total_n": int(len(casewise)),
        "similarity_available_n": int(len(with_sim)),
        "similarity_missing_n": int((~casewise["similarity_available"]).sum()),
        "similarity_source": str(SIM_JSON.relative_to(ROOT)),
        "casewise_source": str(CASEWISE_CSV.relative_to(ROOT)),
        "max_recorded_similarity": round(float(with_sim["max_train_seq_identity"].max()), 4),
        "median_recorded_similarity": round(float(with_sim["max_train_seq_identity"].median()), 4),
        "n_similarity_eq_0": int((with_sim["max_train_seq_identity"] == 0).sum()),
        "n_similarity_le_0_30": int((with_sim["max_train_seq_identity"] <= 0.30).sum()),
        "n_similarity_gt_0_90": int((with_sim["max_train_seq_identity"] > 0.90).sum()),
        "contactec_micro_f1_on_similarity_subset": round(
            micro_f1_from_sets(overall_records, "contactec_pred_ecs"), 4
        ),
        "hitec_micro_f1_on_similarity_subset": round(
            micro_f1_from_sets(overall_records, "hitec_pred_ecs"), 4
        ),
        "interpretation": (
            "This is a sequence-similarity stratification over the 101 temporal "
            "proteins for which EC-Bench test-vs-train similarity values are "
            "available. It is not a fold-disjoint structural evaluation."
        ),
    }

    json_out = OUT_DIR / "temporal_similarity_audit.json"
    json_out.write_text(
        json.dumps({"overall": overall, "bins": summary.to_dict(orient="records")}, indent=2)
        + "\n"
    )

    md_out = OUT_DIR / "temporal_similarity_audit.md"
    lines = [
        "# Temporal Similarity-Stratified Audit",
        "",
        "This audit joins the final 124-protein temporal case-wise comparison with the",
        "available EC-Bench `test_vs_train_sim.json` sequence-similarity file.",
        "",
        "Important limitation: the similarity file covers 101 temporal proteins, not",
        "all 124 fully evaluable proteins. These results therefore support a",
        "sequence-similarity sensitivity analysis, not a fold-disjoint claim.",
        "",
        "## Overall",
        "",
        f"- Case-wise temporal proteins: {overall['casewise_total_n']}",
        f"- Proteins with recorded train similarity: {overall['similarity_available_n']}",
        f"- Proteins without recorded train similarity: {overall['similarity_missing_n']}",
        f"- Max recorded train identity: {overall['max_recorded_similarity']:.4f}",
        f"- Median recorded train identity: {overall['median_recorded_similarity']:.4f}",
        f"- N with identity = 0: {overall['n_similarity_eq_0']}",
        f"- N with identity <= 0.30: {overall['n_similarity_le_0_30']}",
        f"- N with identity > 0.90: {overall['n_similarity_gt_0_90']}",
        f"- Contact-EC micro F1 on similarity subset: {overall['contactec_micro_f1_on_similarity_subset']:.4f}",
        f"- HIT-EC micro F1 on similarity subset: {overall['hitec_micro_f1_on_similarity_subset']:.4f}",
        f"- Figure: {str(fig_out.relative_to(ROOT)) if fig_out else 'not generated'}",
        "",
        "## Bin Summary",
        "",
    ]
    if summary.empty:
        lines.append("No bins available.")
    else:
        lines.append(summary.to_markdown(index=False))
    lines += [
        "",
        "## Use in Manuscript",
        "",
        "Recommended wording: performance was stratified by the available EC-Bench",
        "train-similarity audit for 101 temporal proteins. The analysis shows how",
        "Contact-EC and HIT-EC vary across low- and high-similarity temporal cases,",
        "but does not replace a Foldseek/TM-align or CATH/SCOP fold-disjoint test.",
        "",
    ]
    md_out.write_text("\n".join(lines))

    print(f"Wrote {per_protein_out}")
    print(f"Wrote {summary_out}")
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
