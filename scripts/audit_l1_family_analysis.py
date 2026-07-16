#!/usr/bin/env python3
"""Level-1 enzyme-family analysis on the temporal known-label set.

The script evaluates the main local checkpoints on the 124 fully evaluable
Swiss-Prot 2023-01 temporal proteins and aggregates Level-4 predictions by
true EC Level-1 family.  HIT-EC predictions are read from the existing
case-wise comparison table.

Outputs:
  outputs/audit/l1_family_per_protein.csv
  outputs/audit/l1_family_summary.csv
  outputs/audit/l1_family_audit.json
  outputs/audit/l1_family_audit.md
  outputs/figures/l1_family_micro_f1.png
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.dataset import ProteinDataset, collate_fn
from models.esm2_fc import ESM2FC
from models.contact_resnet import ContactResNet
from models.fusion_v2_flatfc import FusionV2FlatFC


CFG_PATH = ROOT / "configs" / "config_ecbench.yaml"
IDS_FILE = ROOT / "data" / "ecbench" / "splits" / "test_ids_full.txt"
META_CSV = ROOT / "data" / "ecbench" / "processed" / "test_meta_full.csv"
CASEWISE_CSV = ROOT / "outputs" / "results" / "casewise_hitec_contactec.csv"
OUT_DIR = ROOT / "outputs" / "audit"
FIG_DIR = ROOT / "outputs" / "figures"

L1_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}


def load_cfg() -> dict:
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def load_encoders(cfg: dict) -> dict:
    with open(ROOT / cfg["paths"]["label_enc"], "rb") as f:
        return pickle.load(f)


def split_ecs(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {x.strip() for x in text.split(";") if x.strip()}


def ecs_from_multihot(vec: np.ndarray, classes: np.ndarray) -> list[str]:
    idxs = np.flatnonzero(vec)
    return [str(classes[i]) for i in idxs]


def micro_f1(rows: Iterable[dict], pred_col: str) -> float:
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
    return 0.0 if denom == 0 else 2.0 * tp / denom


def hit_rate(rows: Iterable[dict], pred_col: str) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    hits = 0
    for row in rows:
        true = split_ecs(row.get("true_ecs"))
        pred = split_ecs(row.get(pred_col))
        hits += int(bool(true & pred))
    return hits / len(rows)


def build_model(name: str, n_classes: list[int], cfg: dict) -> torch.nn.Module:
    if name == "b1_esm2_fc":
        return ESM2FC(n_classes, esm_dim=cfg["model"]["esm2_dim"], dropout=0.0)
    if name == "b3_contact":
        return ContactResNet(n_classes, dropout=0.0)
    if name == "contact_ec":
        return FusionV2FlatFC(
            n_classes,
            esm_dim=cfg["model"]["esm2_dim"],
            contact_dim=cfg["model"]["resnet_out_dim"],
            fusion_dim=cfg["model"]["fusion_dim"],
            dropout=0.0,
        )
    raise ValueError(name)


@torch.no_grad()
def infer_model(
    model_name: str,
    ckpt_path: Path,
    loader: DataLoader,
    n_classes: list[int],
    l4_classes: np.ndarray,
    cfg: dict,
    device: str,
    threshold: float = 0.5,
) -> dict[str, dict]:
    model = build_model(model_name, n_classes, cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rows: dict[str, dict] = {}
    for batch in loader:
        esm_emb, cmap, _, _, _, l4_mh, accs = batch
        logits = model(esm_emb.to(device), cmap.to(device))
        probs = torch.sigmoid(logits[3]).cpu().numpy()
        labels = l4_mh.numpy().astype(np.int32)
        preds = (probs >= threshold).astype(np.int32)
        for uid, pred, prob, lab in zip(accs, preds, probs, labels):
            pred_ecs = ecs_from_multihot(pred, l4_classes)
            true_ecs = ecs_from_multihot(lab, l4_classes)
            top_idx = np.argsort(prob)[::-1][:5]
            rows[str(uid)] = {
                f"{model_name}_pred_ecs": ";".join(pred_ecs),
                f"{model_name}_hit": bool(set(true_ecs) & set(pred_ecs)),
                f"{model_name}_top5": ";".join(
                    f"{str(l4_classes[i])}:{float(prob[i]):.4f}" for i in top_idx
                ),
            }
    return rows


def make_summary(df: pd.DataFrame, pred_cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    order = ["1", "2", "3", "4", "5", "6", "7"]
    for l1 in order:
        sub = df[df["true_l1"] == l1].copy()
        if sub.empty:
            continue
        records = sub.to_dict(orient="records")
        row = {
            "true_l1": l1,
            "family": L1_NAMES.get(l1, f"EC {l1}"),
            "n": int(len(sub)),
            "mean_seq_len": round(float(sub["seq_len"].mean()), 1),
        }
        for label, col in pred_cols.items():
            row[f"{label}_micro_f1"] = round(micro_f1(records, col), 4)
            row[f"{label}_hit_rate"] = round(hit_rate(records, col), 4)
        row["fusion_minus_esm2"] = round(
            row["ContactEC_micro_f1"] - row["ESM2_micro_f1"], 4
        )
        row["fusion_minus_contact"] = round(
            row["ContactEC_micro_f1"] - row["ContactOnly_micro_f1"], 4
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [f"EC {r.true_l1}\n{r.n}" for r in summary.itertuples()]
    x = np.arange(len(summary))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    ax.bar(x - 1.5 * width, summary["ESM2_micro_f1"], width, label="ESM-2 only", color="#64748b")
    ax.bar(x - 0.5 * width, summary["ContactOnly_micro_f1"], width, label="Contact only", color="#22c55e")
    ax.bar(x + 0.5 * width, summary["ContactEC_micro_f1"], width, label="Contact-EC", color="#3b82f6")
    ax.bar(x + 1.5 * width, summary["HITEC_micro_f1"], width, label="HIT-EC", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Level-4 micro F1")
    ax.set_xlabel("True EC Level-1 family (N)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    encoders = load_encoders(cfg)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    l4_classes = encoders["level4"].classes_
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ds = ProteinDataset(
        ids_file=str(IDS_FILE),
        meta_csv=str(META_CSV),
        embed_dir=str(ROOT / cfg["paths"]["embed_dir"]),
        cmap_dir=str(ROOT / cfg["paths"]["cmap_dir"]),
        label_enc_pkl=str(ROOT / cfg["paths"]["label_enc"]),
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn)

    ckpts = {
        "b1_esm2_fc": ROOT / "outputs" / "checkpoints" / "ecbench_b1_best.pt",
        "b3_contact": ROOT / "outputs" / "checkpoints" / "ecbench_b3_phase1_best.pt",
        "contact_ec": ROOT / "outputs" / "checkpoints" / "ecbench_b4_flatfc_best.pt",
    }

    model_rows: dict[str, dict] = {}
    for name, ckpt_path in ckpts.items():
        preds = infer_model(name, ckpt_path, loader, n_classes, l4_classes, cfg, device)
        for uid, values in preds.items():
            model_rows.setdefault(uid, {}).update(values)

    casewise = pd.read_csv(CASEWISE_CSV)
    meta = pd.read_csv(META_CSV)[["accession", "seq_len", "ec_raw"]]
    meta = meta.rename(columns={"accession": "uid"})
    df = casewise[["uid", "seq_len", "true_ecs", "true_l1", "hitec_pred_ecs"]].copy()
    df["true_l1"] = df["true_l1"].astype(str)
    for uid, values in model_rows.items():
        for k, v in values.items():
            df.loc[df["uid"] == uid, k] = v

    # Normalize column names used in output and summary.
    df = df.rename(
        columns={
            "b1_esm2_fc_pred_ecs": "esm2_pred_ecs",
            "b1_esm2_fc_hit": "esm2_hit",
            "b3_contact_pred_ecs": "contact_only_pred_ecs",
            "b3_contact_hit": "contact_only_hit",
            "contact_ec_pred_ecs": "contactec_pred_ecs_recomputed",
            "contact_ec_hit": "contactec_hit_recomputed",
        }
    )

    per_protein_cols = [
        "uid",
        "seq_len",
        "true_ecs",
        "true_l1",
        "esm2_pred_ecs",
        "esm2_hit",
        "contact_only_pred_ecs",
        "contact_only_hit",
        "contactec_pred_ecs_recomputed",
        "contactec_hit_recomputed",
        "hitec_pred_ecs",
    ]
    per_protein_out = OUT_DIR / "l1_family_per_protein.csv"
    df[per_protein_cols].to_csv(per_protein_out, index=False)

    pred_cols = {
        "ESM2": "esm2_pred_ecs",
        "ContactOnly": "contact_only_pred_ecs",
        "ContactEC": "contactec_pred_ecs_recomputed",
        "HITEC": "hitec_pred_ecs",
    }
    summary = make_summary(df, pred_cols)
    summary_out = OUT_DIR / "l1_family_summary.csv"
    summary.to_csv(summary_out, index=False)

    fig_out = FIG_DIR / "l1_family_micro_f1.png"
    try:
        plot_summary(summary, fig_out)
    except Exception as exc:  # pragma: no cover
        print(f"Figure generation skipped: {exc}")
        fig_out = None

    audit = {
        "n": int(len(df)),
        "device": device,
        "checkpoints": {k: str(v.relative_to(ROOT)) for k, v in ckpts.items()},
        "summary": summary.to_dict(orient="records"),
        "interpretation": (
            "Level-1 family analysis exposes which enzyme groups dominate the "
            "temporal gain. Small family counts should be interpreted as "
            "hypothesis-generating rather than definitive biological evidence."
        ),
    }
    json_out = OUT_DIR / "l1_family_audit.json"
    json_out.write_text(json.dumps(audit, indent=2) + "\n")

    md_out = OUT_DIR / "l1_family_audit.md"
    lines = [
        "# EC Level-1 Family Analysis",
        "",
        f"N = {len(df)} fully evaluable Swiss-Prot 2023-01 temporal proteins.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "Contact-EC is strongest in EC families where ESM-2 and contact-only signals",
        "are both non-trivial, while families with few examples remain unstable.",
        "This analysis adds biological granularity but should not be overinterpreted",
        "as a mechanistic active-site explanation.",
        "",
        f"Figure: {str(fig_out.relative_to(ROOT)) if fig_out else 'not generated'}",
        "",
    ]
    md_out.write_text("\n".join(lines))

    print(f"Wrote {per_protein_out}")
    print(f"Wrote {summary_out}")
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    if fig_out:
        print(f"Wrote {fig_out}")


if __name__ == "__main__":
    main()
