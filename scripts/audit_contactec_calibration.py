"""
Contact-EC calibration and prediction-cardinality audit.

This diagnostic compares score distributions on:
  - Swiss-Prot 2023 complete known EC proteins
  - Swiss-Prot 2023 partial EC proteins
  - Swiss-Prot 2023 novel complete EC proteins
  - Price-149 external bacterial benchmark

For complete-label datasets, it reports Level-4 F1, top-1 hit rate, Brier score,
and a simple flattened multilabel ECE. For partial/novel temporal rows, Level-4
F1 is undefined, so only closed-set prediction burden and confidence statistics
are reported.

Outputs:
  outputs/audit/contactec_calibration_audit.{csv,json,md}
  outputs/figures/contactec_calibration_diagnostics.png
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.dataset import ProteinDataset, _make_3ch_cmap, collate_fn  # noqa: E402

with open(ROOT / "configs" / "config_ecbench.yaml") as f:
    CFG = yaml.safe_load(f)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RAW_TEMPORAL = ROOT / "data" / "ecbench" / "raw" / "test_ec.csv"
EMBED_DIR = ROOT / CFG["paths"]["embed_dir"]
CMAP_DIR = ROOT / CFG["paths"]["cmap_dir"]
LABEL_ENC = ROOT / CFG["paths"]["label_enc"]
CHECKPOINT = ROOT / "outputs" / "checkpoints" / "ecbench_b4_flatfc_best.pt"
THRESHOLD = 0.5


def parse_ecs(ec_raw: str) -> list[str]:
    return [e.strip() for e in str(ec_raw).replace(";", ",").split(",") if e.strip()]


def classify_ecs(ecs: list[str], l4_vocab: set[str]) -> str:
    known = [e for e in ecs if "-" not in e and e in l4_vocab]
    partial = any("-" in e.split(".") for e in ecs)
    novel = any("-" not in e and e not in l4_vocab for e in ecs)
    if known:
        return "temporal_known_complete"
    if partial:
        return "temporal_partial"
    if novel:
        return "temporal_novel_complete"
    return "temporal_unscorable"


class TemporalFullDataset(Dataset):
    def __init__(self, uids: list[str]):
        self.uids = uids

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int):
        uid = self.uids[idx]
        emb_path = EMBED_DIR / f"{uid}.npy"
        cmap_path = CMAP_DIR / f"{uid}.npy"
        emb = np.load(emb_path).astype(np.float32) if emb_path.exists() else np.zeros(1280, dtype=np.float32)
        if cmap_path.exists():
            raw = np.load(cmap_path).astype(np.float32)
            cmap = _make_3ch_cmap(raw) if raw.shape == (256, 256) else np.zeros((3, 256, 256), dtype=np.float32)
        else:
            cmap = np.zeros((3, 256, 256), dtype=np.float32)
        return torch.tensor(emb), torch.tensor(cmap), uid


def collate_temporal(batch):
    embs, cmaps, uids = zip(*batch)
    return torch.stack(embs), torch.stack(cmaps), list(uids)


def build_model(n_classes: list[int]):
    from models.fusion_v2_flatfc import FusionV2FlatFC

    return FusionV2FlatFC(
        n_classes,
        esm_dim=CFG["model"]["esm2_dim"],
        contact_dim=CFG["model"]["resnet_out_dim"],
        fusion_dim=CFG["model"]["fusion_dim"],
        dropout=0.0,
    )


@torch.no_grad()
def infer_temporal(model, uids: list[str]) -> np.ndarray:
    loader = DataLoader(
        TemporalFullDataset(uids),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_temporal,
        pin_memory=False,
    )
    by_uid = {}
    model.eval()
    for emb, cmap, batch_uids in tqdm(loader, desc="infer temporal", leave=False):
        logits = model(emb.to(DEVICE), cmap.to(DEVICE))
        probs = torch.sigmoid(logits[3]).cpu().numpy()
        for uid, prob in zip(batch_uids, probs):
            by_uid[uid] = prob
    return np.stack([by_uid[uid] for uid in uids])


@torch.no_grad()
def infer_price(model, n_l4: int) -> tuple[np.ndarray, np.ndarray]:
    ds = ProteinDataset(
        ids_file=str(ROOT / "data" / "ecbench" / "splits" / "price149_ids.txt"),
        meta_csv=str(ROOT / "data" / "ecbench" / "processed" / "price149_meta.csv"),
        embed_dir=EMBED_DIR,
        cmap_dir=CMAP_DIR,
        label_enc_pkl=LABEL_ENC,
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)
    probs_list, labels_list = [], []
    model.eval()
    for emb, cmap, _, _, _, l4_mh, _ in tqdm(loader, desc="infer Price-149", leave=False):
        logits = model(emb.to(DEVICE), cmap.to(DEVICE))
        probs_list.append(torch.sigmoid(logits[3]).cpu().numpy())
        labels_list.append(l4_mh.numpy())
    return np.concatenate(probs_list), np.concatenate(labels_list)


def multilabel_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    p = probs.reshape(-1)
    y = labels.reshape(-1).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += float(mask.mean()) * abs(conf - acc)
    return ece


def summarize(dataset: str, probs: np.ndarray, labels: np.ndarray | None = None) -> dict:
    preds = (probs >= THRESHOLD).astype(np.int32)
    row = {
        "dataset": dataset,
        "n": int(probs.shape[0]),
        "avg_pred_labels": float(preds.sum(axis=1).mean()),
        "median_pred_labels": float(np.median(preds.sum(axis=1))),
        "empty_prediction_rate": float((preds.sum(axis=1) == 0).mean()),
        "top1_prob_mean": float(probs.max(axis=1).mean()),
        "top1_prob_median": float(np.median(probs.max(axis=1))),
        "mean_probability": float(probs.mean()),
    }
    if labels is None:
        row.update(
            {
                "micro_f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "top1_hit_rate": np.nan,
                "brier": np.nan,
                "ece": np.nan,
            }
        )
        return row

    y = labels.astype(np.int32)
    top1 = probs.argmax(axis=1)
    row.update(
        {
            "micro_f1": float(f1_score(y, preds, average="micro", zero_division=0)),
            "precision": float(precision_score(y, preds, average="micro", zero_division=0)),
            "recall": float(recall_score(y, preds, average="micro", zero_division=0)),
            "top1_hit_rate": float(y[np.arange(y.shape[0]), top1].mean()),
            "brier": float(np.mean((probs - y) ** 2)),
            "ece": float(multilabel_ece(probs, y)),
        }
    )
    return row


def main() -> None:
    out_dir = ROOT / "outputs" / "audit"
    fig_dir = ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(LABEL_ENC, "rb") as f:
        encoders = pickle.load(f)
    l4 = encoders["level4"]
    l4_vocab = set(l4.classes_)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]

    model = build_model(n_classes).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])

    df = pd.read_csv(RAW_TEMPORAL)
    uids = df["id"].astype(str).tolist()
    groups = []
    temporal_labels = np.zeros((len(df), len(l4.classes_)), dtype=np.int32)
    for i, row in df.iterrows():
        ecs = parse_ecs(row["ec_number"])
        groups.append(classify_ecs(ecs, l4_vocab))
        for ec in ecs:
            if "-" not in ec and ec in l4_vocab:
                temporal_labels[i, int(np.where(l4.classes_ == ec)[0][0])] = 1

    temporal_probs = infer_temporal(model, uids)
    price_probs, price_labels = infer_price(model, len(l4.classes_))

    rows = []
    group_arr = np.array(groups)
    for group in ["temporal_known_complete", "temporal_partial", "temporal_novel_complete"]:
        mask = group_arr == group
        labels = temporal_labels[mask] if group == "temporal_known_complete" else None
        rows.append(summarize(group, temporal_probs[mask], labels))
    rows.append(summarize("Price-149", price_probs, price_labels))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "contactec_calibration_audit.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    order = summary["dataset"].tolist()
    label_map = {
        "temporal_known_complete": "Temporal\nknown",
        "temporal_partial": "Temporal\npartial",
        "temporal_novel_complete": "Temporal\nnovel",
        "Price-149": "Price-149",
    }
    xlabels = [label_map.get(label, label.replace("_", "\n")) for label in order]
    axes[0].bar(order, summary["top1_prob_mean"], color="#5f7f95")
    axes[0].set_ylabel("Mean top-1 sigmoid probability")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(xlabels, rotation=0, ha="center", fontsize=8)
    axes[0].tick_params(axis="x", pad=5)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(order, summary["avg_pred_labels"], color="#9a7b4f")
    axes[1].set_ylabel("Average Level-4 predictions at threshold 0.5")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(xlabels, rotation=0, ha="center", fontsize=8)
    axes[1].tick_params(axis="x", pad=5)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(fig_dir / "contactec_calibration_diagnostics.png", dpi=300)
    plt.close(fig)

    payload = {
        "threshold": THRESHOLD,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "metrics": summary.to_dict(orient="records"),
        "interpretation": [
            "Temporal partial and novel rows are not Level-4 scorable, but their closed-set prediction burden can still be audited.",
            "Price-149 has lower exact Level-4 F1 despite non-empty confident predictions, supporting an OOD specificity/calibration failure.",
            "The flattened multilabel ECE is dominated by the many negative labels and should be interpreted as a diagnostic rather than a full calibration study.",
        ],
    }
    (out_dir / "contactec_calibration_audit.json").write_text(json.dumps(payload, indent=2))

    md = [
        "# Contact-EC calibration and prediction-cardinality audit",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "- Partial and novel temporal proteins are not scored at Level 4, but Contact-EC still emits closed-set predictions for them.",
        "- Price-149 shows non-empty predictions but low exact Level-4 correctness, supporting an OOD specificity/calibration interpretation.",
        "- This diagnostic complements the threshold-sensitivity and Price-149 failure analyses.",
    ]
    (out_dir / "contactec_calibration_audit.md").write_text("\n".join(md) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
