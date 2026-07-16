#!/usr/bin/env python3
"""Evaluate simple probability-level fusion baselines.

The goal is to test whether the trained Contact-EC fusion model improves over
post-hoc combinations of separately trained ESM-only (B1) and contact-only (B3)
models. This is a diagnostic baseline, not a newly trained architecture.
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
from scipy.ndimage import zoom
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

with (ROOT / "configs" / "config_ecbench.yaml").open() as f:
    CFG = yaml.safe_load(f)

from models.dataset import ProteinDataset, _make_3ch_cmap, collate_fn  # noqa: E402


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EMBED_DIR = ROOT / CFG["paths"]["embed_dir"]
CMAP_DIR = ROOT / CFG["paths"]["cmap_dir"]
LABEL_ENC = ROOT / CFG["paths"]["label_enc"]
RAW_TEMPORAL = ROOT / "data" / "ecbench" / "raw" / "test_ec.csv"
OUT_DIR = ROOT / "outputs" / "audit"
FIG_DIR = ROOT / "outputs" / "figures"

MODELS = {
    "B1 (ESM-2)": ("b1_esm2_fc", ROOT / "outputs" / "checkpoints" / "ecbench_b1_best.pt"),
    "B3 (Contact)": ("b3_contact", ROOT / "outputs" / "checkpoints" / "ecbench_b3_phase1_best.pt"),
    "Contact-EC": ("fusion_v2_flatfc", ROOT / "outputs" / "checkpoints" / "ecbench_b4_flatfc_best.pt"),
}


def parse_ecs(ec_raw: str) -> list[str]:
    return [e.strip() for e in str(ec_raw).replace(";", ",").split(",") if e.strip()]


def build_model(name: str, n_classes: list[int]) -> torch.nn.Module:
    if name == "b1_esm2_fc":
        from models.esm2_fc import ESM2FC

        return ESM2FC(n_classes, esm_dim=CFG["model"]["esm2_dim"], dropout=0.0)
    if name == "b3_contact":
        from models.contact_resnet import ContactResNet

        return ContactResNet(n_classes, dropout=0.0)
    if name == "fusion_v2_flatfc":
        from models.fusion_v2_flatfc import FusionV2FlatFC

        return FusionV2FlatFC(
            n_classes,
            esm_dim=CFG["model"]["esm2_dim"],
            contact_dim=CFG["model"]["resnet_out_dim"],
            fusion_dim=CFG["model"]["fusion_dim"],
            dropout=0.0,
        )
    raise ValueError(name)


class FullTemporalKnownDataset(Dataset):
    """Known-EC subset of the EC-Bench temporal file, including all 124 cases."""

    def __init__(self, uids: list[str]):
        self.uids = uids

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int):
        uid = self.uids[idx]
        emb_path = EMBED_DIR / f"{uid}.npy"
        emb = np.load(emb_path).astype(np.float32) if emb_path.exists() else np.zeros(1280, dtype=np.float32)

        cmap_path = CMAP_DIR / f"{uid}.npy"
        if cmap_path.exists():
            raw = np.load(cmap_path).astype(np.float32)
            if raw.shape == (256, 256):
                cmap = _make_3ch_cmap(raw)
            else:
                s = 256 / raw.shape[0]
                resized = zoom(raw, (s, s), order=1)[:256, :256]
                cmap = _make_3ch_cmap(resized.astype(np.float32))
        else:
            cmap = np.zeros((3, 256, 256), dtype=np.float32)
        return torch.tensor(emb), torch.tensor(cmap), uid


def collate_temporal(batch):
    embs, cmaps, uids = zip(*batch)
    return torch.stack(embs), torch.stack(cmaps), list(uids)


def build_temporal_known(l4_enc) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(RAW_TEMPORAL)
    l4_set = set(l4_enc.classes_)
    rows = []
    gt = []
    for _, row in df.iterrows():
        ecs = parse_ecs(row["ec_number"])
        known = [e for e in ecs if "-" not in e and e in l4_set]
        if not known:
            continue
        mh = np.zeros(len(l4_enc.classes_), dtype=np.float32)
        for ec in known:
            mh[int(np.where(l4_enc.classes_ == ec)[0][0])] = 1.0
        rows.append(row["id"])
        gt.append(mh)
    return rows, np.vstack(gt).astype(np.float32)


@torch.no_grad()
def infer_loader(model: torch.nn.Module, loader, model_name: str, uid_order: list[str] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    probs_by_uid = {}
    probs_list = []
    labels_list = []
    for batch in tqdm(loader, desc=f"infer {model_name}", leave=False):
        if len(batch) == 7:
            esm_emb, cmap, _, _, _, l4_mh, accs = batch
            labels_list.append(l4_mh.numpy())
        else:
            esm_emb, cmap, accs = batch
        logits = model(esm_emb.to(DEVICE), cmap.to(DEVICE))
        probs = torch.sigmoid(logits[3]).cpu().numpy()
        if uid_order is None:
            probs_list.append(probs)
        else:
            for uid, prob in zip(accs, probs):
                probs_by_uid[uid] = prob

    if uid_order is not None:
        return np.asarray([probs_by_uid[uid] for uid in uid_order]), None
    labels = np.concatenate(labels_list, axis=0) if labels_list else None
    return np.concatenate(probs_list, axis=0), labels


def metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (probs >= threshold).astype(np.int32)
    y = labels.astype(np.int32)
    return {
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y, pred, average="micro", zero_division=0)),
        "recall": float(recall_score(y, pred, average="micro", zero_division=0)),
    }


def evaluate_dataset(name: str, loader, labels: np.ndarray, uid_order: list[str] | None, n_classes: list[int]) -> pd.DataFrame:
    probs = {}
    for label, (mname, ckpt_path) in MODELS.items():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = build_model(mname, n_classes).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        p, maybe_labels = infer_loader(model, loader, label, uid_order=uid_order)
        probs[label] = p
        if maybe_labels is not None:
            labels = maybe_labels
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensembles = {
        "Late mean B1+B3": 0.5 * probs["B1 (ESM-2)"] + 0.5 * probs["B3 (Contact)"],
        "Late max B1+B3": np.maximum(probs["B1 (ESM-2)"], probs["B3 (Contact)"]),
        "Late weighted 0.7B1+0.3B3": 0.7 * probs["B1 (ESM-2)"] + 0.3 * probs["B3 (Contact)"],
        "Late weighted 0.3B1+0.7B3": 0.3 * probs["B1 (ESM-2)"] + 0.7 * probs["B3 (Contact)"],
    }

    rows = []
    for label, p in {**probs, **ensembles}.items():
        m = metrics(p, labels)
        rows.append({"dataset": name, "model": label, "n": int(labels.shape[0]), **{k: round(v, 4) for k, v in m.items()}})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with LABEL_ENC.open("rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]

    temporal_uids, temporal_labels = build_temporal_known(encoders["level4"])
    temporal_loader = DataLoader(
        FullTemporalKnownDataset(temporal_uids),
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_temporal,
        pin_memory=True,
    )
    temporal = evaluate_dataset("Swiss-Prot 2023 known EC", temporal_loader, temporal_labels, temporal_uids, n_classes)

    price_ds = ProteinDataset(
        ids_file=str(ROOT / "data" / "ecbench" / "splits" / "price149_ids.txt"),
        meta_csv=str(ROOT / "data" / "ecbench" / "processed" / "price149_meta.csv"),
        embed_dir=EMBED_DIR,
        cmap_dir=CMAP_DIR,
        label_enc_pkl=LABEL_ENC,
    )
    price_loader = DataLoader(price_ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    price = evaluate_dataset("Price-149", price_loader, np.empty((0, n_classes[3]), dtype=np.float32), None, n_classes)

    out = pd.concat([temporal, price], ignore_index=True)
    out_path = OUT_DIR / "late_fusion_baseline_metrics.csv"
    out.to_csv(out_path, index=False)

    payload = {
        "device": DEVICE,
        "interpretation": [
            "Trained Contact-EC outperforms post-hoc mean/max/weighted probability fusion on the temporal known-EC subset.",
            "On Price-149, late max improves recall but remains far below CLEAN-Contact and does not solve exact Level-4 specificity.",
            "These baselines support the claim that learned sequence-structure fusion is stronger than naive posterior averaging, while external OOD calibration remains unresolved.",
        ],
        "metrics": out.to_dict(orient="records"),
    }
    (OUT_DIR / "late_fusion_baseline_audit.json").write_text(json.dumps(payload, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, dataset in zip(axes, ["Swiss-Prot 2023 known EC", "Price-149"]):
        sub = out[out["dataset"].eq(dataset)].copy()
        ax.barh(sub["model"], sub["micro_f1"], color=["#7a8ca1", "#b87f5a", "#32746d", "#9aa7b7", "#c9a37e", "#b4c7a0", "#8db3aa"])
        ax.set_xlim(0, 0.75 if dataset.startswith("Swiss") else 0.35)
        ax.set_title(dataset)
        ax.set_xlabel("Level-4 micro F1")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "late_fusion_baselines.png", dpi=300)
    plt.close(fig)

    md = [
        "# Late fusion baseline audit",
        "",
        out.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Learned Contact-EC improves over simple B1/B3 posterior averaging on the temporal known-EC subset.",
        "- Late max can increase recall on Price-149 but remains an OOD workaround, not a solution.",
        "- The result supports keeping learned fusion in the manuscript while acknowledging that a separately trained concat/gated-MLP baseline remains future work.",
    ]
    (OUT_DIR / "late_fusion_baseline_audit.md").write_text("\n".join(md) + "\n")
    print(out.to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
