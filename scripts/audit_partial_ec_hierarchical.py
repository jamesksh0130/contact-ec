"""
Hierarchical prefix audit for partial EC annotations in the EC-Bench temporal set.

The main Level-4 temporal score is only well-defined for complete EC labels that
exist in the SP-2018 training vocabulary. This audit evaluates partial EC rows at
the deepest annotated prefix level instead of treating them as Level-4 failures.

Outputs:
  outputs/audit/partial_ec_hierarchical_metrics.csv
  outputs/audit/partial_ec_hierarchical_summary.csv
  outputs/audit/partial_ec_hierarchical_audit.{json,md}
  outputs/figures/partial_ec_hierarchical_f1.png
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

from models.dataset import _make_3ch_cmap  # noqa: E402

with open(ROOT / "configs" / "config_ecbench.yaml") as f:
    CFG = yaml.safe_load(f)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RAW_TEST = ROOT / "data" / "ecbench" / "raw" / "test_ec.csv"
EMBED_DIR = ROOT / CFG["paths"]["embed_dir"]
CMAP_DIR = ROOT / CFG["paths"]["cmap_dir"]
LABEL_ENC = ROOT / CFG["paths"]["label_enc"]

MODELS = {
    "B1 (ESM-2)": ("b1_esm2_fc", "outputs/checkpoints/ecbench_b1_best.pt"),
    "B3 (Contact)": ("b3_contact", "outputs/checkpoints/ecbench_b3_phase1_best.pt"),
    "Contact-EC": ("fusion_v2_flatfc", "outputs/checkpoints/ecbench_b4_flatfc_best.pt"),
    "Contact-EC-Hier": ("fusion_v2", "outputs/checkpoints/ecbench_fv2_phase2_best.pt"),
}


def parse_ecs(ec_raw: str) -> list[str]:
    return [e.strip() for e in str(ec_raw).replace(";", ",").split(",") if e.strip()]


def ec_prefix(ec: str, level: int) -> str | None:
    parts = ec.split(".")
    if len(parts) < level:
        return None
    prefix = parts[:level]
    if any(p == "-" or p == "" for p in prefix):
        return None
    return ".".join(prefix)


def classify_ecs(ecs: list[str], l4_vocab: set[str]) -> str:
    known_l4 = [e for e in ecs if ec_prefix(e, 4) and e in l4_vocab]
    partial = any("-" in e.split(".") for e in ecs)
    novel = any(ec_prefix(e, 4) and e not in l4_vocab for e in ecs)
    if known_l4:
        return "known_complete"
    if partial:
        return "partial"
    if novel:
        return "novel_complete"
    return "unscorable"


def build_hierarchical_labels(df: pd.DataFrame, encoders: dict) -> tuple[list[str], dict[int, np.ndarray], dict[int, np.ndarray], list[str], list[int]]:
    labels: dict[int, list[np.ndarray]] = {}
    masks: dict[int, list[bool]] = {}
    class_maps = {}
    for level in range(1, 5):
        classes = list(encoders[f"level{level}"].classes_)
        class_maps[level] = {c: i for i, c in enumerate(classes)}
        labels[level] = []
        masks[level] = []

    l4_vocab = set(encoders["level4"].classes_)
    uids: list[str] = []
    groups: list[str] = []
    max_depths: list[int] = []

    for _, row in df.iterrows():
        uids.append(str(row["id"]))
        ecs = parse_ecs(row["ec_number"])
        groups.append(classify_ecs(ecs, l4_vocab))
        row_max = 0

        for level in range(1, 5):
            mh = np.zeros(len(class_maps[level]), dtype=np.int32)
            for ec in ecs:
                p = ec_prefix(ec, level)
                if p is not None and p in class_maps[level]:
                    mh[class_maps[level][p]] = 1
            has_label = bool(mh.sum() > 0)
            if has_label:
                row_max = max(row_max, level)
            labels[level].append(mh)
            masks[level].append(has_label)
        max_depths.append(row_max)

    label_arrays = {k: np.stack(v).astype(np.int32) for k, v in labels.items()}
    mask_arrays = {k: np.array(v, dtype=bool) for k, v in masks.items()}
    return uids, label_arrays, mask_arrays, groups, max_depths


class TemporalDataset(Dataset):
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


def collate_fn(batch):
    embs, cmaps, uids = zip(*batch)
    return torch.stack(embs), torch.stack(cmaps), list(uids)


def build_model(name: str, n_classes: list[int]):
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
    if name == "fusion_v2":
        from models.fusion_v2 import FusionModelV2

        return FusionModelV2(
            n_classes,
            esm_dim=CFG["model"]["esm2_dim"],
            contact_dim=CFG["model"]["resnet_out_dim"],
            fusion_dim=CFG["model"]["fusion_dim"],
            dropout=0.0,
        )
    raise ValueError(name)


@torch.no_grad()
def infer(model, loader, uid_order: list[str]) -> dict[int, np.ndarray]:
    model.eval()
    probs_by_uid: dict[str, list[np.ndarray]] = {}
    for emb, cmap, uids in tqdm(loader, desc="infer", leave=False):
        logits = model(emb.to(DEVICE), cmap.to(DEVICE))
        probs = [torch.sigmoid(x).cpu().numpy() for x in logits]
        for i, uid in enumerate(uids):
            probs_by_uid[uid] = [p[i] for p in probs]
    return {
        level: np.stack([probs_by_uid[uid][level - 1] for uid in uid_order])
        for level in range(1, 5)
    }


def metric_row(model: str, subset: str, level: int, probs: np.ndarray, labels: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {
            "model": model,
            "subset": subset,
            "level": level,
            "n": 0,
            "micro_f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "any_hit_rate": np.nan,
            "avg_pred_labels": np.nan,
        }
    y = labels[mask].astype(np.int32)
    p = (probs[mask] >= threshold).astype(np.int32)
    tp_any = ((y & p).sum(axis=1) > 0).mean()
    return {
        "model": model,
        "subset": subset,
        "level": level,
        "n": n,
        "micro_f1": float(f1_score(y, p, average="micro", zero_division=0)),
        "precision": float(precision_score(y, p, average="micro", zero_division=0)),
        "recall": float(recall_score(y, p, average="micro", zero_division=0)),
        "any_hit_rate": float(tp_any),
        "avg_pred_labels": float(p.sum(axis=1).mean()),
    }


def main() -> None:
    out_dir = ROOT / "outputs" / "audit"
    fig_dir = ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_TEST)
    with open(LABEL_ENC, "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    uids, labels, masks, groups, max_depths = build_hierarchical_labels(df, encoders)

    group_arr = np.array(groups)
    depth_arr = np.array(max_depths)
    subset_masks = {
        "all_prefix_scorable": depth_arr >= 1,
        "known_complete": group_arr == "known_complete",
        "partial": group_arr == "partial",
        "novel_complete": group_arr == "novel_complete",
    }

    ds = TemporalDataset(uids)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)

    rows = []
    for model_label, (model_name, ckpt_rel) in MODELS.items():
        print(f"Evaluating {model_label}")
        ckpt = torch.load(ROOT / ckpt_rel, map_location=DEVICE, weights_only=False)
        model = build_model(model_name, n_classes).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        probs = infer(model, loader, uids)
        for subset_name, subset_mask in subset_masks.items():
            for level in range(1, 5):
                rows.append(
                    metric_row(
                        model_label,
                        subset_name,
                        level,
                        probs[level],
                        labels[level],
                        masks[level] & subset_mask,
                    )
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "partial_ec_hierarchical_metrics.csv", index=False)

    contact = metrics[(metrics["model"] == "Contact-EC") & (metrics["subset"].isin(["all_prefix_scorable", "partial", "novel_complete"]))]
    summary = contact.pivot_table(index="subset", columns="level", values="micro_f1", aggfunc="first")
    summary.columns = [f"L{c}_micro_f1" for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out_dir / "partial_ec_hierarchical_summary.csv", index=False)

    fig_data = metrics[
        (metrics["subset"] == "partial")
        & (metrics["level"].isin([1, 2, 3]))
        & (metrics["model"].isin(["B1 (ESM-2)", "B3 (Contact)", "Contact-EC", "Contact-EC-Hier"]))
    ].copy()
    fig_data["level_name"] = "L" + fig_data["level"].astype(str)
    pivot = fig_data.pivot(index="level_name", columns="model", values="micro_f1").loc[["L1", "L2", "L3"]]
    ax = pivot.plot(kind="bar", figsize=(9, 4.8), width=0.78)
    ax.set_ylabel("Micro F1")
    ax.set_xlabel("Deepest evaluated EC prefix level for partial annotations")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "partial_ec_hierarchical_f1.png", dpi=300)
    plt.close()

    audit = {
        "n_total": int(len(df)),
        "group_counts": {k: int((group_arr == k).sum()) for k in sorted(set(groups))},
        "max_depth_counts": {str(k): int((depth_arr == k).sum()) for k in sorted(set(max_depths))},
        "contact_ec_summary": summary.to_dict(orient="records"),
        "interpretation": [
            "Partial EC annotations are evaluated only up to the deepest available annotated prefix.",
            "These prefix-level scores are not directly comparable to complete Level-4 closed-set F1.",
            "Novel complete EC labels can still be evaluated at lower known EC hierarchy levels when their prefixes exist in the SP-2018 vocabulary.",
        ],
    }
    with open(out_dir / "partial_ec_hierarchical_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    md = [
        "# Partial EC hierarchical prefix audit",
        "",
        "Partial EC labels are scored only up to the deepest available known prefix.",
        "",
        "## Counts",
        "",
        pd.DataFrame([audit["group_counts"]]).to_markdown(index=False),
        "",
        "## Contact-EC summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## All metrics",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f"),
    ]
    (out_dir / "partial_ec_hierarchical_audit.md").write_text("\n".join(md) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
