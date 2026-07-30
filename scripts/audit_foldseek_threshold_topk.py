#!/usr/bin/env python
"""Threshold and top-k audit for Foldseek/TM-score-disjoint EC evaluation.

This script reuses trained checkpoints and does not retrain models. It caches
Level-4 probabilities so threshold sweeps can be rerun without another forward
pass.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Make evaluate.py pick up the same config when imported.
if "--config" not in sys.argv:
    sys.argv.extend(["--config", "config_foldseek.yaml"])

import evaluate as ev  # noqa: E402
from models.dataset import ProteinDataset, collate_fn  # noqa: E402


MODELS = {
    "B1_ESM2_only": {
        "model": "b1_esm2_fc",
        "checkpoint": "outputs/checkpoints/foldseek_tmscore50_b1_ml_best.pt",
        "batch_size": 512,
    },
    "B3_contact_only": {
        "model": "b3_contact",
        "checkpoint": "outputs/checkpoints/foldseek_tmscore50_b3_ml_best.pt",
        "batch_size": 256,
    },
    "Fusion_ESM2_contact": {
        "model": "fusion",
        "checkpoint": "outputs/checkpoints/foldseek_tmscore50_fusion_ml_best.pt",
        "batch_size": 128,
    },
}


def parse_thresholds(text: str) -> list[float]:
    if ":" in text:
        start, stop, step = map(float, text.split(":"))
        vals = []
        cur = start
        while cur <= stop + 1e-12:
            vals.append(round(cur, 6))
            cur += step
        return vals
    return [float(x) for x in text.split(",") if x.strip()]


def micro_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    labels = labels.astype(bool, copy=False)
    preds = preds.astype(bool, copy=False)
    tp_per_class = np.logical_and(labels, preds).sum(axis=0, dtype=np.float64)
    fp_per_class = np.logical_and(~labels, preds).sum(axis=0, dtype=np.float64)
    fn_per_class = np.logical_and(labels, ~preds).sum(axis=0, dtype=np.float64)
    tp = float(tp_per_class.sum())
    fp = float(fp_per_class.sum())
    fn = float(fn_per_class.sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    denom = (2 * tp_per_class + fp_per_class + fn_per_class)
    per_class_f1 = np.divide(
        2 * tp_per_class,
        denom,
        out=np.zeros_like(tp_per_class, dtype=np.float64),
        where=denom > 0,
    )
    return {
        "micro_f1": micro_f1,
        "macro_f1": float(per_class_f1.mean()),
        "precision": precision,
        "recall": recall,
        "avg_pred_labels": float(preds.sum(axis=1).mean()),
        "median_pred_labels": float(np.median(preds.sum(axis=1))),
    }


def topk_predictions(probs: np.ndarray, k: int) -> np.ndarray:
    k = min(k, probs.shape[1])
    idx = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    preds = np.zeros_like(probs, dtype=np.int32)
    rows = np.arange(probs.shape[0])[:, None]
    preds[rows, idx] = 1
    return preds


def topk_hit_rate(labels: np.ndarray, preds: np.ndarray) -> float:
    return float(((labels * preds).sum(axis=1) > 0).mean())


def load_cfg(config: str) -> dict:
    with open(ROOT / "configs" / config) as f:
        return yaml.safe_load(f)


def build_loader(cfg: dict, split: str, batch_size: int, num_workers: int = 0) -> DataLoader:
    ds = ProteinDataset(
        ids_file=ROOT / cfg["paths"]["splits_dir"] / f"{split}_ids.txt",
        meta_csv=ROOT / cfg["paths"]["meta_csv"],
        embed_dir=ROOT / cfg["paths"]["embed_dir"],
        cmap_dir=ROOT / cfg["paths"]["cmap_dir"],
        label_enc_pkl=ROOT / cfg["paths"]["label_enc"],
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def infer_or_load(
    model_label: str,
    spec: dict,
    cfg: dict,
    split: str,
    encoders: dict,
    cache_dir: Path,
    gpu: int,
    force: bool,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{model_label}_{split}_l4_probs_labels.npz"
    if cache_path.exists() and not force:
        data = np.load(cache_path)
        return data["probs"], data["labels"]

    ev.DEVICE = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    model = ev.build_model(spec["model"], n_classes).to(ev.DEVICE)
    ckpt = torch.load(ROOT / spec["checkpoint"], map_location=ev.DEVICE)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)

    loader = build_loader(cfg, split, spec["batch_size"], num_workers=num_workers)
    hier_maps = (
        ev.build_parent_child_matrix(encoders, 1, 2),
        ev.build_parent_child_matrix(encoders, 2, 3),
        ev.build_parent_child_matrix(encoders, 3, 4),
    )
    _, _, masks_all, probs, labels = ev.run_inference(
        model, loader, spec["model"], hier_maps=hier_maps
    )
    valid_mask = np.array([m[3] == 1 for m in masks_all])
    probs = probs[valid_mask].astype(np.float32)
    labels = labels[valid_mask].astype(np.int32)
    np.savez_compressed(cache_path, probs=probs, labels=labels)
    return probs, labels


def rare_class_indices(cfg: dict, split_prefix: str, max_count: int = 25) -> np.ndarray:
    meta = pd.read_csv(ROOT / cfg["paths"]["meta_csv"])
    train_ids = set(
        (ROOT / cfg["paths"]["splits_dir"] / f"{split_prefix}train_ids.txt")
        .read_text()
        .splitlines()
    )
    counts = meta[meta["accession"].isin(train_ids)]["l4_idx"].value_counts()
    return counts[counts <= max_count].index.to_numpy(dtype=np.int64)


def run_audit(args: argparse.Namespace) -> dict:
    cfg = load_cfg(args.config)
    with open(ROOT / cfg["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)

    thresholds = parse_thresholds(args.thresholds)
    topks = [int(x) for x in args.topk.split(",") if x.strip()]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / args.cache_dir
    rare_idx = rare_class_indices(cfg, args.split_prefix, args.rare_max_count)

    rows = []
    best_rows = []
    topk_rows = []
    rare_rows = []

    for model_label, spec in MODELS.items():
        print(f"[audit] {model_label}: inference/cache loading", flush=True)
        probs, labels = infer_or_load(
            model_label=model_label,
            spec=spec,
            cfg=cfg,
            split=args.split,
            encoders=encoders,
            cache_dir=cache_dir,
            gpu=args.gpu,
            force=args.force_infer,
            num_workers=args.num_workers,
        )
        print(f"[audit] {model_label}: probs={probs.shape}, labels={labels.shape}", flush=True)

        for thr in thresholds:
            preds = (probs >= thr).astype(np.int32)
            m = micro_metrics(labels, preds)
            row = {
                "model": model_label,
                "mode": "threshold",
                "threshold": thr,
                "top_k": "",
                "n": int(labels.shape[0]),
                **m,
            }
            rows.append(row)

            if len(rare_idx):
                valid_rare = rare_idx[rare_idx < labels.shape[1]]
                rare_preds = preds[:, valid_rare]
                rare_labels = labels[:, valid_rare]
                rare = micro_metrics(rare_labels, rare_preds)
                rare_rows.append(
                    {
                        "model": model_label,
                        "threshold": thr,
                        "n_rare_classes": int(len(valid_rare)),
                        **rare,
                    }
                )

        model_rows = [r for r in rows if r["model"] == model_label and r["mode"] == "threshold"]
        best = max(model_rows, key=lambda r: r["micro_f1"])
        best_rows.append(best)
        print(
            f"[audit] {model_label}: best threshold={best['threshold']:.3f}, "
            f"micro_f1={best['micro_f1']:.4f}",
            flush=True,
        )

        for k in topks:
            preds = topk_predictions(probs, k)
            m = micro_metrics(labels, preds)
            topk_rows.append(
                {
                    "model": model_label,
                    "mode": "topk",
                    "threshold": "",
                    "top_k": k,
                    "n": int(labels.shape[0]),
                    "hit_rate": topk_hit_rate(labels, preds),
                    **m,
                }
            )

    sweep_df = pd.DataFrame(rows)
    best_df = pd.DataFrame(best_rows)
    topk_df = pd.DataFrame(topk_rows)
    rare_df = pd.DataFrame(rare_rows)

    prefix = args.output_prefix
    sweep_df.to_csv(out_dir / f"{prefix}_threshold_sweep.csv", index=False)
    best_df.to_csv(out_dir / f"{prefix}_best_threshold.csv", index=False)
    topk_df.to_csv(out_dir / f"{prefix}_topk.csv", index=False)
    rare_df.to_csv(out_dir / f"{prefix}_rare_threshold_sweep.csv", index=False)

    payload = {
        "split": args.split,
        "split_prefix": args.split_prefix,
        "thresholds": thresholds,
        "topk": topks,
        "rare_max_count": args.rare_max_count,
        "rare_class_count": int(len(rare_idx)),
        "best_threshold": best_df.to_dict(orient="records"),
        "topk": topk_df.to_dict(orient="records"),
    }
    (out_dir / f"{prefix}_threshold_topk_audit.json").write_text(json.dumps(payload, indent=2))

    md = [
        "# Foldseek Threshold and Top-k Audit",
        "",
        f"- Split: `{args.split}`",
        f"- Foldseek train prefix for rare-class counts: `{args.split_prefix}`",
        f"- Rare-class cutoff: <= {args.rare_max_count} train proteins",
        f"- Rare Level-4 classes: {len(rare_idx)}",
        "",
        "## Best Threshold by Level-4 Micro F1",
        "",
        "| Model | Best threshold | Micro F1 | Macro F1 | Precision | Recall | Avg predicted labels |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in best_df.iterrows():
        md.append(
            f"| {r['model']} | {r['threshold']:.3f} | {r['micro_f1']:.4f} | "
            f"{r['macro_f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['avg_pred_labels']:.2f} |"
        )
    md.extend(
        [
            "",
            "## Top-k Evaluation",
            "",
            "| Model | k | Micro F1 | Macro F1 | Precision | Recall | Hit rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in topk_df.iterrows():
        md.append(
            f"| {r['model']} | {int(r['top_k'])} | {r['micro_f1']:.4f} | "
            f"{r['macro_f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['hit_rate']:.4f} |"
        )
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "The default threshold of 0.5 is conservative on the Foldseek-disjoint split. "
            "The best-threshold and top-k views distinguish model ranking ability from "
            "fixed-threshold calibration. These values should be reported as diagnostic "
            "analyses unless the threshold is selected on a held-out validation split.",
        ]
    )
    (out_dir / f"{prefix}_threshold_topk_audit.md").write_text("\n".join(md) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_foldseek.yaml")
    parser.add_argument("--split", default="foldseek_tmscore50_cc_test")
    parser.add_argument("--split-prefix", default="foldseek_tmscore50_cc_")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--thresholds", default="0.01:0.90:0.01")
    parser.add_argument("--topk", default="1,3,5,10")
    parser.add_argument("--rare-max-count", type=int, default=25)
    parser.add_argument("--cache-dir", default="outputs/cache/foldseek_threshold_topk")
    parser.add_argument("--out-dir", default="outputs/audit")
    parser.add_argument("--output-prefix", default="foldseek_tmscore50_cc")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-infer", action="store_true")
    args = parser.parse_args()
    payload = run_audit(args)
    print(json.dumps(payload["best_threshold"], indent=2))


if __name__ == "__main__":
    main()
