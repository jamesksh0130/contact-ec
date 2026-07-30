#!/usr/bin/env python
"""Summarize repeated model runs on the Foldseek/TM-score-disjoint split."""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_foldseek_threshold_topk import (  # noqa: E402
    infer_or_load,
    load_cfg,
    micro_metrics,
    parse_thresholds,
    topk_hit_rate,
    topk_predictions,
)


def discover_checkpoints(pattern: str) -> list[tuple[str, Path]]:
    ckpt_dir = ROOT / "outputs" / "checkpoints"
    paths = sorted(ckpt_dir.glob(pattern))
    items: list[tuple[str, Path]] = []
    for path in paths:
        stem = path.name.replace("_best.pt", "")
        match = re.search(r"seed(\d+)", stem)
        label = f"seed{match.group(1)}" if match else "seed42_base"
        items.append((label, path))
    return items


def row_stats(rows: list[dict], keys: list[str]) -> dict:
    out = {"n_runs": len(rows)}
    for key in keys:
        vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean()) if len(vals) else 0.0
        out[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_foldseek.yaml")
    parser.add_argument("--split", default="foldseek_tmscore50_cc_test")
    parser.add_argument("--model", default="fusion")
    parser.add_argument("--model-label", default="Fusion_ESM2_contact")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint-pattern", default="foldseek_tmscore50_fusion_ml*_best.pt")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--thresholds", default="0.01:0.90:0.01")
    parser.add_argument("--val-threshold", type=float, default=0.08)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--topk", default="1,3,5,10")
    parser.add_argument("--out-dir", default="outputs/audit")
    parser.add_argument("--cache-dir", default="outputs/cache/foldseek_seed_repeats")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-infer", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    with open(ROOT / cfg["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)

    ckpts = discover_checkpoints(args.checkpoint_pattern)
    if not ckpts:
        raise SystemExit(f"No checkpoints matched outputs/checkpoints/{args.checkpoint_pattern}")

    thresholds = parse_thresholds(args.thresholds)
    topks = [int(x) for x in args.topk.split(",") if x.strip()]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_rows = []
    val_rows = []
    best_rows = []
    topk_rows = []

    for run_label, ckpt_path in ckpts:
        spec = {
            "model": args.model,
            "checkpoint": str(ckpt_path.relative_to(ROOT)),
            "batch_size": args.batch_size,
        }
        model_label = f"{args.model_label}_{run_label}"
        print(f"[seed-audit] {model_label}: {spec['checkpoint']}", flush=True)
        probs, labels = infer_or_load(
            model_label=model_label,
            spec=spec,
            cfg=cfg,
            split=args.split,
            encoders=encoders,
            cache_dir=ROOT / args.cache_dir,
            gpu=args.gpu,
            force=args.force_infer,
            num_workers=args.num_workers,
        )

        fixed = micro_metrics(labels, (probs >= 0.5).astype(np.int32))
        fixed_rows.append({"run": run_label, "threshold": 0.5, "n": int(labels.shape[0]), **fixed})

        val = micro_metrics(labels, (probs >= args.val_threshold).astype(np.int32))
        val_rows.append(
            {"run": run_label, "threshold": args.val_threshold, "n": int(labels.shape[0]), **val}
        )

        sweep = []
        for thr in thresholds:
            sweep.append({"threshold": thr, **micro_metrics(labels, (probs >= thr).astype(np.int32))})
        best = max(sweep, key=lambda r: r["micro_f1"])
        best_rows.append({"run": run_label, "n": int(labels.shape[0]), **best})

        for k in topks:
            preds = topk_predictions(probs, k)
            topk_rows.append(
                {
                    "run": run_label,
                    "top_k": k,
                    "n": int(labels.shape[0]),
                    "hit_rate": topk_hit_rate(labels, preds),
                    **micro_metrics(labels, preds),
                }
            )

    prefix = args.output_prefix or f"foldseek_tmscore50_cc_{args.model_label}_seed_repeats"
    pd.DataFrame(fixed_rows).to_csv(out_dir / f"{prefix}_fixed_threshold.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out_dir / f"{prefix}_val_threshold.csv", index=False)
    pd.DataFrame(best_rows).to_csv(out_dir / f"{prefix}_best_threshold.csv", index=False)
    pd.DataFrame(topk_rows).to_csv(out_dir / f"{prefix}_topk.csv", index=False)

    metric_keys = ["micro_f1", "macro_f1", "precision", "recall", "avg_pred_labels"]
    summary = {
        "split": args.split,
        "checkpoints": [str(p.relative_to(ROOT)) for _, p in ckpts],
        "fixed_threshold_0_5": row_stats(fixed_rows, metric_keys),
        "validation_threshold": {"threshold": args.val_threshold, **row_stats(val_rows, metric_keys)},
        "posthoc_best_threshold": row_stats(best_rows, metric_keys),
        "topk": {},
    }
    for k in topks:
        rows_k = [r for r in topk_rows if int(r["top_k"]) == k]
        summary["topk"][str(k)] = row_stats(rows_k, ["micro_f1", "macro_f1", "hit_rate"])

    (out_dir / f"{prefix}.json").write_text(json.dumps(summary, indent=2))

    md = [
        "# Foldseek Seed Repeat Audit",
        "",
        f"- Model: `{args.model}`",
        f"- Model label: `{args.model_label}`",
        f"- Split: `{args.split}`",
        f"- Runs: {len(ckpts)}",
        f"- Validation-selected threshold used for test audit: {args.val_threshold:.3f}",
        "",
        "## Summary",
        "",
        "| Mode | n runs | Micro F1 mean | Micro F1 sd | Precision mean | Recall mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, stats in [
        ("Fixed threshold 0.5", summary["fixed_threshold_0_5"]),
        (f"Validation threshold {args.val_threshold:.2f}", summary["validation_threshold"]),
        ("Post-hoc best threshold", summary["posthoc_best_threshold"]),
    ]:
        md.append(
            f"| {mode} | {stats['n_runs']} | {stats['micro_f1_mean']:.4f} | "
            f"{stats['micro_f1_std']:.4f} | {stats['precision_mean']:.4f} | "
            f"{stats['recall_mean']:.4f} |"
        )
    md.extend(
        [
            "",
            "## Individual Runs",
            "",
            "### Validation Threshold",
            "",
            pd.DataFrame(val_rows).to_markdown(index=False),
            "",
            "### Post-hoc Best Threshold",
            "",
            pd.DataFrame(best_rows).to_markdown(index=False),
            "",
        ]
    )
    (out_dir / f"{prefix}.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
