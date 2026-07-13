"""
Case-wise comparison between HIT-EC and Contact-EC on the 124 known-label
Swiss-Prot 2023-01 temporal proteins.

Outputs:
  outputs/results/casewise_hitec_contactec.csv
  outputs/results/casewise_hitec_contactec_summary.json
  outputs/results/casewise_hitec_contactec_summary.md
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.dataset import ProteinDataset, collate_fn


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_cfg():
    with open(ROOT / "configs" / "config_ecbench.yaml") as f:
        return yaml.safe_load(f)


def build_contact_ec_flat(n_classes, cfg):
    from models.fusion_v2_flatfc import FusionV2FlatFC

    return FusionV2FlatFC(
        n_classes,
        esm_dim=cfg["model"]["esm2_dim"],
        contact_dim=cfg["model"]["resnet_out_dim"],
        fusion_dim=cfg["model"]["fusion_dim"],
        dropout=0.0,
    )


@torch.no_grad()
def infer_contact_ec(model, loader):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="Contact-EC inference"):
        esm_emb, cmap, _, _, _, l4_mh, accs = batch
        logits = model(esm_emb.to(DEVICE), cmap.to(DEVICE))
        probs = torch.sigmoid(logits[3]).cpu().numpy()
        labels = l4_mh.numpy().astype(np.int32)
        for acc, prob, lab in zip(accs, probs, labels):
            rows.append((acc, prob, lab))
    return rows


def ecs_from_vector(vec, classes):
    idxs = np.flatnonzero(vec)
    return [str(classes[i]) for i in idxs]


def top_ecs(prob, classes, k=5):
    idxs = np.argsort(prob)[::-1][:k]
    return [(str(classes[i]), float(prob[i])) for i in idxs]


def per_protein_f1(true_ecs, pred_ecs):
    true_set = set(true_ecs)
    pred_set = set(pred_ecs)
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    return 2 * tp / (len(true_set) + len(pred_set))


def status(hit, contact):
    if hit and contact:
        return "both_correct"
    if hit and not contact:
        return "hitec_only"
    if contact and not hit:
        return "contactec_only"
    return "both_wrong"


def main():
    cfg = load_cfg()
    with open(ROOT / cfg["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    l4_classes = encoders["level4"].classes_
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]

    ds = ProteinDataset(
        ids_file=str(ROOT / cfg["paths"]["splits_dir"] / "test_ids_full.txt"),
        meta_csv=str(ROOT / "data" / "ecbench" / "processed" / "test_meta_full.csv"),
        embed_dir=str(ROOT / cfg["paths"]["embed_dir"]),
        cmap_dir=str(ROOT / cfg["paths"]["cmap_dir"]),
        label_enc_pkl=str(ROOT / cfg["paths"]["label_enc"]),
    )
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = build_contact_ec_flat(n_classes, cfg).to(DEVICE)
    ckpt_path = ROOT / "outputs" / "checkpoints" / "ecbench_b4_flatfc_best.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])

    contact_rows = infer_contact_ec(model, loader)
    contact_by_uid = {}
    all_contact_probs = []
    all_labels = []
    for uid, prob, lab in contact_rows:
        pred_mask = (prob >= 0.5).astype(np.int32)
        true_ecs = ecs_from_vector(lab, l4_classes)
        pred_ecs = ecs_from_vector(pred_mask, l4_classes)
        contact_by_uid[uid] = {
            "true_ecs": true_ecs,
            "pred_ecs": pred_ecs,
            "top5": top_ecs(prob, l4_classes, 5),
            "f1": per_protein_f1(true_ecs, pred_ecs),
            "hit": len(set(true_ecs) & set(pred_ecs)) > 0,
        }
        all_contact_probs.append(pred_mask)
        all_labels.append(lab)

    hitec = json.load(open(ROOT / "outputs" / "results" / "hitec_eval.json"))
    hitec_by_uid = {r["uid"]: r for r in hitec["per_protein"]}

    meta = pd.read_csv(ROOT / "data" / "ecbench" / "processed" / "test_meta_full.csv")
    meta_by_uid = meta.set_index("accession").to_dict(orient="index")

    rows = []
    for uid in [x[0] for x in contact_rows]:
        c = contact_by_uid[uid]
        h = hitec_by_uid.get(uid, {})
        true_ecs = c["true_ecs"]
        h_pred = [str(x) for x in h.get("pred_ecs", [])]
        h_hit = len(set(true_ecs) & set(h_pred)) > 0
        c_hit = c["hit"]
        st = status(h_hit, c_hit)
        m = meta_by_uid.get(uid, {})
        rows.append({
            "uid": uid,
            "seq_len": int(m.get("seq_len", -1)),
            "true_ecs": ";".join(true_ecs),
            "true_l1": ";".join(sorted({e.split(".")[0] for e in true_ecs})),
            "hitec_pred_ecs": ";".join(h_pred),
            "hitec_top1_prob": h.get("top1_prob", None),
            "hitec_per_protein_f1": per_protein_f1(true_ecs, h_pred),
            "hitec_hit": h_hit,
            "contactec_pred_ecs": ";".join(c["pred_ecs"]),
            "contactec_top5": ";".join([f"{e}:{p:.4f}" for e, p in c["top5"]]),
            "contactec_per_protein_f1": c["f1"],
            "contactec_hit": c_hit,
            "case_status": st,
        })

    out_dir = ROOT / "outputs" / "results"
    out_csv = out_dir / "casewise_hitec_contactec.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    y_true = np.stack(all_labels).astype(np.int32)
    y_pred_contact = np.stack(all_contact_probs).astype(np.int32)
    contact_micro = float(f1_score(y_true, y_pred_contact, average="micro", zero_division=0))

    status_counts = df["case_status"].value_counts().to_dict()
    l1_status = (
        df.groupby(["true_l1", "case_status"]).size().unstack(fill_value=0).reset_index()
        .to_dict(orient="records")
    )
    examples = {}
    for st in ["contactec_only", "hitec_only", "both_correct", "both_wrong"]:
        sub = df[df["case_status"] == st].copy()
        if st == "contactec_only":
            sub = sub.sort_values(["contactec_per_protein_f1", "hitec_top1_prob"], ascending=[False, True])
        elif st == "hitec_only":
            sub = sub.sort_values(["hitec_per_protein_f1", "contactec_per_protein_f1"], ascending=[False, True])
        elif st == "both_wrong":
            sub = sub.sort_values(["hitec_top1_prob"], ascending=[False])
        else:
            sub = sub.sort_values(["contactec_per_protein_f1", "hitec_per_protein_f1"], ascending=False)
        examples[st] = sub.head(5).to_dict(orient="records")

    summary = {
        "n": int(len(df)),
        "contact_ec_checkpoint": str(ckpt_path),
        "contact_ec_threshold": 0.5,
        "hitec_threshold": hitec.get("threshold"),
        "contact_ec_micro_f1_recomputed": round(contact_micro, 4),
        "hitec_micro_f1_reported": hitec.get("results", {}).get("micro_f1"),
        "case_status_counts": {k: int(v) for k, v in status_counts.items()},
        "l1_status_counts": l1_status,
        "examples": examples,
    }
    out_json = out_dir / "casewise_hitec_contactec_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    out_md = out_dir / "casewise_hitec_contactec_summary.md"
    lines = [
        "# HIT-EC vs Contact-EC Case-wise Comparison",
        "",
        f"N = {summary['n']} known-label Swiss-Prot 2023-01 proteins.",
        f"Contact-EC recomputed micro F1 = {summary['contact_ec_micro_f1_recomputed']:.4f}.",
        f"HIT-EC reported micro F1 = {summary['hitec_micro_f1_reported']:.4f}.",
        "",
        "## Case Status Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for k in ["both_correct", "hitec_only", "contactec_only", "both_wrong"]:
        lines.append(f"| {k} | {summary['case_status_counts'].get(k, 0)} |")
    lines += ["", "## Level-1 Status Counts", ""]
    l1_df = pd.DataFrame(l1_status).fillna(0)
    if not l1_df.empty:
        lines.append(l1_df.to_markdown(index=False))
    lines += ["", "## Representative Contact-EC-only Cases", ""]
    for ex in examples["contactec_only"][:5]:
        lines.append(
            f"- {ex['uid']}: true={ex['true_ecs']}; "
            f"Contact-EC={ex['contactec_pred_ecs']}; HIT-EC={ex['hitec_pred_ecs']}"
        )
    lines += ["", "## Representative HIT-EC-only Cases", ""]
    for ex in examples["hitec_only"][:5]:
        lines.append(
            f"- {ex['uid']}: true={ex['true_ecs']}; "
            f"HIT-EC={ex['hitec_pred_ecs']}; Contact-EC top5={ex['contactec_top5']}"
        )
    out_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(json.dumps(summary["case_status_counts"], indent=2))


if __name__ == "__main__":
    main()
