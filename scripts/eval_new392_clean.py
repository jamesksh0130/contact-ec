"""
Exp 3: New-392 clean evaluation (16개 EC-Bench train 오염 제거)

EC-Bench 훈련 데이터와 16개 중복 → 376개 clean subset으로 재평가.
per-protein F1, micro F1 모두 계산.
"""
import sys, pickle, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config_ecbench.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import _make_3ch_cmap

DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"
EMBED_DIR = ROOT / CFG["paths"]["embed_dir"]
CMAP_DIR  = ROOT / CFG["paths"]["cmap_dir"]
LABEL_ENC = ROOT / CFG["paths"]["label_enc"]
NEW392_CSV= ROOT / "data" / "new392" / "new392_ec_labels.csv"
TRAIN_IDS = ROOT / "data" / "ecbench" / "splits" / "train_ids.txt"

MODELS = {
    "B1 (ESM-2)":  ("b1_esm2_fc",  "outputs/checkpoints/ecbench_b1_best.pt"),
    "B3 (Contact)":("b3_contact",   "outputs/checkpoints/ecbench_b3_phase1_best.pt"),
    "Contact-EC":  ("fusion_v2",    "outputs/checkpoints/ecbench_fv2_phase2_best.pt"),
}
CLEAN_REF = {"CLEAN-Contact": 0.566, "CLEAN": 0.504, "ProteInfer": 0.309}


class New392Dataset(Dataset):
    def __init__(self, csv_path, l4_enc, exclude_ids=None):
        df      = pd.read_csv(csv_path, sep="\t")
        l4_set  = set(l4_enc.classes_)
        n_l4    = len(l4_enc.classes_)
        exclude = set(exclude_ids or [])

        self.rows, self.labels = [], []
        skipped_contaminated = 0
        for _, row in df.iterrows():
            uid = row["Entry"]
            if uid in exclude:
                skipped_contaminated += 1
                continue
            if not (EMBED_DIR / f"{uid}.npy").exists():
                continue
            ecs   = [e.strip() for e in str(row["EC number"])
                     .replace(";", ",").split(",")]
            valid = [e for e in ecs if e in l4_set]
            if not valid:
                continue
            mh = np.zeros(n_l4, dtype=np.float32)
            for e in valid:
                mh[int(np.where(l4_enc.classes_ == e)[0][0])] = 1.0
            self.rows.append(uid)
            self.labels.append(mh)

        print(f"  유효: {len(self.rows)}/{len(df)}  "
              f"오염 제거: {skipped_contaminated}개")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        uid  = self.rows[idx]
        emb  = np.load(EMBED_DIR / f"{uid}.npy").astype(np.float32)
        cp   = CMAP_DIR / f"{uid}.npy"
        if cp.exists():
            raw  = np.load(cp).astype(np.float32)
            cmap = _make_3ch_cmap(raw) if raw.shape == (256, 256) \
                   else np.zeros((3, 256, 256), dtype=np.float32)
        else:
            cmap = np.zeros((3, 256, 256), dtype=np.float32)
        return torch.tensor(emb), torch.tensor(cmap), \
               torch.tensor(self.labels[idx]), uid


def collate_fn(batch):
    embs, cmaps, labels, uids = zip(*batch)
    return torch.stack(embs), torch.stack(cmaps), torch.stack(labels), list(uids)


def build_model(name, n_classes):
    if name == "b1_esm2_fc":
        from models.esm2_fc import ESM2FC
        return ESM2FC(n_classes, esm_dim=CFG["model"]["esm2_dim"], dropout=0.0)
    elif name == "b3_contact":
        from models.contact_resnet import ContactResNet
        return ContactResNet(n_classes, dropout=0.0)
    elif name == "fusion_v2":
        from models.fusion_v2 import FusionModelV2
        return FusionModelV2(n_classes, esm_dim=CFG["model"]["esm2_dim"],
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"], dropout=0.0)


def per_protein_f1(probs, labels, thr=0.5):
    scores = []
    for p, gt in zip(probs, labels):
        pred = (p >= thr).astype(float)
        tp   = float((pred * gt).sum())
        np_  = float(pred.sum())
        ng   = float(gt.sum())
        if np_ == 0 and ng == 0:
            scores.append(1.0)
        elif np_ == 0 or ng == 0:
            scores.append(0.0)
        else:
            prec = tp / np_; rec = tp / ng
            scores.append(2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0)
    return float(np.mean(scores))


@torch.no_grad()
def run_inference(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for emb, cmap, labels, _ in tqdm(loader, desc="  추론", leave=False):
        logits = model(emb.to(DEVICE), cmap.to(DEVICE))
        all_probs.append(torch.sigmoid(logits[3]).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    # EC-Bench train과 중복되는 New-392 IDs
    train_ids = set(TRAIN_IDS.read_text().splitlines())
    df392     = pd.read_csv(NEW392_CSV, sep="\t")
    contaminated = [r["Entry"] for _, r in df392.iterrows()
                    if r["Entry"] in train_ids]
    print(f"EC-Bench train 오염: {len(contaminated)}/392개 → 제거 후 {392-len(contaminated)}개 clean")
    print(f"오염 IDs: {contaminated}")

    with open(LABEL_ENC, "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    l4_enc    = encoders["level4"]

    print("\n[New-392 Clean Dataset]")
    ds     = New392Dataset(NEW392_CSV, l4_enc, exclude_ids=contaminated)
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    results = {}
    for label, (mname, ckpt_path) in MODELS.items():
        print(f"\n  [{label}]")
        ckpt  = torch.load(ROOT / ckpt_path, map_location=DEVICE, weights_only=False)
        model = build_model(mname, n_classes).to(DEVICE)
        model.load_state_dict(ckpt["model"])

        probs, gts = run_inference(model, loader)
        preds      = (probs >= 0.5).astype(np.int32)
        micro_f1   = float(f1_score(gts, preds, average="micro", zero_division=0))
        pp_f1      = per_protein_f1(probs, gts)

        print(f"    Micro F1={micro_f1:.4f}  Per-protein F1={pp_f1:.4f}")
        for ref_name, ref_val in CLEAN_REF.items():
            print(f"      vs {ref_name}: {ref_val:.3f}  (Δ={pp_f1-ref_val:+.4f})")

        results[label] = {"micro_f1": round(micro_f1, 4),
                          "per_protein_f1": round(pp_f1, 4)}
        del model; torch.cuda.empty_cache()

    print(f"\n{'='*55}")
    print(f"  New-392 Clean (N=376) vs Full (N=392) 비교")
    print(f"{'='*55}")
    full_results = {"B1 (ESM-2)": 0.1701, "B3 (Contact)": 0.1539, "Contact-EC": 0.3580}
    print(f"  {'모델':<18}  {'Full 392':>10}  {'Clean 376':>10}  {'차이':>8}")
    for lbl, r in results.items():
        full = full_results.get(lbl, 0)
        diff = r["per_protein_f1"] - full
        print(f"  {lbl:<18}  {full:>10.4f}  {r['per_protein_f1']:>10.4f}  {diff:>+8.4f}")

    out = ROOT / "outputs" / "results" / "new392_clean_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"n_clean": len(ds), "contaminated": contaminated,
                   "results": results}, f, indent=2)
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
