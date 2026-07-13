"""
FlatFC E2E 모델 Temporal Test 평가
- Swiss-Prot 2023-01 (N=124)
- Price-149 (N=136)
- forward() 사용 (전체 ESM-2 on-the-fly, 캐시 없음)
"""
import sys, pickle, argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.dataset import _make_3ch_cmap

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5
RARE_MAX  = 25


class SeqCmapDataset(Dataset):
    def __init__(self, ids_file, meta_csv, cmap_dir, cmap_size=256, enc=None):
        self.ids      = Path(ids_file).read_text().strip().split("\n")
        meta          = pd.read_csv(meta_csv).set_index("accession")
        self.meta     = meta
        self.cmap_dir = Path(cmap_dir)
        self.cmap_size= cmap_size
        # Build l4 map from ExpA encoder for re-encoding raw EC strings
        if enc is not None:
            l4_classes = list(enc["level4"].classes_)
            self.l4_map = {c: i for i, c in enumerate(l4_classes)}
        else:
            self.l4_map = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        row = self.meta.loc[uid] if uid in self.meta.index else None

        seq = row["sequence"] if row is not None else "M"

        cmap_f = self.cmap_dir / f"{uid}.npy"
        if cmap_f.exists():
            raw  = np.load(cmap_f).astype(np.float32)
            cmap = _make_3ch_cmap(raw, self.cmap_size)
        else:
            cmap = np.zeros((3, self.cmap_size, self.cmap_size), dtype=np.float32)

        # Encode l4 label: prefer raw EC re-encoding via ExpA map
        l4_idx = -1
        if row is not None:
            if self.l4_map is not None:
                ec_raw = str(row.get("ec_raw", ""))
                l4_idx = self.l4_map.get(ec_raw, -1)
            else:
                v = row.get("l4_idx", -1)
                l4_idx = int(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else -1

        labels = [-1, -1, -1, l4_idx]
        masks  = [0.0, 0.0, 0.0, 1.0 if l4_idx >= 0 else 0.0]
        return uid, seq, torch.tensor(cmap, dtype=torch.float32), labels, masks


def collate_fn(batch):
    uids, seqs, cmaps, labels, masks = zip(*batch)
    return list(uids), list(seqs), torch.stack(cmaps), labels, masks


def get_n_l4_train(enc_pkl):
    with open(enc_pkl, "rb") as f:
        enc = pickle.load(f)
    return len(enc["level4"].classes_), enc


def get_rare_mask(enc_pkl, meta_train_csv, n_l4, max_samples=RARE_MAX):
    with open(enc_pkl, "rb") as f:
        enc = pickle.load(f)
    meta = pd.read_csv(meta_train_csv)
    counts = np.zeros(n_l4, dtype=int)
    for v in meta["l4_idx"].dropna():
        idx = int(v)
        if 0 <= idx < n_l4:
            counts[idx] += 1
    return counts <= max_samples


@torch.no_grad()
def evaluate(model, loader, n_l4, rare_mask):
    model.eval()
    all_probs, all_mh = [], []

    for uids, seqs, cmaps, labels_batch, masks_batch in loader:
        cmaps = cmaps.to(DEVICE)
        logits = model(seqs, cmaps)
        probs  = torch.sigmoid(logits[3]).cpu().numpy()
        B = len(seqs)
        mh = np.zeros((B, n_l4))
        for i, (lbls, msks) in enumerate(zip(labels_batch, masks_batch)):
            if msks[3] > 0 and lbls[3] >= 0:
                mh[i, lbls[3]] = 1.0
        all_probs.append(probs)
        all_mh.append(mh)

    probs = np.concatenate(all_probs)
    mh    = np.concatenate(all_mh)
    preds = (probs >= THRESHOLD).astype(int)

    micro    = f1_score(mh, preds, average="micro",    zero_division=0)
    weighted = f1_score(mh, preds, average="weighted", zero_division=0)

    # Rare-EC F1 (only on classes present in test)
    rare_cols = np.where(rare_mask & (mh.sum(0) > 0))[0]
    if len(rare_cols) > 0:
        rare_f1 = f1_score(mh[:, rare_cols], preds[:, rare_cols],
                           average="micro", zero_division=0)
    else:
        rare_f1 = 0.0

    return micro, weighted, rare_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/flatfc_e2e_cached_best.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    enc_pkl      = ROOT / "data/expa/label_encoders.pkl"
    train_meta   = ROOT / "data/expa/dataset_meta_reenc.csv"
    cmap_dir     = ROOT / "data/processed/contact_maps"
    test_meta    = ROOT / "data/ecbench/processed/test_meta_full_expa_enc.csv"
    test_ids     = ROOT / "data/ecbench/splits/test_ids_full.txt"
    price_meta   = ROOT / "data/ecbench/processed/price149_meta.csv"
    price_ids    = ROOT / "data/ecbench/splits/price149_ids.txt"

    n_l4, enc = get_n_l4_train(enc_pkl)
    n_classes  = [len(enc[f"level{i}"].classes_) for i in range(1, 5)]
    rare_mask  = get_rare_mask(enc_pkl, train_meta, n_l4)
    print(f"n_classes: {n_classes}, rare classes (≤{RARE_MAX}): {rare_mask.sum()}")

    # Load model
    from models.fusion_flatfc_esm_ft import FusionFlatFCESMFt
    ckpt_path = ROOT / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model = FusionFlatFCESMFt(n_classes).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"체크포인트 로드: {ckpt_path.name}  (val micro_f1={ckpt.get('micro_f1', '?')})")

    # ── Temporal test (N=124) ──────────────────────────────────────────────────
    test_ds = SeqCmapDataset(test_ids, test_meta, cmap_dir, enc=enc)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=2)
    micro, weighted, rare_f1 = evaluate(model, test_loader, n_l4, rare_mask)
    print(f"\n[Swiss-Prot 2023-01 (N=124)]")
    print(f"  Micro F1   : {micro:.4f}")
    print(f"  Weighted F1: {weighted:.4f}")
    print(f"  Rare-EC F1 : {rare_f1:.4f}")

    # ── Price-149 ──────────────────────────────────────────────────────────────
    if price_meta.exists() and price_ids.exists():
        price_ds = SeqCmapDataset(price_ids, price_meta, cmap_dir, enc=enc)
        price_loader = DataLoader(price_ds, batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_fn, num_workers=2)
        p_micro, p_weighted, _ = evaluate(model, price_loader, n_l4, rare_mask)
        print(f"\n[Price-149]")
        print(f"  Micro F1   : {p_micro:.4f}")
        print(f"  Weighted F1: {p_weighted:.4f}")
    else:
        print("\n[Price-149] 데이터 없음, 스킵")


if __name__ == "__main__":
    main()
