"""
FlatFC + ESM-2 E2E 재실험 (Layer26 캐시 활용)

목적: FusionV2FlatFC (expa_flatfc_phase2_best.pt)에서
      ESM-2 마지막 6층을 unfreeze하여 E2E fine-tuning.
      동일 아키텍처(FlatFC + binary BCE)로 공정한 E2E 비교.

사용법:
  python scripts/train_flatfc_e2e_cached.py \
    --flatfc_ckpt outputs/checkpoints/expa_flatfc_phase2_best.pt \
    --epochs 10 --batch_size 16 --grad_accum 2 \
    --lr_esm 1e-5 --lr_rest 1e-4 --fp16 --tag flatfc_e2e_cached
"""
import argparse, os, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.dataset import _make_3ch_cmap
from train import (hierarchical_loss, build_parent_child_matrix,
                   consistency_loss, compute_metrics_ml)

_CONS_WEIGHT = 0.05


# ── Layer26 Dataset ──────────────────────────────────────────────────────────
class Layer26Dataset(Dataset):
    def __init__(self, ids_file, meta_csv, cmap_dir, cache_dir,
                 label_enc_pkl, cmap_size=256):
        import pandas as pd
        self.ids      = Path(ids_file).read_text().strip().split("\n")
        meta          = pd.read_csv(meta_csv).set_index("accession")
        self.meta     = meta
        self.cmap_dir = Path(cmap_dir)
        self.cache_dir= Path(cache_dir)
        self.cmap_size= cmap_size

        with open(label_enc_pkl, "rb") as f:
            self.enc = pickle.load(f)
        self.n_l4 = len(self.enc["level4"].classes_)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid  = self.ids[idx]
        row  = self.meta.loc[uid]

        # Layer26 cache (fp16 → fp32)
        cache_f = self.cache_dir / f"{uid}.npy"
        h26     = np.load(cache_f).astype(np.float32)   # (seq_len, 1280)
        seq_len = h26.shape[0]

        # Contact map (3채널)
        cmap_f = self.cmap_dir / f"{uid}.npy"
        if cmap_f.exists():
            raw  = np.load(cmap_f).astype(np.float32)
            cmap = (_make_3ch_cmap(raw, self.cmap_size)
                    if raw.shape == (self.cmap_size, self.cmap_size)
                    else np.zeros((3, self.cmap_size, self.cmap_size), dtype=np.float32))
        else:
            cmap = np.zeros((3, self.cmap_size, self.cmap_size), dtype=np.float32)

        # Labels
        def get_idx(col):
            v = row.get(col, -1)
            return int(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else -1

        labels = torch.tensor([get_idx(f"l{i}_idx") for i in range(1, 5)],
                              dtype=torch.long)
        masks  = (labels >= 0).float()

        # L4 multihot
        l4_mh = torch.zeros(self.n_l4)
        raw   = row.get("l4_all_idxs", None)
        if raw is not None and str(raw) not in ("nan", "None", ""):
            import ast
            try:
                idxs = ast.literal_eval(str(raw))
                if isinstance(idxs, int):
                    idxs = [idxs]
                for i in idxs:
                    if 0 <= i < self.n_l4:
                        l4_mh[i] = 1.0
            except Exception:
                if labels[3] >= 0:
                    l4_mh[labels[3]] = 1.0
        elif labels[3] >= 0:
            l4_mh[labels[3]] = 1.0

        return h26, seq_len, cmap, labels, masks, l4_mh


def collate_cached(batch):
    h26_list, lens, cmaps, labels, masks, l4_mh = zip(*batch)
    max_len = max(lens)
    B, D = len(h26_list), h26_list[0].shape[1]

    h26_pad  = torch.zeros(B, max_len, D)
    attn_mask= torch.zeros(B, max_len)
    for i, (h, l) in enumerate(zip(h26_list, lens)):
        h26_pad[i, :l]  = torch.from_numpy(h)
        attn_mask[i, :l]= 1.0

    return (h26_pad, attn_mask,
            torch.tensor(np.stack(cmaps), dtype=torch.float32),
            torch.stack(labels), torch.stack(masks), torch.stack(l4_mh))


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config_expa_e2e.yaml")
    parser.add_argument("--flatfc_ckpt", required=True)
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--grad_accum",  type=int,   default=2)
    parser.add_argument("--lr_esm",      type=float, default=1e-5)
    parser.add_argument("--lr_rest",     type=float, default=1e-4)
    parser.add_argument("--unfreeze",    type=int,   default=6)
    parser.add_argument("--fp16",        action="store_true")
    parser.add_argument("--gpu",         default="0")
    parser.add_argument("--tag",         default="flatfc_e2e_cached")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Config
    with open(ROOT / "configs" / args.config) as f:
        CFG = yaml.safe_load(f)

    label_enc = CFG.get("paths", CFG).get("label_enc",
                ROOT / "data/expa/label_encoders.pkl")
    splits_dir= CFG.get("paths", CFG).get("splits_dir",
                ROOT / "data/expa/splits")
    meta_csv  = CFG.get("paths", CFG).get("meta_csv",
                ROOT / "data/expa/dataset_meta_reenc.csv")
    cmap_dir  = CFG.get("paths", {}).get("cmap_dir",
                ROOT / "data/processed/contact_maps")
    cache_dir = ROOT / "data/processed/esm2_layer26"
    ckpt_dir  = ROOT / "outputs/checkpoints"
    log_dir   = ROOT / "outputs/logs"

    with open(label_enc, "rb") as f:
        enc = pickle.load(f)
    n_classes = [len(enc[f"level{i}"].classes_) for i in range(1, 5)]
    print(f"클래스 수: {n_classes}")

    loss_w = CFG.get("train", {}).get("loss_weights", [0.1, 0.1, 0.2, 0.6])
    loss_w = torch.tensor(loss_w, dtype=torch.float32).to(DEVICE)
    M12 = build_parent_child_matrix(enc, 1, 2).to(DEVICE)
    M23 = build_parent_child_matrix(enc, 2, 3).to(DEVICE)
    M34 = build_parent_child_matrix(enc, 3, 4).to(DEVICE)

    # Dataset
    def make_ds(split):
        return Layer26Dataset(
            ids_file    = f"{splits_dir}/{split}_ids.txt",
            meta_csv    = str(meta_csv),
            cmap_dir    = str(cmap_dir),
            cache_dir   = str(cache_dir),
            label_enc_pkl = str(label_enc),
            cmap_size   = CFG.get("data", {}).get("cmap_size", 256),
        )

    train_ds = make_ds("train")
    val_ds   = make_ds("val")
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4,
                              collate_fn=collate_cached, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4,
                              collate_fn=collate_cached, pin_memory=True)

    # Model
    from models.fusion_flatfc_esm_ft import FusionFlatFCESMFt
    model = FusionFlatFCESMFt(n_classes, unfreeze_layers=args.unfreeze).to(DEVICE)

    flatfc_path = Path(args.flatfc_ckpt)
    if not flatfc_path.is_absolute():
        flatfc_path = ROOT / flatfc_path
    flatfc_path = str(flatfc_path)
    model.load_flatfc_checkpoint(flatfc_path)
    model.unfreeze_esm_last(args.unfreeze)

    param_groups = model.get_param_groups(args.lr_esm, args.lr_rest)
    optimizer    = torch.optim.AdamW(param_groups, weight_decay=5e-4)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = GradScaler() if args.fp16 else None
    if args.fp16:
        print("fp16 AMP 활성화")
    print(f"Gradient accumulation: {args.grad_accum} steps")

    best_micro  = 0.0
    best_path   = ckpt_dir / f"{args.tag}_best.pt"
    log_path    = log_dir  / f"{args.tag}.log"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        optimizer.zero_grad()
        train_losses = []

        for step, batch in enumerate(train_loader):
            h26, attn, cmap, labels, masks, l4_mh = batch
            h26 = h26.to(DEVICE); attn = attn.to(DEVICE)
            cmap   = cmap.to(DEVICE)
            labels = labels.to(DEVICE); masks = masks.to(DEVICE)
            l4_mh  = l4_mh.to(DEVICE)

            use_amp = (scaler is not None)
            with autocast(enabled=use_amp):
                logits = model.forward_cached(h26, attn, cmap)
                loss = hierarchical_loss(logits, labels, masks, loss_w,
                                         l4_multihot=l4_mh, M12=M12,
                                         M23=M23, M34=M34)
                cons = _CONS_WEIGHT * consistency_loss(logits, M12, M23, M34)
                loss = (loss + cons) / args.grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            train_losses.append(loss.item() * args.grad_accum)

        # ── Val ──
        model.eval()
        val_losses = []
        all_l4_probs, all_l4_mh = [], []

        with torch.no_grad():
            for batch in val_loader:
                h26, attn, cmap, labels, masks, l4_mh = batch
                h26 = h26.to(DEVICE); attn = attn.to(DEVICE)
                cmap   = cmap.to(DEVICE)
                labels = labels.to(DEVICE); masks = masks.to(DEVICE)
                l4_mh  = l4_mh.to(DEVICE)

                logits = model.forward_cached(h26, attn, cmap)
                vloss  = hierarchical_loss(logits, labels, masks, loss_w,
                                           l4_multihot=l4_mh, M12=M12,
                                           M23=M23, M34=M34)
                val_losses.append(vloss.item())

                probs = torch.sigmoid(logits[3]).cpu().numpy()
                all_l4_probs.append(probs)
                all_l4_mh.append(l4_mh.cpu().numpy())

        scheduler.step()

        micro, macro, weighted = compute_metrics_ml(
            np.concatenate(all_l4_probs), np.concatenate(all_l4_mh))

        elapsed = int(time.time() - t0)
        line = (f"[{epoch:03d}/{args.epochs}]  "
                f"train_loss={np.mean(train_losses):.4f}  "
                f"val_loss={np.mean(val_losses):.4f}  "
                f"micro_f1={micro:.4f}  "
                f"weighted_f1={weighted:.4f}  "
                f"macro_f1={macro:.4f}  ({elapsed}s)")
        print(line, flush=True)

        with open(log_path, "a") as f:
            f.write(line + "\n")

        if micro > best_micro:
            best_micro = micro
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "micro_f1": micro}, best_path)
            print(f"  ✓ 베스트 저장: {best_path.name}  (micro={micro:.4f})",
                  flush=True)

    print(f"\nFlatFC E2E 완료. 베스트 micro_f1: {best_micro:.4f}")
    print(f"체크포인트: {best_path}")


if __name__ == "__main__":
    main()
