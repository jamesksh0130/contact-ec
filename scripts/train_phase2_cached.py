"""
Phase 2 End-to-End Fine-tuning with Layer 26 Cache

기존 train.py의 Phase 2를 대체:
  - ESM-2 layer 27-32만 실행 (layer 0-26은 캐시에서 로드)
  - 속도: ~3-4× 빠름
  - 결과: train.py --phase 2와 수학적으로 동일

사용법:
  python scripts/train_phase2_cached.py \
      --config config_expa_e2e.yaml \
      --resume outputs/checkpoints/expa_e2e_phase1_best.pt \
      --epochs 10 \
      --batch_size 8 \
      --grad_accum 4 \
      --fp16
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/home/user/Desktop/unlv")
sys.path.insert(0, str(ROOT))

import yaml
from sklearn.metrics import f1_score

from models.fusion_esm_ft_cached import FusionESMFinetuneV2
from models.dataset import _make_3ch_cmap
from train import hierarchical_loss, build_parent_child_matrix, consistency_loss, compute_metrics_ml

CACHE_DIR = ROOT / "data/processed/esm2_layer26"
ESM_DIM   = 1280


class Layer26Dataset(Dataset):
    """
    phase 2 캐시 기반 Dataset.
    layer26_h: (seq_len, 1280) fp16 numpy → tensor
    attn_mask: ones of length seq_len
    cmap: (3, 256, 256) float32
    labels/masks/l4_mh: 기존과 동일
    """

    def __init__(self, ids_file: str, meta_csv: str,
                 cmap_dir: str, label_enc_pkl: str, cmap_size: int = 256):
        self.cmap_dir  = Path(cmap_dir)
        self.cmap_size = cmap_size

        self.ids = Path(ids_file).read_text().strip().splitlines()
        self.ids = [i.strip() for i in self.ids if i.strip()]

        self.meta = pd.read_csv(meta_csv).set_index("accession")

        with open(label_enc_pkl, "rb") as f:
            self.encoders = pickle.load(f)
        self.n_classes = [len(self.encoders[f"level{i}"].classes_) for i in range(1, 5)]
        self.n_l4 = self.n_classes[3]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        acc = self.ids[idx]
        row = self.meta.loc[acc] if acc in self.meta.index else None

        # ── Layer 26 캐시 ────────────────────────────────────────
        cache_path = CACHE_DIR / f"{acc}.npy"
        if cache_path.exists():
            layer26_h = np.load(cache_path).astype(np.float32)  # (seq_len, 1280) fp32
            seq_len   = layer26_h.shape[0]
        else:
            # Fallback: zero cache of length 2 (BOS+EOS only)
            layer26_h = np.zeros((2, ESM_DIM), dtype=np.float32)
            seq_len   = 2

        attn_mask = np.ones(seq_len, dtype=np.float32)  # all valid

        # ── 3채널 Contact Map ─────────────────────────────────────
        cmap_path = self.cmap_dir / f"{acc}.npy"
        if cmap_path.exists():
            raw = np.load(cmap_path).astype(np.float32)
            cmap = (_make_3ch_cmap(raw, self.cmap_size)
                    if raw.shape == (self.cmap_size, self.cmap_size)
                    else np.zeros((3, self.cmap_size, self.cmap_size), dtype=np.float32))
        else:
            cmap = np.zeros((3, self.cmap_size, self.cmap_size), dtype=np.float32)

        # ── Labels ────────────────────────────────────────────────
        if row is not None:
            labels = np.array([row.get(f"l{i}_idx", -1) for i in range(1, 5)], dtype=np.int64)
            masks  = np.array([float(row.get(f"m{i}", 0)) for i in range(1, 5)], dtype=np.float32)
        else:
            labels = np.full(4, -1, dtype=np.int64)
            masks  = np.zeros(4, dtype=np.float32)

        l4_multihot = np.zeros(self.n_l4, dtype=np.float32)
        if row is not None:
            idxs_raw = str(row.get("l4_all_idxs", "")).strip("[]")
            if idxs_raw and idxs_raw != "nan":
                sep = "|" if "|" in idxs_raw else ","
                for s in idxs_raw.split(sep):
                    s = s.strip().strip("[]")
                    if s:
                        i4 = int(float(s))
                        if 0 <= i4 < self.n_l4:
                            l4_multihot[i4] = 1.0
            if l4_multihot.sum() == 0 and labels[3] >= 0:
                l4_multihot[labels[3]] = 1.0

        return (torch.tensor(layer26_h),   # (seq_len, 1280)
                torch.tensor(attn_mask),   # (seq_len,)
                torch.tensor(cmap),        # (3, 256, 256)
                torch.tensor(labels),      # (4,)
                torch.tensor(masks),       # (4,)
                torch.tensor(l4_multihot), # (n_l4,)
                acc)


def collate_cached(batch):
    """Pad layer26_h and attn_mask to max_seq_len in batch."""
    h_list, mask_list, cmaps, labels, masks, l4_mh, accs = zip(*batch)

    max_len = max(h.shape[0] for h in h_list)
    B       = len(h_list)

    h_padded    = torch.zeros(B, max_len, ESM_DIM, dtype=torch.float32)
    mask_padded = torch.zeros(B, max_len, dtype=torch.float32)

    for i, (h, m) in enumerate(zip(h_list, mask_list)):
        slen = h.shape[0]
        h_padded[i, :slen, :]  = h
        mask_padded[i, :slen]  = m

    return (h_padded,
            mask_padded,
            torch.stack(cmaps),
            torch.stack(labels),
            torch.stack(masks),
            torch.stack(l4_mh),
            list(accs))


# ── Training / Eval loops ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scaler, weights, grad_accum, use_amp,
                device, M12, M23, M34):
    model.train()
    total_loss = 0.0
    n_batch    = 0
    optimizer.zero_grad()

    for step, (h, amask, cmap, labels, masks, l4_mh, _) in enumerate(loader):
        h      = h.to(device)
        amask  = amask.to(device)
        cmap   = cmap.to(device)
        labels = labels.to(device)
        masks  = masks.to(device)
        l4_mh  = l4_mh.to(device)

        with autocast(enabled=use_amp):
            logits = model.forward_cached(h, amask, cmap)
            loss   = hierarchical_loss(logits, labels, masks, weights,
                                       l4_multihot=l4_mh, M12=M12, M23=M23, M34=M34)
            loss   = loss + 0.05 * consistency_loss(logits, M12, M23, M34)

        if use_amp:
            scaler.scale(loss / grad_accum).backward()
        else:
            (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        n_batch    += 1

    return total_loss / max(n_batch, 1)


@torch.no_grad()
def eval_epoch(model, loader, weights, device, M12, M23, M34):
    model.eval()
    total_loss   = 0.0
    all_l4_probs = []
    all_l4_mh    = []

    for h, amask, cmap, labels, masks, l4_mh, _ in loader:
        h      = h.to(device)
        amask  = amask.to(device)
        cmap   = cmap.to(device)
        labels = labels.to(device)
        masks  = masks.to(device)
        l4_mh  = l4_mh.to(device)

        logits = model.forward_cached(h, amask, cmap)
        loss   = hierarchical_loss(logits, labels, masks, weights, l4_multihot=l4_mh,
                                   M12=M12, M23=M23, M34=M34)  # no consistency in val
        total_loss += loss.item()

        all_l4_probs.append(torch.sigmoid(logits[3]).cpu().numpy())
        all_l4_mh.append(l4_mh.cpu().numpy())

    all_l4_probs = np.concatenate(all_l4_probs, axis=0)
    all_l4_mh    = np.concatenate(all_l4_mh,    axis=0)
    micro, macro, weighted = compute_metrics_ml(all_l4_probs, all_l4_mh)
    return total_loss / len(loader), micro, macro, weighted


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config_expa_e2e.yaml")
    parser.add_argument("--resume",     required=True,
                        help="Phase 1 체크포인트 경로")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--grad_accum", type=int,   default=4)
    parser.add_argument("--lr_esm",     type=float, default=1e-5)
    parser.add_argument("--lr_rest",    type=float, default=1e-4)
    parser.add_argument("--fp16",       action="store_true")
    parser.add_argument("--gpu",        type=int,   default=0)
    parser.add_argument("--tag",        default="expa_e2e_phase2_cached")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device} | Phase 2 (Layer26 캐시) | fp16={args.fp16}", flush=True)

    # ── Config ──────────────────────────────────────────────────────────────
    with open(ROOT / "configs" / args.config) as f:
        CFG = yaml.safe_load(f)

    PATHS  = CFG["paths"]
    TCONF  = CFG["train"]
    MCONF  = CFG["model"]
    splits_dir = ROOT / PATHS["splits_dir"]
    bs = args.batch_size or TCONF.get("batch_size", 8)

    # ── Label encoders ───────────────────────────────────────────────────────
    with open(ROOT / PATHS["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    print(f"클래스 수: {n_classes}", flush=True)

    M12 = build_parent_child_matrix(encoders, 1, 2).to(device)
    M23 = build_parent_child_matrix(encoders, 2, 3).to(device)
    M34 = build_parent_child_matrix(encoders, 3, 4).to(device)

    # ── Datasets ─────────────────────────────────────────────────────────────
    meta_csv  = str(ROOT / PATHS["meta_csv"])
    cmap_dir  = str(ROOT / PATHS["cmap_dir"])
    label_enc = str(ROOT / PATHS["label_enc"])

    train_ds = Layer26Dataset(
        ids_file=str(splits_dir / "train_ids.txt"),
        meta_csv=meta_csv, cmap_dir=cmap_dir, label_enc_pkl=label_enc,
    )
    val_ds = Layer26Dataset(
        ids_file=str(splits_dir / "val_ids.txt"),
        meta_csv=meta_csv, cmap_dir=cmap_dir, label_enc_pkl=label_enc,
    )

    n_workers = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=n_workers, pin_memory=True,
                              collate_fn=collate_cached, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=n_workers, pin_memory=True,
                              collate_fn=collate_cached)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | "
          f"Batch: {bs} (×{args.grad_accum} accum = effective {bs*args.grad_accum})", flush=True)

    # ── Model ────────────────────────────────────────────────────────────────
    unfreeze_n = MCONF.get("esm_unfreeze_layers", 6)
    model = FusionESMFinetuneV2(
        n_classes=n_classes,
        contact_dim=MCONF.get("contact_dim", 512),
        fusion_dim=MCONF.get("fusion_dim", 1024),
        num_heads=MCONF.get("num_heads", 8),
        dropout=MCONF.get("dropout", 0.3),
        unfreeze_layers=unfreeze_n,
    ).to(device)

    # Load Phase 1 checkpoint
    resume = Path(args.resume)
    if not resume.is_absolute():
        resume = ROOT / "outputs/checkpoints" / args.resume
    ckpt = torch.load(resume, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    print(f"Phase 1 체크포인트 로드: {resume.name}", flush=True)

    # Unfreeze last N layers
    model.unfreeze_esm_last(unfreeze_n)

    # ── Optimizer ────────────────────────────────────────────────────────────
    param_groups = model.get_param_groups(lr_esm=args.lr_esm, lr_rest=args.lr_rest)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler() if args.fp16 else None
    use_amp = args.fp16

    # ── Class weights ────────────────────────────────────────────────────────
    l4_classes = n_classes[3]
    weights = [0.1, 0.1, 0.2, 0.6]

    # ── Training loop ────────────────────────────────────────────────────────
    best_micro = 0.0
    ckpt_dir   = ROOT / "outputs/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path  = ckpt_dir / f"{args.tag}_best.pt"
    log_path   = ROOT / "outputs/logs" / f"{args.tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file   = open(log_path, "w", buffering=1)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler,
                                  weights, args.grad_accum, use_amp, device, M12, M23, M34)
        val_loss, micro, macro, weighted = eval_epoch(model, val_loader,
                                                       weights, device, M12, M23, M34)
        scheduler.step()
        elapsed = int(time.time() - t0)

        line = (f"[{epoch:03d}/{args.epochs}]  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"micro_f1={micro:.4f}  weighted_f1={weighted:.4f}  "
                f"macro_f1={macro:.4f}  ({elapsed}s)")
        print(line, flush=True)
        log_file.write(line + "\n")

        if micro > best_micro:
            best_micro = micro
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "micro_f1": micro}, best_path)
            note = f"  ✓ 베스트 저장: {best_path.name}  (micro={micro:.4f}  weighted={weighted:.4f})"
            print(note, flush=True)
            log_file.write(note + "\n")

    log_file.close()
    print(f"\nPhase 2 (캐시) 완료. 베스트 micro_f1: {best_micro:.4f}", flush=True)
    print(f"체크포인트: {best_path}", flush=True)


if __name__ == "__main__":
    main()
