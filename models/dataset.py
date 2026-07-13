"""
ProteinDataset — 모든 모델 공통 데이터셋
반환: (esm_emb, cmap, sequence, labels, masks, l4_multihot, accession)
  - esm_emb    : (1280,) float32  — ESM-2 CLS 임베딩 (없으면 zeros)
  - cmap       : (3, 256, 256) float32  — 3채널 Contact Map (없으면 zeros)
  - sequence   : str  — 아미노산 서열 (B0 1D-CNN용)
  - labels     : (4,) int64  — Level 1~4 단일 인덱스, 무효=-1
  - masks      : (4,) float32  — 유효 레벨=1.0, 무효=0.0
  - l4_multihot: (n_l4,) float32  — Level4 멀티핫 (없으면 zeros)
  - accession  : str

ContactPairDataset (FusionV3용) — 추가 반환값:
  - pair_emb   : (K, 2, 1280) float16  — contact pair ESM-2 tokens
"""
import os, pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

# (256,256) short-range/long-range mask — 모듈 로드 시 1회만 계산
_SIZE_DEFAULT = 256
_idx  = np.arange(_SIZE_DEFAULT)
_diff = np.abs(_idx[:, None] - _idx[None, :])   # (256,256)
_SHORT_MASK = (_diff < 12).astype(np.float32)   # |i-j|<12 → 2차 구조
_LONG_MASK  = 1.0 - _SHORT_MASK                  # |i-j|≥12 → 3차 접촉


def _make_3ch_cmap(binary: np.ndarray, size: int = 256) -> np.ndarray:
    """(256,256) binary contact map → (3,256,256): all / short-range / long-range."""
    if size != _SIZE_DEFAULT:
        idx  = np.arange(size)
        diff = np.abs(idx[:, None] - idx[None, :])
        sm   = (diff < 12).astype(np.float32)
        lm   = 1.0 - sm
    else:
        sm, lm = _SHORT_MASK, _LONG_MASK
    return np.stack([binary, binary * sm, binary * lm], axis=0)  # (3,H,W)


class ProteinDataset(Dataset):
    def __init__(self, ids_file: str, meta_csv: str, embed_dir: str,
                 cmap_dir: str, label_enc_pkl: str, cmap_size: int = 256):
        self.embed_dir = Path(embed_dir)
        self.cmap_dir  = Path(cmap_dir)
        self.cmap_size = cmap_size

        self.ids = Path(ids_file).read_text().strip().split("\n")

        meta = pd.read_csv(meta_csv).set_index("accession")
        self.meta = meta

        with open(label_enc_pkl, "rb") as f:
            self.encoders = pickle.load(f)

        self.n_classes = [len(self.encoders[f"level{i}"].classes_)
                          for i in range(1, 5)]
        self.n_l4 = self.n_classes[3]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        acc = self.ids[idx]
        row = self.meta.loc[acc] if acc in self.meta.index else None

        # ── ESM-2 임베딩 ──────────────────────────────────────
        emb_path = self.embed_dir / f"{acc}.npy"
        esm_emb  = (np.load(emb_path).astype(np.float32)
                    if emb_path.exists()
                    else np.zeros(1280, dtype=np.float32))

        # ── 3채널 Contact Map ─────────────────────────────────
        cmap_path = self.cmap_dir / f"{acc}.npy"
        if cmap_path.exists():
            raw = np.load(cmap_path).astype(np.float32)
            cmap = (_make_3ch_cmap(raw, self.cmap_size)
                    if raw.shape == (self.cmap_size, self.cmap_size)
                    else np.zeros((3, self.cmap_size, self.cmap_size), dtype=np.float32))
        else:
            cmap = np.zeros((3, self.cmap_size, self.cmap_size), dtype=np.float32)

        # ── 서열 (B0 1D-CNN용) ────────────────────────────────
        sequence = str(row["sequence"]) if row is not None else ""

        # ── 단일 레이블 & 마스크 (L1~L4) ──────────────────────
        if row is not None:
            labels = np.array([row.get(f"l{i}_idx", -1) for i in range(1, 5)],
                              dtype=np.int64)
            masks  = np.array([float(row.get(f"m{i}", 0)) for i in range(1, 5)],
                              dtype=np.float32)
        else:
            labels = np.full(4, -1, dtype=np.int64)
            masks  = np.zeros(4, dtype=np.float32)

        # ── Level4 멀티핫 레이블 ─────────────────────────────
        l4_multihot = np.zeros(self.n_l4, dtype=np.float32)
        if row is not None:
            idxs_raw = str(row.get("l4_all_idxs", "")).strip("[]")
            if idxs_raw and idxs_raw != "nan":
                # 파이프(|) 또는 쉼표(,) 구분자 모두 지원
                sep = "|" if "|" in idxs_raw else ","
                for s in idxs_raw.split(sep):
                    s = s.strip().strip("[]")
                    if s:
                        i4 = int(float(s))
                        if 0 <= i4 < self.n_l4:
                            l4_multihot[i4] = 1.0
            # fallback: single label
            if l4_multihot.sum() == 0 and labels[3] >= 0:
                l4_multihot[labels[3]] = 1.0

        return (torch.tensor(esm_emb),
                torch.tensor(cmap),
                sequence,
                torch.tensor(labels),
                torch.tensor(masks),
                torch.tensor(l4_multihot),
                acc)


def collate_fn(batch):
    """DataLoader용 collate — 문자열 서열은 list로 반환."""
    esm_embs, cmaps, sequences, labels, masks, l4_mh, accs = zip(*batch)
    return (torch.stack(esm_embs),
            torch.stack(cmaps),
            list(sequences),
            torch.stack(labels),
            torch.stack(masks),
            torch.stack(l4_mh),
            list(accs))


# ── FusionV3용 Contact Pair Dataset ──────────────────────────
class ContactPairDataset(ProteinDataset):
    """
    ProteinDataset + contact pair 임베딩 추가 로드 (FusionV3용).
    pair_emb_dir: data/processed/contact_pair_embs/ 경로
    K_PAIRS: contact pair 수 (전처리 시 고정, 기본 32)
    """
    def __init__(self, ids_file: str, meta_csv: str, embed_dir: str,
                 cmap_dir: str, label_enc_pkl: str,
                 pair_emb_dir: str, k_pairs: int = 32,
                 cmap_size: int = 256):
        super().__init__(ids_file, meta_csv, embed_dir, cmap_dir,
                         label_enc_pkl, cmap_size)
        self.pair_emb_dir = Path(pair_emb_dir)
        self.k_pairs      = k_pairs

    def __getitem__(self, idx):
        # 부모 클래스에서 기본 아이템 로드
        esm_emb, cmap, sequence, labels, masks, l4_mh, acc = super().__getitem__(idx)

        # contact pair 임베딩 로드 (K, 2, 1280) float16
        pair_path = self.pair_emb_dir / f"{acc}.npy"
        if pair_path.exists():
            pair_emb = np.load(str(pair_path))   # (K, 2, 1280) float16
            if pair_emb.shape != (self.k_pairs, 2, 1280):
                pair_emb = np.zeros((self.k_pairs, 2, 1280), dtype=np.float16)
        else:
            pair_emb = np.zeros((self.k_pairs, 2, 1280), dtype=np.float16)

        return (esm_emb, cmap, sequence, labels, masks, l4_mh, acc,
                torch.tensor(pair_emb))   # float16 tensor


def collate_fn_v3(batch):
    """FusionV3용 collate — pair_emb 추가."""
    esm_embs, cmaps, sequences, labels, masks, l4_mh, accs, pair_embs = zip(*batch)
    return (torch.stack(esm_embs),
            torch.stack(cmaps),
            list(sequences),
            torch.stack(labels),
            torch.stack(masks),
            torch.stack(l4_mh),
            list(accs),
            torch.stack(pair_embs))
