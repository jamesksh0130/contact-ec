"""
Step 5: ESM-2 임베딩 추출 (HuggingFace transformers)
- 입력: data/processed/dataset_meta.csv
- 출력: data/processed/embeddings/{accession}.npy  (1280,) float32
- 모델: facebook/esm2_t33_650M_UR50D  (초기 실험은 esm2_t6_8M_UR50D)
"""
import os, sys, time, yaml, argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

META_CSV  = ROOT / CFG["paths"]["meta_csv"]
EMBED_DIR = ROOT / CFG["paths"]["embed_dir"]
EMBED_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN   = CFG["data"]["max_seq_len"]
ESM_MODEL = CFG["model"]["esm2_model"]
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, records):
        self.records = records   # [(accession, sequence), ...]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]   # (accession, sequence)


def collate_fn(batch):
    accs  = [b[0] for b in batch]
    seqs  = [b[1] for b in batch]
    return accs, seqs


# ── 메인 ───────────────────────────────────────────────────────
def main(batch_size: int, model_name: str, workers: int):
    meta = pd.read_csv(META_CSV)

    # 이미 추출된 것 제외
    existing = {p.stem for p in EMBED_DIR.glob("*.npy")}
    todo = [(row["accession"], row["sequence"])
            for _, row in meta.iterrows()
            if row["accession"] not in existing
            and isinstance(row["sequence"], str)
            and len(row["sequence"]) >= 4]

    print(f"총 {len(meta):,}개 중  기존={len(existing):,}  남은={len(todo):,}")
    if not todo:
        print("모든 임베딩이 이미 존재합니다.")
        return

    print(f"\nESM-2 모델 로드: {model_name}")
    print(f"디바이스: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = EsmModel.from_pretrained(model_name)
    model.eval().to(DEVICE)

    # GPU 메모리에 따라 batch_size 조정
    if DEVICE == "cuda":
        free_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU 메모리: {free_gb:.0f}GB  |  batch_size={batch_size}")

    ds     = SeqDataset(todo)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, collate_fn=collate_fn)

    t0     = time.time()
    total  = len(todo)
    done   = 0

    with torch.no_grad():
        for accs, seqs in loader:
            # 토크나이징 (길이 잘림 포함)
            inputs = tokenizer(
                list(seqs),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN + 2,   # BOS/EOS 포함
            ).to(DEVICE)

            out = model(**inputs)
            # CLS token (index 0) → (B, hidden_dim)
            cls_emb = out.last_hidden_state[:, 0, :].cpu().float().numpy()

            for acc, emb in zip(accs, cls_emb):
                np.save(EMBED_DIR / f"{acc}.npy", emb)

            done += len(accs)
            elapsed = time.time() - t0
            rate    = done / elapsed
            eta     = (total - done) / rate if rate > 0 else 0
            print(f"\r  [{done:>6}/{total}]  {rate:.1f}/s  ETA {eta/60:.1f}min  "
                  f"shape={cls_emb.shape[1]}", end="", flush=True)

    print(f"\n\n완료: {(time.time()-t0)/60:.1f}분")
    n_saved = len(list(EMBED_DIR.glob("*.npy")))
    print(f"저장된 임베딩: {n_saved:,}개")

    # 검증
    samples = list(EMBED_DIR.glob("*.npy"))[:3]
    for s in samples:
        arr = np.load(s)
        print(f"  {s.stem}: shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=ESM_MODEL,
                        help="HuggingFace ESM-2 모델 ID")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers",    type=int, default=4)
    args = parser.parse_args()
    main(args.batch_size, args.model, args.workers)
