"""
Exp 1: ESM-2 3B (facebook/esm2_t36_3B_UR50D, dim=2560) 임베딩 추출
EC-Bench train + val + test + New-392 + Price-149 대상

사용법:
  CUDA_VISIBLE_DEVICES=1 python -u scripts/07_extract_esm2_3b_embeddings.py
출력: data/processed/embeddings_3b/{UniProtID}.npy  (2560,)
"""
import sys, pickle, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_DIR     = ROOT / "data" / "processed" / "embeddings_3b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = "facebook/esm2_t36_3B_UR50D"
MAX_LEN     = 1024
BATCH_SIZE  = 8   # 3B × batch8 ≈ 24GB

SPLITS_DIR  = ROOT / "data" / "ecbench" / "splits"
META_CSV    = ROOT / "data" / "ecbench" / "processed" / "train_meta.csv"
NEW392_CSV  = ROOT / "data" / "new392" / "new392_ec_labels.csv"
PRICE149_CSV= ROOT / "data" / "new392" / "price149_labels.csv"


def collect_ids():
    ids, seq_map = set(), {}

    # EC-Bench splits
    for fname in ["train_ids.txt", "val_ids.txt", "test_ids.txt",
                  "val_hard_ids.txt", "val_easy_ids.txt"]:
        fp = SPLITS_DIR / fname
        if fp.exists():
            ids.update(fp.read_text().splitlines())

    # Load sequences from meta CSV
    meta = pd.read_csv(META_CSV)
    for _, row in meta.iterrows():
        uid = row["accession"] if "accession" in meta.columns else row.get("Entry", "")
        if uid and uid in ids:
            seq_map[uid] = row.get("sequence", row.get("Sequence", ""))

    # New-392 + Price-149
    for csv_path in [NEW392_CSV, PRICE149_CSV]:
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep="\t")
            for _, row in df.iterrows():
                uid = row["Entry"]
                seq_map[uid] = row["Sequence"]
                ids.add(uid)

    print(f"수집된 ID: {len(ids)}  시퀀스 로드: {len(seq_map)}")
    return seq_map


def extract_batch(seqs, tokenizer, model):
    enc = tokenizer(seqs, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
    cls = out.last_hidden_state[:, 0, :].cpu().float().numpy()  # (B, 2560)
    return cls


def main():
    print(f"[Exp 1] ESM-2 3B 임베딩 추출 → {OUT_DIR}")

    seq_map = collect_ids()

    # 이미 있는 것 제외
    to_do = [(uid, seq) for uid, seq in seq_map.items()
             if not (OUT_DIR / f"{uid}.npy").exists()]
    print(f"  추출 필요: {len(to_do)}개 (기존: {len(seq_map)-len(to_do)}개)")

    if not to_do:
        print("  모두 완료 — 종료")
        return

    print(f"  ESM-2 3B 로드 중... (시간 소요)")
    t0 = time.time()
    from transformers import AutoTokenizer, EsmModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = EsmModel.from_pretrained(MODEL_NAME).eval().to(DEVICE)
    print(f"  로드 완료 ({time.time()-t0:.0f}s)  dim=2560")

    ok, fail = 0, 0
    for i in tqdm(range(0, len(to_do), BATCH_SIZE), desc="  3B 임베딩"):
        batch = to_do[i:i+BATCH_SIZE]
        uids  = [b[0] for b in batch]
        seqs  = [b[1][:MAX_LEN] for b in batch]
        try:
            embs = extract_batch(seqs, tokenizer, model)
            for uid, emb in zip(uids, embs):
                np.save(OUT_DIR / f"{uid}.npy", emb)
            ok += len(batch)
        except Exception as e:
            print(f"\n  ✗ {uids}: {e}")
            fail += len(batch)

    print(f"\n완료: {ok}개 성공  {fail}개 실패")
    print(f"출력: {OUT_DIR}")


if __name__ == "__main__":
    main()
