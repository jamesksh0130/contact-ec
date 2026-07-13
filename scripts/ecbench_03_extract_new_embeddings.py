"""
EC-Bench Step 3: 새 단백질 ESM-2 임베딩 추출

data/ecbench/new_proteins.txt에 있는 단백질 중
기존 임베딩이 없는 것만 추출.
(대부분은 재활용 가능 — 소수만 새로 추출)
"""
import torch, pickle
from pathlib import Path
from transformers import AutoTokenizer, EsmModel
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT     = Path("/home/user/Desktop/unlv")
ECBENCH  = ROOT / "data/ecbench"
EMB_DIR  = ROOT / "data/processed/embeddings"
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"
ESM_ID   = "facebook/esm2_t33_650M_UR50D"
BATCH    = 8
MAX_LEN  = 1024


def main():
    # 새 단백질 목록
    new_ids = set((ECBENCH / "new_proteins.txt").read_text().strip().split("\n"))
    existing = set(p.stem for p in EMB_DIR.glob("*.npy"))
    to_process = [uid for uid in sorted(new_ids) if uid not in existing]
    print(f"새로 추출 필요: {len(to_process):,}개")
    if not to_process:
        print("모두 재활용 가능 — 종료")
        return

    # 서열 로드
    train_meta = pd.read_csv(ECBENCH / "processed/train_meta.csv")
    test_meta  = pd.read_csv(ECBENCH / "processed/test_meta.csv")
    all_meta   = pd.concat([train_meta, test_meta])
    seq_map    = dict(zip(all_meta["accession"], all_meta["sequence"]))

    # 추가로 test/price149 서열 포함
    price_meta = pd.read_csv(ECBENCH / "processed/price149_meta.csv")
    seq_map.update(dict(zip(price_meta["accession"], price_meta["sequence"])))

    # 존재하는 서열만 처리
    to_process = [(uid, seq_map[uid]) for uid in to_process if uid in seq_map]
    print(f"서열 확보: {len(to_process):,}개")

    # ESM-2 로드
    print(f"ESM-2 로드 중: {ESM_ID}")
    tokenizer = AutoTokenizer.from_pretrained(ESM_ID)
    model     = EsmModel.from_pretrained(ESM_ID).eval().to(DEVICE)

    # 배치 추출
    for i in tqdm(range(0, len(to_process), BATCH), desc="임베딩 추출"):
        batch = to_process[i:i+BATCH]
        uids  = [x[0] for x in batch]
        seqs  = [x[1] for x in batch]

        enc = tokenizer(seqs, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LEN).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
        cls_embs = out.last_hidden_state[:, 0, :].cpu().numpy()

        for uid, emb in zip(uids, cls_embs):
            np.save(EMB_DIR / f"{uid}.npy", emb.astype(np.float32))

    print(f"완료: {len(to_process):,}개 임베딩 저장")


if __name__ == "__main__":
    main()
