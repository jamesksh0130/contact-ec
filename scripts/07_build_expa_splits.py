"""
Experiment A: full Swiss-Prot 270K 데이터로 train/val 분할 및 label encoder 생성
- 270,336 단백질에서 EC-Bench temporal test 115개 제거
- train 90% / val 10% random split (seed=42)
- label encoder 재피팅
- 출력: data/expa/splits/, data/expa/label_encoders.pkl
"""
import pickle, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

ROOT = Path("/home/user/Desktop/unlv")
OUT_DIR = ROOT / "data/expa/splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENC_OUT = ROOT / "data/expa/label_encoders.pkl"

# ── 1. 전체 meta 로드 및 test 제거 ───────────────────────────
meta = pd.read_csv(ROOT / "data/processed/dataset_meta.csv")
test_ids = set((ROOT / "data/ecbench/splits/test_ids_full.txt").read_text().split())

train_meta = meta[~meta["accession"].isin(test_ids)].copy()
print(f"전체 단백질     : {len(meta):,}")
print(f"Test 제거       : {len(test_ids)} → {len(meta) - len(train_meta)} 실제 제거")
print(f"학습 가능 단백질: {len(train_meta):,}")

# ── 2. label encoder 피팅 ────────────────────────────────────
encoders = {}
for level, col in enumerate(["l1_str", "l2_str", "l3_str", "l4_str"], start=1):
    le = LabelEncoder()
    valid = sorted(train_meta[col].dropna().astype(str).unique())
    le.fit(valid)
    encoders[f"level{level}"] = le
    print(f"  level{level}: {len(le.classes_)} classes")

with open(ENC_OUT, "wb") as f:
    pickle.dump(encoders, f)
print(f"Label encoders 저장: {ENC_OUT}")

# ── 3. train/val split (90/10, seed=42) ──────────────────────
ids = train_meta["accession"].tolist()
train_ids, val_ids = train_test_split(ids, test_size=0.10, random_state=42)

(OUT_DIR / "train_ids.txt").write_text("\n".join(train_ids))
(OUT_DIR / "val_ids.txt").write_text("\n".join(val_ids))

print(f"\nSplit 완료:")
print(f"  train : {len(train_ids):,}")
print(f"  val   : {len(val_ids):,}")
print(f"  test  : {len(test_ids)} (EC-Bench temporal, 별도 고정)")
print(f"\n저장 위치: {OUT_DIR}")
