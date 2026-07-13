"""
EC-Bench Step 2: 데이터셋 구성 (벡터화 버전 — fast)

입력:
  data/ecbench/raw/swissprot_2018_02.tsv
  data/ecbench/raw/test_ec.csv
  data/ecbench/raw/price149.csv
출력:
  data/ecbench/processed/train_meta.csv
  data/ecbench/processed/test_meta.csv
  data/ecbench/processed/price149_meta.csv
  data/ecbench/label_encoders.pkl
  data/ecbench/splits/{train,val,test,price149}_ids.txt
  data/ecbench/new_proteins.txt
"""
import os, re, pickle, random
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

ROOT       = Path("/home/user/Desktop/unlv")
ECBENCH    = ROOT / "data/ecbench"
RAW        = ECBENCH / "raw"
PROCESSED  = ECBENCH / "processed"
SPLITS     = ECBENCH / "splits"

EMBED_DIR  = ROOT / "data/processed/embeddings"
CMAP_DIR   = ROOT / "data/processed/contact_maps"

SEED      = 42
VAL_RATIO = 0.1

for d in [PROCESSED, SPLITS]:
    d.mkdir(parents=True, exist_ok=True)


def parse_ec_row(ec_str: str):
    """'1.1.1.1; 2.2.2.2' → (l1_set, l2_set, l3_set, l4_set) 각각 '|' 구분 str."""
    ecs = [e.strip() for e in str(ec_str).split(";") if e.strip()]
    l1, l2, l3, l4 = set(), set(), set(), set()
    for ec in ecs:
        parts = ec.split(".")
        if len(parts) < 4:
            continue
        if parts[0] != "-":
            l1.add(parts[0])
        if parts[1] != "-":
            l2.add(f"{parts[0]}.{parts[1]}")
        if parts[2] != "-":
            l3.add(f"{parts[0]}.{parts[1]}.{parts[2]}")
        if parts[3] != "-":
            l4.add(f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}")
    return ("|".join(sorted(l1)), "|".join(sorted(l2)),
            "|".join(sorted(l3)), "|".join(sorted(l4)))


def build_meta_df(raw_df: pd.DataFrame, id_col: str, seq_col: str, ec_col: str) -> pd.DataFrame:
    """원본 DataFrame → l1~l4 문자열 컬럼을 가진 DataFrame."""
    print(f"  EC 파싱 중 ({len(raw_df):,}개)...", flush=True)

    parsed = raw_df[ec_col].map(parse_ec_row)
    meta = pd.DataFrame({
        "accession": raw_df[id_col].astype(str).values,
        "sequence":  raw_df[seq_col].astype(str).values,
        "seq_len":   raw_df[seq_col].astype(str).str.len().values,
        "ec_raw":    raw_df[ec_col].astype(str).values,
        "l1_set":    [x[0] for x in parsed],
        "l2_set":    [x[1] for x in parsed],
        "l3_set":    [x[2] for x in parsed],
        "l4_set":    [x[3] for x in parsed],
    })
    # L4 완전 EC 없으면 제외
    meta = meta[meta["l4_set"] != ""].reset_index(drop=True)
    print(f"  L4 완전 EC 있는 것: {len(meta):,}개", flush=True)
    return meta


def encode_meta_vectorized(meta_df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    벡터화 인코딩 — iterrows() 없이 처리.
    L4 멀티레이블 → '|' 구분 인덱스 문자열.
    """
    print(f"  인코딩 중 ({len(meta_df):,}개)...", flush=True)

    # 인코더 클래스를 집합으로 변환 (1회만)
    cls_sets  = {lv: set(encoders[f"level{lv}"].classes_) for lv in range(1, 5)}
    cls_maps  = {lv: {c: i for i, c in enumerate(encoders[f"level{lv}"].classes_)}
                 for lv in range(1, 5)}

    def encode_l4_idxs(l4_str: str) -> str:
        items = [x for x in l4_str.split("|") if x in cls_sets[4]]
        if not items:
            return ""
        idxs = sorted(cls_maps[4][x] for x in items)
        return "|".join(map(str, idxs))

    def encode_first(lv_str: str, lv: int) -> int:
        items = [x for x in lv_str.split("|") if x in cls_sets[lv]]
        return cls_maps[lv][items[0]] if items else -1

    l4_all_idxs = meta_df["l4_set"].map(encode_l4_idxs)

    # 인코딩 안 된 행 제거 (테스트셋에서 새 EC class일 경우)
    valid = l4_all_idxs != ""
    if not valid.all():
        n_skip = (~valid).sum()
        print(f"  ⚠ {n_skip}개 제외 (학습 EC class에 없는 것)", flush=True)
        meta_df      = meta_df[valid].reset_index(drop=True)
        l4_all_idxs  = l4_all_idxs[valid].reset_index(drop=True)

    l1_idx = meta_df["l1_set"].map(lambda s: encode_first(s, 1))
    l2_idx = meta_df["l2_set"].map(lambda s: encode_first(s, 2))
    l3_idx = meta_df["l3_set"].map(lambda s: encode_first(s, 3))

    # 대표 L4 (첫 번째 인덱스)
    l4_idx = l4_all_idxs.map(lambda s: int(s.split("|")[0]) if s else -1)

    result = meta_df[["accession", "sequence", "seq_len", "ec_raw"]].copy()
    result["l4_all_idxs"] = l4_all_idxs.values
    result["l1_idx"]      = l1_idx.values
    result["l2_idx"]      = l2_idx.values
    result["l3_idx"]      = l3_idx.values
    result["l4_idx"]      = l4_idx.values
    result["m1"]          = 1
    result["m2"]          = 1
    result["m3"]          = 1
    result["m4"]          = 1

    print(f"  완료: {len(result):,}개", flush=True)
    return result


def main():
    # ── 1. 훈련 데이터 ──────────────────────────────────────────
    print("훈련 데이터 로드 중...", flush=True)
    train_raw  = pd.read_csv(RAW / "swissprot_2018_02.tsv", sep="\t")
    print(f"  원본: {len(train_raw):,}개", flush=True)
    train_meta = build_meta_df(train_raw, "accession", "sequence", "ec_number")

    # ── 2. 테스트 데이터 ────────────────────────────────────────
    print("테스트 데이터 로드 중...", flush=True)
    test_raw   = pd.read_csv(RAW / "test_ec.csv")
    test_meta  = build_meta_df(test_raw, "id", "seq", "ec_number")
    price_raw  = pd.read_csv(RAW / "price149.csv")
    price_meta = build_meta_df(price_raw, "id", "seq", "ec_number")

    # ── 3. 라벨 인코더 (이미 있으면 재사용) ────────────────────
    enc_path = ECBENCH / "label_encoders.pkl"
    if enc_path.exists():
        print(f"기존 라벨 인코더 재사용: {enc_path}", flush=True)
        with open(enc_path, "rb") as f:
            encoders = pickle.load(f)
        for k, v in encoders.items():
            print(f"  {k}: {len(v.classes_)}개 클래스", flush=True)
    else:
        print("라벨 인코더 구축 중...", flush=True)
        all_l = {1: set(), 2: set(), 3: set(), 4: set()}
        for col, lv in [("l1_set", 1), ("l2_set", 2), ("l3_set", 3), ("l4_set", 4)]:
            for s in train_meta[col]:
                all_l[lv].update(s.split("|"))
            all_l[lv].discard("")

        encoders = {}
        for lv in range(1, 5):
            enc = LabelEncoder()
            enc.fit(sorted(all_l[lv]))
            encoders[f"level{lv}"] = enc
            print(f"  level{lv}: {len(enc.classes_)}개 클래스", flush=True)

        with open(enc_path, "wb") as f:
            pickle.dump(encoders, f)
        print(f"라벨 인코더 저장: {enc_path}", flush=True)

    # ── 4. 벡터화 인코딩 ────────────────────────────────────────
    print("메타 CSV 인코딩 중...", flush=True)
    train_enc = encode_meta_vectorized(train_meta, encoders)
    test_enc  = encode_meta_vectorized(test_meta,  encoders)
    price_enc = encode_meta_vectorized(price_meta, encoders)

    train_enc.to_csv(PROCESSED / "train_meta.csv",    index=False)
    test_enc.to_csv(PROCESSED  / "test_meta.csv",     index=False)
    price_enc.to_csv(PROCESSED / "price149_meta.csv", index=False)
    print(f"  훈련: {len(train_enc):,}  테스트: {len(test_enc):,}  Price-149: {len(price_enc):,}", flush=True)

    # ── 5. train/val split ──────────────────────────────────────
    random.seed(SEED)
    all_train_ids = train_enc["accession"].tolist()
    random.shuffle(all_train_ids)
    n_val     = int(len(all_train_ids) * VAL_RATIO)
    val_ids   = all_train_ids[:n_val]
    train_ids = all_train_ids[n_val:]

    (SPLITS / "train_ids.txt").write_text("\n".join(train_ids))
    (SPLITS / "val_ids.txt").write_text("\n".join(val_ids))
    (SPLITS / "test_ids.txt").write_text("\n".join(test_enc["accession"].tolist()))
    (SPLITS / "price149_ids.txt").write_text("\n".join(price_enc["accession"].tolist()))
    print(f"  Split: train={len(train_ids):,}  val={len(val_ids):,}", flush=True)

    # ── 6. 임베딩/contact map 재활용 분석 ──────────────────────
    existing_emb  = set(p.stem for p in EMBED_DIR.glob("*.npy"))
    existing_cmap = set(p.stem for p in CMAP_DIR.glob("*.npy"))

    train_id_set = set(train_enc["accession"])
    need_emb  = train_id_set - existing_emb
    need_cmap = train_id_set - existing_cmap

    print(f"\n임베딩 현황:", flush=True)
    print(f"  훈련 {len(train_id_set):,}개 중 재활용={len(train_id_set & existing_emb):,}  새로={len(need_emb):,}", flush=True)
    print(f"\nContact Map 현황:", flush=True)
    print(f"  훈련 {len(train_id_set):,}개 중 재활용={len(train_id_set & existing_cmap):,}  새로={len(need_cmap):,}", flush=True)

    # 테스트/price149 새 임베딩
    all_test_ids = set(test_enc["accession"]) | set(price_enc["accession"])
    new_test_emb = all_test_ids - existing_emb
    print(f"\n테스트셋 새 임베딩 필요: {len(new_test_emb):,}개", flush=True)

    new_proteins = need_emb | need_cmap | new_test_emb
    (ECBENCH / "new_proteins.txt").write_text("\n".join(sorted(new_proteins)))
    print(f"새 단백질 목록 저장: {len(new_proteins):,}개", flush=True)

    print("\n완료!", flush=True)


if __name__ == "__main__":
    main()
