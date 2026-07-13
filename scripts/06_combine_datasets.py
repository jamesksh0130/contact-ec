"""
Step 6: 2018 기존 데이터 + 2022 신규 데이터 통합
- 기존: data/processed/dataset_meta.csv  (~270K)
- 신규: data/raw/hitec_new_ec.tsv         (~43K)
- 출력: data/processed/dataset_meta_2022.csv
        data/label_encoders_2022.pkl
        data/splits_2022/{train,val,test}_ids.txt

EC-Bench 테스트 셋(124개)은 별도 유지 — 재학습 후 동일 셋으로 평가
"""
import re, pickle, yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

ROOT = Path("/file/user/Desktop/unlv").resolve()
ROOT = Path("/home/user/Desktop/unlv")

OLD_META   = ROOT / "data/processed/dataset_meta.csv"
NEW_EC_TSV = ROOT / "data/raw/hitec_new_ec.tsv"
ECBENCH_TEST = ROOT / "data/ecbench/processed/test_meta_full.csv"

OUT_META   = ROOT / "data/processed/dataset_meta_2022.csv"
OUT_ENC    = ROOT / "data/label_encoders_2022.pkl"
OUT_SPLITS = ROOT / "data/splits_2022"
OUT_SPLITS.mkdir(parents=True, exist_ok=True)

SEED = 42
MIN_LEN = 10
MAX_LEN = 1024


def _score_ec(ec: str) -> int:
    parts = ec.split(".")
    if len(parts) != 4:
        return 0
    return sum(1 for p in parts if re.match(r"^\d+$", p.strip()))


def best_ec(ec_raw: str):
    if pd.isna(ec_raw) or str(ec_raw).strip() == "":
        return None
    candidates = [e.strip() for e in str(ec_raw).split(";") if e.strip()]
    return max(candidates, key=_score_ec) if candidates else None


def parse_ec(ec_str: str):
    parts = ec_str.split(".")
    if len(parts) != 4:
        return None
    prefixes, masks = [], []
    valid_so_far = True
    for i, p in enumerate(parts):
        p = p.strip()
        if re.match(r"^\d+$", p) and valid_so_far:
            prefixes.append(".".join(pt.strip() for pt in parts[:i+1]))
            masks.append(1)
        else:
            valid_so_far = False
            prefixes.append(None)
            masks.append(0)
    return (*prefixes, *masks)


# ── 1. 기존 메타 로드 ───────────────────────────────────────────
print("기존 메타 로드...")
old = pd.read_csv(OLD_META)
print(f"  기존: {len(old):,} 단백질")

# EC-Bench 테스트 셋 accession
test_meta = pd.read_csv(ECBENCH_TEST)
test_accs = set(test_meta["accession"].tolist())
print(f"  EC-Bench 테스트 셋: {len(test_accs):,} 단백질 (학습에서 제외)")

# 테스트 단백질을 기존 메타에서도 제거 (leakage 방지)
old_train = old[~old["accession"].isin(test_accs)].copy()
print(f"  기존 (테스트 제외): {len(old_train):,}")

# ── 2. 신규 데이터 파싱 ─────────────────────────────────────────
print("\n신규 데이터 파싱 중...")
new_df = pd.read_csv(NEW_EC_TSV, sep="\t")
print(f"  신규 원본: {len(new_df):,}")

# 이미 기존 메타에 있는 accession 제외
old_accs = set(old_train["accession"].tolist())
new_df = new_df[~new_df["accession"].isin(old_accs)]
new_df = new_df[~new_df["accession"].isin(test_accs)]
print(f"  신규 (중복/테스트 제외): {len(new_df):,}")

records = []
stat = {"ok": 0, "bad_len": 0, "parse_fail": 0}

for _, row in new_df.iterrows():
    acc = str(row["accession"]).strip()
    seq = str(row["sequence"]).strip() if isinstance(row["sequence"], str) else ""
    if not seq or seq == "nan":
        continue
    slen = len(seq)
    if slen < MIN_LEN or slen > MAX_LEN:
        stat["bad_len"] += 1
        continue

    chosen = best_ec(str(row["ec"]))
    if chosen is None:
        continue

    parsed = parse_ec(chosen)
    if parsed is None:
        stat["parse_fail"] += 1
        continue

    l1, l2, l3, l4, m1, m2, m3, m4 = parsed
    records.append({
        "accession": acc,
        "sequence":  seq,
        "seq_len":   slen,
        "ec_chosen": chosen,
        "l1_str": l1, "l2_str": l2, "l3_str": l3, "l4_str": l4,
        "m1": m1,    "m2": m2,    "m3": m3,    "m4": m4,
    })
    stat["ok"] += 1

new_clean = pd.DataFrame(records)
print(f"  파싱 OK: {stat['ok']:,}  bad_len: {stat['bad_len']:,}  parse_fail: {stat['parse_fail']:,}")

# ── 3. 통합 ────────────────────────────────────────────────────
# 기존 메타는 이미 인코딩된 인덱스 컬럼 보유 → 제거 후 재인코딩
shared_cols = ["accession","sequence","seq_len","ec_chosen",
               "l1_str","l2_str","l3_str","l4_str",
               "m1","m2","m3","m4"]
combined = pd.concat([
    old_train[shared_cols],
    new_clean[shared_cols]
], ignore_index=True).drop_duplicates("accession").reset_index(drop=True)
print(f"\n통합 데이터셋: {len(combined):,} 단백질")

# ── 4. 라벨 인코딩 (재피팅) ──────────────────────────────────────
print("\n라벨 인코딩 재피팅...")
encoders = {}
for lvl in range(1, 5):
    str_col = f"l{lvl}_str"
    m_col   = f"m{lvl}"
    valid   = combined.loc[combined[m_col] == 1, str_col].dropna().astype(str).unique()
    le = LabelEncoder()
    le.fit(sorted(valid))
    encoders[f"level{lvl}"] = le
    idx_col = combined[str_col].copy()
    vmask = (combined[m_col] == 1) & combined[str_col].notna()
    idx_col[vmask]  = le.transform(combined.loc[vmask, str_col])
    idx_col[~vmask] = -1
    combined[f"l{lvl}_idx"] = idx_col.astype(int)
    print(f"  Level {lvl}: {len(le.classes_):>5} 클래스  "
          f"(유효: {combined[m_col].sum():,})")

# l4_all_idxs (단순히 l4_idx 복사 — 다중라벨은 필요시 확장)
combined["l4_all_idxs"] = combined["l4_idx"]

# ── 5. Train/Val 분할 ───────────────────────────────────────────
# 테스트 셋 = EC-Bench temporal (기존과 동일)
from sklearn.model_selection import train_test_split

all_ids = combined["accession"].tolist()
tr_ids, va_ids = train_test_split(all_ids, test_size=0.1, random_state=SEED)
print(f"\n분할: Train={len(tr_ids):,}  Val={len(va_ids):,}  "
      f"Test(EC-Bench)={len(test_accs):,}")

# ── 6. 저장 ─────────────────────────────────────────────────────
combined.to_csv(OUT_META, index=False)
print(f"\nMeta 저장: {OUT_META}  ({len(combined):,} rows)")

with open(OUT_ENC, "wb") as f:
    pickle.dump(encoders, f)
print(f"Encoder 저장: {OUT_ENC}")

for name, ids in [("train", tr_ids), ("val", va_ids)]:
    path = OUT_SPLITS / f"{name}_ids.txt"
    path.write_text("\n".join(ids))
    print(f"{name} ids: {path}  ({len(ids):,})")

# EC-Bench 테스트 메타도 새 인코더로 재인코딩
print("\nEC-Bench 테스트 셋 재인코딩...")
test_meta_new = test_meta.copy()
for lvl in range(1, 5):
    str_col = f"l{lvl}_str" if f"l{lvl}_str" in test_meta_new.columns else None
    if str_col is None:
        continue
    le = encoders[f"level{lvl}"]
    vmask = (test_meta_new[f"m{lvl}"] == 1) & test_meta_new[str_col].notna()
    known = test_meta_new.loc[vmask, str_col].isin(le.classes_)
    idx_col = test_meta_new[str_col].copy()
    idx_col[vmask & known]   = le.transform(test_meta_new.loc[vmask & known, str_col])
    idx_col[vmask & ~known]  = -1
    idx_col[~vmask]          = -1
    test_meta_new[f"l{lvl}_idx"] = idx_col.astype(int)

out_test = ROOT / "data/ecbench/processed/test_meta_full_2022enc.csv"
test_meta_new.to_csv(out_test, index=False)
print(f"  테스트 메타 (2022 인코더): {out_test}")

print("\n=== 완료 ===")
print(f"  학습 데이터: {len(combined):,}")
print(f"  Level 4 클래스: {len(encoders['level4'].classes_):,}")
print(f"  (기존 5307 → 새로운 클래스 수)")
