"""
Step 2: EC 라벨 파싱, 인코딩, Train/Val/Test 분할
핵심 설계:
  - 다중 EC → 가장 완전한 것(Level4 완전) 우선 선택
  - Level별 라벨: 개별 숫자가 아닌 EC PREFIX 문자열 사용
      l1_str = "3"       → 7 classes
      l2_str = "3.2"     → ~70 classes
      l3_str = "3.2.2"   → ~300 classes
      l4_str = "3.2.2.6" → ~1938 classes  (HIT-EC 비교 기준)
  - mask: 해당 레벨이 완전 숫자이면 1, '-' 이면 0
"""
import re, pickle, yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

RAW_TSV  = ROOT / CFG["paths"]["raw_tsv"]
META_CSV = ROOT / CFG["paths"]["meta_csv"]
ENC_PKL  = ROOT / CFG["paths"]["label_enc"]
SPL_DIR  = ROOT / CFG["paths"]["splits_dir"]

Path(META_CSV).parent.mkdir(parents=True, exist_ok=True)
SPL_DIR.mkdir(parents=True, exist_ok=True)

SEED     = CFG["data"]["seed"]
MIN_LEN  = CFG["data"]["min_seq_len"]
MAX_LEN  = CFG["data"]["max_seq_len"]


# ── EC 파싱 헬퍼 ──────────────────────────────────────────────
def _score_ec(ec: str) -> int:
    """완전한 레벨 수(0~4) 반환 — 완전할수록 높은 점수."""
    parts = ec.split(".")
    if len(parts) != 4:
        return 0
    return sum(1 for p in parts if re.match(r"^\d+$", p.strip()))


def best_ec(ec_raw: str):
    """
    세미콜론으로 구분된 다중 EC에서 가장 완전한 것 선택.
    동점이면 첫 번째.
    """
    if pd.isna(ec_raw) or str(ec_raw).strip() == "":
        return None
    candidates = [e.strip() for e in str(ec_raw).split(";") if e.strip()]
    if not candidates:
        return None
    return max(candidates, key=_score_ec)


def parse_ec(ec_str: str):
    """
    'x.y.z.w' → prefix 문자열과 마스크.
    반환: (l1_str, l2_str, l3_str, l4_str, m1, m2, m3, m4)
    Level i prefix는 숫자 부분까지만 포함 (마지막 '-' 제외).
    """
    parts = ec_str.split(".")
    if len(parts) != 4:
        return None

    prefixes, masks = [], []
    valid_so_far = True
    for i, p in enumerate(parts):
        p = p.strip()
        if re.match(r"^\d+$", p) and valid_so_far:
            prefix = ".".join(parts[:i+1]).replace(".-", "")
            prefixes.append(".".join(p2.strip() for p2 in parts[:i+1]))
            masks.append(1)
        else:
            valid_so_far = False
            prefixes.append(None)
            masks.append(0)

    return (*prefixes, *masks)   # l1,l2,l3,l4, m1,m2,m3,m4


def load_and_filter():
    print(f"TSV 읽는 중: {RAW_TSV}")
    df = pd.read_csv(RAW_TSV, sep="\t", low_memory=False)
    print(f"  원본 행 수: {len(df):,}")

    # 컬럼 표준화
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if "entry" in lc and "name" not in lc and "id" not in lc:
            rename[c] = "accession"
        elif "sequence" == lc:
            rename[c] = "sequence"
        elif "ec" in lc and "number" in lc:
            rename[c] = "ec_raw"
        elif lc == "length":
            rename[c] = "length"
        elif "organism" in lc and "id" in lc:
            rename[c] = "organism_id"
    df = df.rename(columns=rename)
    # fallback: Entry → accession
    if "accession" not in df.columns and "Entry" in df.columns:
        df = df.rename(columns={"Entry": "accession"})
    if "sequence" not in df.columns and "Sequence" in df.columns:
        df = df.rename(columns={"Sequence": "sequence"})
    if "ec_raw" not in df.columns and "EC number" in df.columns:
        df = df.rename(columns={"EC number": "ec_raw"})

    print(f"  컬럼: {list(df.columns)}")

    records = []
    stat = {"no_ec": 0, "bad_len": 0, "parse_fail": 0, "ok": 0}

    for _, row in df.iterrows():
        acc = str(row.get("accession", "")).strip()
        seq = str(row.get("sequence", "")).strip()

        if not acc or acc == "nan":
            continue
        slen = len(seq)
        if slen < MIN_LEN or slen > MAX_LEN or seq == "nan":
            stat["bad_len"] += 1
            continue

        ec_raw = row.get("ec_raw", "")
        chosen = best_ec(ec_raw)
        if chosen is None:
            stat["no_ec"] += 1
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

    meta = pd.DataFrame(records).drop_duplicates("accession").reset_index(drop=True)
    print(f"  통계: ok={stat['ok']:,}  no_ec={stat['no_ec']:,}  "
          f"bad_len={stat['bad_len']:,}  parse_fail={stat['parse_fail']:,}")
    print(f"  최종 단백질: {len(meta):,}")
    return meta


def encode_labels(meta):
    print("\n라벨 인코딩 중...")
    encoders = {}
    for lvl in range(1, 5):
        str_col = f"l{lvl}_str"
        m_col   = f"m{lvl}"

        valid   = meta.loc[meta[m_col] == 1, str_col].dropna().unique()
        le      = LabelEncoder()
        le.fit(sorted(valid))
        encoders[f"level{lvl}"] = le

        idx = meta[str_col].copy()
        valid_mask = (meta[m_col] == 1) & meta[str_col].notna()
        idx[valid_mask]  = le.transform(meta.loc[valid_mask, str_col])
        idx[~valid_mask] = -1
        meta[f"l{lvl}_idx"] = idx.astype(int)

        full_only = meta[m_col].sum()
        print(f"  Level {lvl}: {len(le.classes_):>5} 클래스  "
              f"(유효 단백질 {full_only:,}개  /  전체 {len(meta):,})")
    return meta, encoders


def split_data(meta):
    print("\nTrain/Val/Test 분할 (80/10/10) ...")
    ids = meta["accession"].tolist()
    tr, tmp = train_test_split(ids, test_size=0.2, random_state=SEED)
    va, te  = train_test_split(tmp, test_size=0.5, random_state=SEED)
    print(f"  Train: {len(tr):,}  Val: {len(va):,}  Test: {len(te):,}")
    return tr, va, te


def main():
    meta = load_and_filter()
    meta, encoders = encode_labels(meta)

    print("\n[EC Level 1 분포]")
    dist = meta[meta["m1"] == 1]["l1_str"].value_counts().sort_index()
    for k, v in dist.items():
        print(f"  EC {k}: {v:,}")

    print(f"\n[Level 4 완전 EC 수: {meta['m4'].sum():,}]")
    print(f"  HIT-EC 비교 가능한 샘플 (Level4 완전): {meta['m4'].sum():,}")

    train_ids, val_ids, test_ids = split_data(meta)

    # 저장
    meta.to_csv(META_CSV, index=False)
    print(f"\nMeta CSV 저장: {META_CSV}  ({len(meta):,} rows)")

    with open(ENC_PKL, "wb") as f:
        pickle.dump(encoders, f)
    print(f"Label encoder 저장: {ENC_PKL}")

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        path = SPL_DIR / f"{name}_ids.txt"
        path.write_text("\n".join(ids))
        print(f"{name} ids: {path}")

    # 요약
    print("\n=== 최종 데이터 요약 ===")
    print(f"  총 단백질     : {len(meta):,}")
    print(f"  Level 1 클래스: {len(encoders['level1'].classes_)}")
    print(f"  Level 2 클래스: {len(encoders['level2'].classes_)}")
    print(f"  Level 3 클래스: {len(encoders['level3'].classes_)}")
    print(f"  Level 4 클래스: {len(encoders['level4'].classes_)} "
          f"(HIT-EC 기준 ~1938)")


if __name__ == "__main__":
    main()
