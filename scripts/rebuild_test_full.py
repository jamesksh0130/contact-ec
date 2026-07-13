"""
전체 테스트셋 재빌드 스크립트
- test_ec.csv 468개 → 알려진 EC 보유 단백질 최대 123개 추출
- 멀티 EC 라벨 올바르게 처리 (l4_all_idxs)
- 임베딩 없는 A0A3G9HRC2 ESM-2 추출
- test_meta_full.csv + test_ids_full.txt 생성
"""
import sys, pickle, json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config_ecbench.yaml") as f:
    CFG = yaml.safe_load(f)

RAW_TEST  = ROOT / "data" / "ecbench" / "raw" / "test_ec.csv"
LABEL_ENC = ROOT / CFG["paths"]["label_enc"]
EMB_DIR   = ROOT / CFG["paths"]["embed_dir"]
SPLITS    = ROOT / CFG["paths"]["splits_dir"]
PROC_DIR  = ROOT / "data" / "ecbench" / "processed"


# ── 라벨 인코더 로드 ──────────────────────────────────────────────────────────
with open(LABEL_ENC, "rb") as f:
    encoders = pickle.load(f)

l1_enc = encoders["level1"]
l2_enc = encoders["level2"]
l3_enc = encoders["level3"]
l4_enc = encoders["level4"]
l4_set = set(l4_enc.classes_)

def safe_encode(enc, key):
    try:
        return int(enc.transform([key])[0])
    except Exception:
        return -1


# ── EC 파싱 유틸 ─────────────────────────────────────────────────────────────
def parse_all_ecs(ec_raw):
    """'3.2.1.4,3.2.1.17;2.7.7.7' 형태에서 EC 리스트 반환."""
    return [e.strip() for e in ec_raw.replace(";", ",").split(",") if e.strip()]

def ec_level(ec, lvl):
    """EC string → level-N prefix (e.g. '1.2.3.4' lvl=2 → '1.2')"""
    parts = ec.split(".")
    if len(parts) < lvl or parts[lvl-1] in ["-", "", "nan"]:
        return None
    return ".".join(parts[:lvl])


# ── 전체 468개 → 알려진 EC 단백질만 필터링 ──────────────────────────────────
raw_df = pd.read_csv(RAW_TEST)
rows_out = []

for _, row in raw_df.iterrows():
    uid    = row["id"]
    ec_raw = str(row["ec_number"])
    seq    = str(row["seq"])
    ecs    = parse_all_ecs(ec_raw)

    # 알려진 L4 클래스만 모음
    known_l4 = [e for e in ecs if e in l4_set]
    if not known_l4:
        continue  # 부분 EC 또는 신규 EC → 스킵

    # 대표 EC (첫 번째 알려진 것)
    primary = known_l4[0]
    l4_idxs = sorted(set(safe_encode(l4_enc, e) for e in known_l4
                         if safe_encode(l4_enc, e) >= 0))

    l1_key = ec_level(primary, 1)
    l2_key = ec_level(primary, 2)
    l3_key = ec_level(primary, 3)

    l1_idx = safe_encode(l1_enc, l1_key) if l1_key else -1
    l2_idx = safe_encode(l2_enc, l2_key) if l2_key else -1
    l3_idx = safe_encode(l3_enc, l3_key) if l3_key else -1
    l4_idx = l4_idxs[0] if l4_idxs else -1

    rows_out.append({
        "accession":   uid,
        "sequence":    seq,
        "seq_len":     len(seq),
        "ec_raw":      ec_raw,
        "l4_all_idxs": "|".join(str(i) for i in l4_idxs),
        "l1_idx":      l1_idx,
        "l2_idx":      l2_idx,
        "l3_idx":      l3_idx,
        "l4_idx":      l4_idx,
        "m1":          1 if l1_idx >= 0 else 0,
        "m2":          1 if l2_idx >= 0 else 0,
        "m3":          1 if l3_idx >= 0 else 0,
        "m4":          1 if l4_idx >= 0 else 0,
    })

meta_df = pd.DataFrame(rows_out)
print(f"알려진 EC 단백질: {len(meta_df)}개")

# 임베딩 현황
meta_df["has_emb"] = meta_df["accession"].apply(
    lambda uid: (EMB_DIR / f"{uid}.npy").exists()
)
print(f"  임베딩 있음: {meta_df['has_emb'].sum()}개")
print(f"  임베딩 없음: {(~meta_df['has_emb']).sum()}개")
missing_emb = meta_df[~meta_df["has_emb"]]["accession"].tolist()
if missing_emb:
    print(f"  없는 ID: {missing_emb}")


# ── 임베딩 없는 단백질 ESM-2 추출 ────────────────────────────────────────────
if missing_emb:
    print(f"\nESM-2 임베딩 추출 중: {missing_emb}")
    import torch
    from transformers import AutoTokenizer, EsmModel

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_ID = CFG["model"]["esm2_model"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    esm2      = EsmModel.from_pretrained(MODEL_ID).eval().to(DEVICE)

    seq_map = dict(zip(meta_df["accession"], meta_df["sequence"]))
    for uid in missing_emb:
        seq = seq_map[uid]
        inputs = tokenizer(seq, return_tensors="pt",
                           truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            out = esm2(**inputs)
        emb = out.last_hidden_state[:, 0, :].squeeze().cpu().float().numpy()
        np.save(EMB_DIR / f"{uid}.npy", emb)
        print(f"  ✓ {uid}  shape={emb.shape}")

    # 임베딩 현황 업데이트
    meta_df["has_emb"] = meta_df["accession"].apply(
        lambda uid: (EMB_DIR / f"{uid}.npy").exists()
    )
    print(f"추출 후 임베딩 있음: {meta_df['has_emb'].sum()}개")


# ── CSV / ID 파일 저장 ────────────────────────────────────────────────────────
eval_df = meta_df[meta_df["has_emb"]].drop(columns=["has_emb"]).reset_index(drop=True)
print(f"\n최종 평가 가능: {len(eval_df)}개")

out_meta = PROC_DIR / "test_meta_full.csv"
out_ids  = SPLITS   / "test_ids_full.txt"
eval_df.to_csv(out_meta, index=False)
out_ids.write_text("\n".join(eval_df["accession"].tolist()))
print(f"저장: {out_meta}")
print(f"저장: {out_ids}")

# 멀티 EC 단백질 통계
multi_label = eval_df[eval_df["l4_all_idxs"].str.contains("|", regex=False)]
print(f"\n멀티 EC 단백질: {len(multi_label)}개 (이전 평가에서 누락됐던 것)")
print(f"단일 EC 단백질: {len(eval_df) - len(multi_label)}개")
print("\n멀티 EC 예시:")
print(multi_label[["accession","ec_raw","l4_all_idxs"]].head(5).to_string())
