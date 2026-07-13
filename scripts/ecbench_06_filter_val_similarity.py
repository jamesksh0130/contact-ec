"""
EC-Bench Step 6: Validation Set Similarity Filtering

문제:
  현재 val set = train에서 무작위 10% 분할
  → train-val sequence similarity 높음 → 검증 F1 인플레이션

해결:
  MMSeqs2로 train vs val pairwise similarity 계산
  → val 단백질 중 train 최고 유사도 < 75% (sequence identity) 인 것만 유지
  → 진짜 일반화 성능을 평가하는 "hard" validation set

참고: CLEAN 논문의 split 방식 (Yu et al., 2023, Science)
"""
import subprocess, os, shutil
from pathlib import Path
import pandas as pd

ROOT       = Path("/home/user/Desktop/unlv")
ECBENCH    = ROOT / "data/ecbench"
SPLITS     = ECBENCH / "splits"
MMSEQS     = "/home/user/anaconda3/bin/mmseqs"

# 유사도 threshold (< 75% identity → hard val)
SIM_THRESHOLD = 0.75

TMP = Path("/tmp/ecbench_mmseqs")
TMP.mkdir(exist_ok=True)


def write_fasta(ids: list, meta_df: pd.DataFrame, out_path: Path):
    with open(out_path, "w") as f:
        for uid in ids:
            if uid in meta_df.index:
                seq = meta_df.loc[uid, "sequence"]
                f.write(f">{uid}\n{seq}\n")


def run_mmseqs_search(query_fasta: Path, db_fasta: Path, out_tsv: Path):
    """MMSeqs2 easy-search: query vs db → identity 결과."""
    db_dir   = TMP / "db"
    tmp_dir  = TMP / "tmp"
    db_dir.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)

    cmd = [
        MMSEQS, "easy-search",
        str(query_fasta), str(db_fasta), str(out_tsv), str(tmp_dir),
        "--min-seq-id", "0.0",   # 모든 유사도 출력
        "--alignment-mode", "3", # 전체 alignment
        "--format-output", "query,target,fident",
        "-c", "0.8",             # 80% coverage
        "--cov-mode", "0",
        "--threads", "16",
        "-s", "7.5",             # sensitivity
    ]
    print(f"MMSeqs2 실행 중: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDERR:", result.stderr[:500])
        raise RuntimeError(f"MMSeqs2 실패 (exit {result.returncode})")
    print(f"  결과: {out_tsv}")
    return out_tsv


def main():
    print("=== Validation Similarity Filtering ===")

    # 메타 데이터 로드
    meta = pd.read_csv(ECBENCH / "processed/train_meta.csv").set_index("accession")

    train_ids = (SPLITS / "train_ids.txt").read_text().strip().split("\n")
    val_ids   = (SPLITS / "val_ids.txt").read_text().strip().split("\n")
    print(f"Train: {len(train_ids):,}개, Val: {len(val_ids):,}개")

    # FASTA 파일 생성
    train_fasta = TMP / "train.fasta"
    val_fasta   = TMP / "val.fasta"
    write_fasta(train_ids, meta, train_fasta)
    write_fasta(val_ids,   meta, val_fasta)
    print(f"FASTA 생성: {train_fasta} ({len(train_ids):,}개)")
    print(f"FASTA 생성: {val_fasta} ({len(val_ids):,}개)")

    # MMSeqs2: val → train 검색
    result_tsv = TMP / "val_vs_train.tsv"
    run_mmseqs_search(val_fasta, train_fasta, result_tsv)

    # 결과 파싱: 각 val 단백질의 최고 identity
    print("결과 파싱 중...")
    hits = pd.read_csv(result_tsv, sep="\t", header=None,
                       names=["query", "target", "fident"])
    max_identity = hits.groupby("query")["fident"].max()

    # 필터링: max identity < threshold인 val 단백질만 유지
    hard_val = [v for v in val_ids
                if max_identity.get(v, 0.0) < SIM_THRESHOLD]
    easy_val = [v for v in val_ids
                if max_identity.get(v, 0.0) >= SIM_THRESHOLD]
    no_hit   = [v for v in val_ids if v not in max_identity]

    print(f"\n필터링 결과 (threshold={SIM_THRESHOLD:.0%}):")
    print(f"  Hard val (< {SIM_THRESHOLD:.0%} identity): {len(hard_val):,}개")
    print(f"  Easy val (≥ {SIM_THRESHOLD:.0%} identity): {len(easy_val):,}개")
    print(f"  Hit 없음 (MMSeqs2 미매칭): {len(no_hit):,}개")

    # hard_val에 no_hit 추가 (매칭 없음 = 매우 다름 → hard)
    hard_val_all = hard_val + no_hit

    # 저장
    (SPLITS / "val_hard_ids.txt").write_text("\n".join(hard_val_all))
    (SPLITS / "val_easy_ids.txt").write_text("\n".join(easy_val))
    print(f"\nHard val 저장: {SPLITS / 'val_hard_ids.txt'} ({len(hard_val_all):,}개)")
    print(f"Easy val 저장: {SPLITS / 'val_easy_ids.txt'} ({len(easy_val):,}개)")

    # 통계
    if len(max_identity) > 0:
        import numpy as np
        identities = [max_identity.get(v, 0.0) for v in val_ids]
        print(f"\n유사도 분포:")
        print(f"  평균: {np.mean(identities):.3f}")
        print(f"  중앙값: {np.median(identities):.3f}")
        print(f"  90th percentile: {np.percentile(identities, 90):.3f}")

    print("\n완료!")


if __name__ == "__main__":
    main()
