"""
클러스터 기반 Train/Val/Test 분할
- MMseqs2로 30% 서열 동일성 기준 클러스터링
- 클러스터 단위로 분할 → 서열 유사 단백질이 같은 셋에만 존재
- 출력: data/splits/cluster_{train,val,test}_ids.txt

사용법:
  python scripts/cluster_split.py
"""
import os, sys, yaml, subprocess, random, shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

META_CSV   = ROOT / CFG["paths"]["meta_csv"]
SPLITS_DIR = ROOT / CFG["paths"]["splits_dir"]
TMP_DIR    = ROOT / "tmp_mmseqs"
SEED       = 42

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    TMP_DIR.mkdir(exist_ok=True)

    print("데이터 로드 중...")
    meta = pd.read_csv(META_CSV)
    meta = meta[meta["sequence"].notna() & (meta["sequence"].str.len() >= 4)].copy()
    print(f"  총 단백질: {len(meta):,}")

    # ── FASTA 파일 생성 ──────────────────────────────
    fasta_path = TMP_DIR / "all_proteins.fasta"
    print(f"FASTA 파일 생성 중: {fasta_path}")
    with open(fasta_path, "w") as f:
        for _, row in meta.iterrows():
            seq = str(row["sequence"])[:1024]   # MMseqs2도 1024로 제한
            f.write(f">{row['accession']}\n{seq}\n")
    print(f"  {len(meta):,}개 단백질 작성 완료")

    # ── MMseqs2 클러스터링 ──────────────────────────
    db_path      = str(TMP_DIR / "seqdb")
    cluster_path = str(TMP_DIR / "clusters")
    tsv_path     = str(TMP_DIR / "cluster_result.tsv")
    tmp_path     = str(TMP_DIR / "mmseqs_tmp")

    print("\nMMseqs2 DB 생성 중...")
    subprocess.run([
        "mmseqs", "createdb", str(fasta_path), db_path
    ], check=True, capture_output=True)

    print("클러스터링 중 (30% sequence identity, coverage 0.8)...")
    subprocess.run([
        "mmseqs", "cluster",
        db_path, cluster_path, tmp_path,
        "--min-seq-id", "0.30",
        "-c", "0.80",
        "--cov-mode", "0",
        "--cluster-mode", "1",
        "--threads", "16",
    ], check=True)

    print("TSV 변환 중...")
    subprocess.run([
        "mmseqs", "createtsv",
        db_path, db_path, cluster_path, tsv_path
    ], check=True, capture_output=True)

    # ── 클러스터 파싱 ───────────────────────────────
    print("\n클러스터 파싱 중...")
    cluster_map = defaultdict(list)   # rep → [members]
    with open(tsv_path) as f:
        for line in f:
            rep, member = line.strip().split("\t")
            cluster_map[rep].append(member)

    clusters    = list(cluster_map.values())
    n_clusters  = len(clusters)
    print(f"  총 클러스터 수: {n_clusters:,}")
    sizes = [len(c) for c in clusters]
    print(f"  클러스터 크기: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")

    # ── 클러스터 단위 분할 (8:1:1) ──────────────────
    random.shuffle(clusters)
    n_val  = max(1, int(n_clusters * 0.1))
    n_test = max(1, int(n_clusters * 0.1))

    val_clusters   = clusters[:n_val]
    test_clusters  = clusters[n_val:n_val + n_test]
    train_clusters = clusters[n_val + n_test:]

    train_ids = [acc for c in train_clusters for acc in c]
    val_ids   = [acc for c in val_clusters   for acc in c]
    test_ids  = [acc for c in test_clusters  for acc in c]

    # accession이 meta에 있는 것만 유지
    valid_accs = set(meta["accession"].tolist())
    train_ids  = [a for a in train_ids if a in valid_accs]
    val_ids    = [a for a in val_ids   if a in valid_accs]
    test_ids   = [a for a in test_ids  if a in valid_accs]

    print(f"\n분할 결과:")
    print(f"  Train: {len(train_ids):,} proteins ({len(train_clusters):,} clusters)")
    print(f"  Val  : {len(val_ids):,} proteins ({len(val_clusters):,} clusters)")
    print(f"  Test : {len(test_ids):,} proteins ({len(test_clusters):,} clusters)")
    print(f"  합계 : {len(train_ids)+len(val_ids)+len(test_ids):,} / {len(meta):,}")

    # ── 저장 ────────────────────────────────────────
    SPLITS_DIR.mkdir(exist_ok=True)
    Path(SPLITS_DIR / "cluster_train_ids.txt").write_text("\n".join(train_ids))
    Path(SPLITS_DIR / "cluster_val_ids.txt").write_text("\n".join(val_ids))
    Path(SPLITS_DIR / "cluster_test_ids.txt").write_text("\n".join(test_ids))

    print(f"\n저장 완료:")
    print(f"  {SPLITS_DIR}/cluster_train_ids.txt")
    print(f"  {SPLITS_DIR}/cluster_val_ids.txt")
    print(f"  {SPLITS_DIR}/cluster_test_ids.txt")

    # 임시 파일 정리
    shutil.rmtree(TMP_DIR)
    print("\n클러스터 분할 완료!")


if __name__ == "__main__":
    main()
