#!/bin/bash
# EC-Bench 전체 파이프라인 실행 스크립트
# 사용법: bash scripts/ecbench_run_pipeline.sh [GPU_ID]

set -e
GPU=${1:-0}
ROOT="/home/user/Desktop/unlv"
cd "$ROOT"

echo "========================================"
echo "  EC-Bench 전체 파이프라인"
echo "  GPU: $GPU"
echo "========================================"

# Step 1: 파싱 (이미 완료됐을 경우 skip)
if [ ! -f "data/ecbench/raw/swissprot_2018_02.tsv" ]; then
    echo "[Step 1] Swiss-Prot 2018-02 파싱..."
    python scripts/ecbench_01_parse_train.py
else
    echo "[Step 1] 파싱 완료 (skip)"
fi

# Step 2: 데이터셋 구성
if [ ! -f "data/ecbench/processed/train_meta.csv" ]; then
    echo "[Step 2] 데이터셋 구성..."
    python scripts/ecbench_02_build_dataset.py
else
    echo "[Step 2] 데이터셋 구성 완료 (skip)"
fi

# Step 3: 새 임베딩 추출
if [ -f "data/ecbench/new_proteins.txt" ]; then
    echo "[Step 3] 새 단백질 ESM-2 임베딩 추출..."
    python scripts/ecbench_03_extract_new_embeddings.py
else
    echo "[Step 3] new_proteins.txt 없음 (skip)"
fi

# Step 4: 새 PDB 다운로드
echo "[Step 4] 새 PDB 다운로드..."
python scripts/ecbench_04_download_new_pdbs.py

# Step 5: 새 contact map 생성
echo "[Step 5] 새 contact map 생성..."
python scripts/ecbench_05_build_new_contact_maps.py

# Step 6: B2 baseline 학습
echo "[Step 6] B2 (ESM-2 Hierarchical) 학습..."
python train.py --model b2_esm2_hier \
    --config config_ecbench.yaml \
    --epochs 30 --gpu $GPU \
    --tag ecbench_b2_phase1 \
    2>&1 | tee outputs/logs/ecbench_b2.log

# Step 7: Fusion V2 Phase 1 학습
echo "[Step 7] Fusion V2 Phase 1 학습..."
python train.py --model fusion_v2 \
    --config config_ecbench.yaml \
    --epochs 30 --gpu $GPU \
    --tag ecbench_fusion_v2_phase1 \
    2>&1 | tee outputs/logs/ecbench_fusion_v2_p1.log

# Step 8: Fusion V2 Phase 2 학습
echo "[Step 8] Fusion V2 Phase 2 학습..."
python train.py --model fusion_v2 \
    --config config_ecbench.yaml \
    --epochs 20 --phase 2 --gpu $GPU \
    --tag ecbench_fusion_v2_phase2 \
    --resume outputs/checkpoints/ecbench_fusion_v2_phase1_best.pt \
    2>&1 | tee outputs/logs/ecbench_fusion_v2_p2.log

# Step 9: 평가
echo "[Step 9] EC-Bench 평가..."
python scripts/ecbench_eval.py \
    --checkpoint outputs/checkpoints/ecbench_fusion_v2_phase2_best.pt \
    --model fusion_v2

python scripts/ecbench_eval.py \
    --checkpoint outputs/checkpoints/ecbench_b2_phase1_best.pt \
    --model b2_esm2_hier

echo ""
echo "========================================"
echo "  EC-Bench 파이프라인 완료!"
echo "========================================"
