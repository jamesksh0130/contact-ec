#!/bin/bash
# B2 (ESM-2 + Hierarchical FC) 멀티레이블 재훈련
# contact map 없음 → 배치 크기 크게 설정 가능
set -e
cd /home/user/Desktop/unlv

LOG="outputs/results/b2_multilabel.log"
echo "=== B2 멀티레이블 훈련 시작: $(date) ===" | tee "$LOG"

python train.py \
    --model b2_esm2_hier \
    --phase 1 \
    --epochs 30 \
    --gpu 1 \
    --batch_size 512 \
    --tag b2_ml \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== B2 테스트 평가 ===" | tee -a "$LOG"

python evaluate.py \
    --model b2_esm2_hier \
    --checkpoint outputs/checkpoints/b2_ml_best.pt \
    --split test \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== B2 클러스터 테스트 평가 (유사도 <50% split) ===" | tee -a "$LOG"

python evaluate.py \
    --model b2_esm2_hier \
    --checkpoint outputs/checkpoints/b2_ml_best.pt \
    --split cluster_test \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== B2 Price-149 공정 평가 (CLEAN 기준) ===" | tee -a "$LOG"

python evaluate.py \
    --model b2_esm2_hier \
    --checkpoint outputs/checkpoints/b2_ml_best.pt \
    --split price149 \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "=== B2 완료: $(date) ===" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== B3 멀티레이블 훈련 자동 시작 ===" | tee -a "$LOG"
bash /home/user/Desktop/unlv/scripts/train_b3_multilabel.sh
