#!/bin/bash
# B1 (ESM-2 + Flat FC) 멀티레이블 재훈련
# contact map 없음 → 빠름 (배치 크기 크게)
set -e
cd /home/user/Desktop/unlv

LOG="outputs/results/b1_multilabel.log"
echo "=== B1 멀티레이블 훈련 시작: $(date) ===" | tee "$LOG"

python train.py \
    --model b1_esm2_fc \
    --phase 1 \
    --epochs 30 \
    --gpu 1 \
    --batch_size 512 \
    --tag b1_ml \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== B1 테스트 평가 ===" | tee -a "$LOG"

python evaluate.py \
    --model b1_esm2_fc \
    --checkpoint outputs/checkpoints/b1_ml_best.pt \
    --split test \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== B1 클러스터 테스트 평가 (유사도 <50% split) ===" | tee -a "$LOG"

python evaluate.py \
    --model b1_esm2_fc \
    --checkpoint outputs/checkpoints/b1_ml_best.pt \
    --split cluster_test \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== B1 Price-149 공정 평가 (CLEAN 기준) ===" | tee -a "$LOG"

python evaluate.py \
    --model b1_esm2_fc \
    --checkpoint outputs/checkpoints/b1_ml_best.pt \
    --split price149 \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "=== B1 완료: $(date) ===" | tee -a "$LOG"
