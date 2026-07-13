#!/bin/bash
# FusionV2 멀티레이블 재훈련 (L1~L4 모두 BCE, 멀티레이블 F1 기준)
set -e
cd /home/user/Desktop/unlv

LOG="outputs/results/fusion_v2_multilabel.log"
echo "=== FusionV2 멀티레이블 훈련 시작: $(date) ===" | tee "$LOG"

python train.py \
    --model fusion_v2 \
    --phase 1 \
    --epochs 30 \
    --gpu 1 \
    --batch_size 96 \
    --tag fusion_v2_ml \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== 테스트 평가 ===" | tee -a "$LOG"

python evaluate.py \
    --model fusion_v2 \
    --checkpoint outputs/checkpoints/fusion_v2_ml_best.pt \
    --split test \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== V2 클러스터 테스트 평가 (유사도 <50% split) ===" | tee -a "$LOG"
python evaluate.py \
    --model fusion_v2 \
    --checkpoint outputs/checkpoints/fusion_v2_ml_best.pt \
    --split cluster_test \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== V2 Price-149 공정 평가 (CLEAN 기준) ===" | tee -a "$LOG"
python evaluate.py \
    --model fusion_v2 \
    --checkpoint outputs/checkpoints/fusion_v2_ml_best.pt \
    --split price149 \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "=== V2 완료: $(date) ===" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== V3 멀티레이블 훈련 자동 시작 ===" | tee -a "$LOG"
bash /home/user/Desktop/unlv/scripts/train_v3_multilabel.sh
