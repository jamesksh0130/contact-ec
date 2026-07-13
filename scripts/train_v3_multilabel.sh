#!/bin/bash
# FusionV3 멀티레이블 재훈련 (V2 완료 후 자동 실행)
set -e
cd /home/user/Desktop/unlv

LOG="outputs/results/fusion_v3_multilabel.log"
echo "=== FusionV3 멀티레이블 훈련 시작: $(date) ===" | tee "$LOG"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
    --model fusion_v3 \
    --phase 1 \
    --epochs 30 \
    --gpu 1 \
    --batch_size 64 \
    --tag fusion_v3_ml \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== V3 테스트 평가 ===" | tee -a "$LOG"

python evaluate.py \
    --model fusion_v3 \
    --checkpoint outputs/checkpoints/fusion_v3_ml_best.pt \
    --split test \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== V3 클러스터 테스트 평가 (유사도 <50% split) ===" | tee -a "$LOG"
python evaluate.py \
    --model fusion_v3 \
    --checkpoint outputs/checkpoints/fusion_v3_ml_best.pt \
    --split cluster_test \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== V3 Price-149 공정 평가 (CLEAN 기준) ===" | tee -a "$LOG"
python evaluate.py \
    --model fusion_v3 \
    --checkpoint outputs/checkpoints/fusion_v3_ml_best.pt \
    --split price149 \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "=== V3 완료: $(date) ===" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== B2 멀티레이블 훈련 자동 시작 ===" | tee -a "$LOG"
bash /home/user/Desktop/unlv/scripts/train_b2_multilabel.sh
