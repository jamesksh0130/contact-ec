#!/bin/bash
# B3 (Contact Map + ResNet + FC) 멀티레이블 재훈련
# 구조 정보만 사용 (ESM-2 없음) → ablation용
set -e
cd /home/user/Desktop/unlv

LOG="outputs/results/b3_multilabel.log"
echo "=== B3 멀티레이블 훈련 시작: $(date) ===" | tee "$LOG"

python train.py \
    --model b3_contact \
    --phase 1 \
    --epochs 30 \
    --gpu 1 \
    --batch_size 256 \
    --tag b3_ml \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== B3 테스트 평가 ===" | tee -a "$LOG"

python evaluate.py \
    --model b3_contact \
    --checkpoint outputs/checkpoints/b3_ml_best.pt \
    --split test \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "=== B3 완료: $(date) ===" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== B1 멀티레이블 훈련 자동 시작 ===" | tee -a "$LOG"
bash /home/user/Desktop/unlv/scripts/train_b1_multilabel.sh
