#!/bin/bash
# FusionV3 — 클러스터 split 기반 학습 (V2 완료 후 GPU 1에서 실행)
# 30% sequence identity 클러스터 분할 사용
set -e
cd /home/user/Desktop/unlv

LOG="outputs/results/fusion_v3_cluster_split.log"
echo "=== FusionV3 Cluster Split 학습 시작: $(date) ===" | tee "$LOG"

python train.py \
    --model fusion_v3 \
    --phase 1 \
    --epochs 30 \
    --gpu 1 \
    --split_prefix cluster_ \
    --tag fusion_v3_cluster_phase1 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== 클러스터 split 테스트 평가 ===" | tee -a "$LOG"

python evaluate.py \
    --model fusion_v3 \
    --checkpoint outputs/checkpoints/fusion_v3_cluster_phase1_best.pt \
    --split cluster_test \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "=== 완료: $(date) ===" | tee -a "$LOG"
