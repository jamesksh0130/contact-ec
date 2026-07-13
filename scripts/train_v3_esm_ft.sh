#!/bin/bash
# FusionV3 + ESM-2 last 2 layers unfreeze (Phase 2)
# V3 30ep 체크포인트에서 시작, ESM-2 마지막 2층 + 나머지 lr=1e-4
set -e
cd /home/user/Desktop/unlv

LOG="outputs/results/fusion_v3_esm_ft.log"
echo "=== FusionV3 ESM-2 Unfreeze (Phase 2) 시작: $(date) ===" | tee "$LOG"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train.py \
    --model fusion_v3_esm_ft \
    --phase 2 \
    --epochs 20 \
    --gpu 1 \
    --batch_size 8 \
    --resume outputs/checkpoints/fusion_v3_phase1_best.pt \
    --tag fusion_v3_esm_ft_phase2 \
    --lr_esm 1e-5 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== 테스트 평가 ===" | tee -a "$LOG"

python evaluate.py \
    --model fusion_v3_esm_ft \
    --checkpoint outputs/checkpoints/fusion_v3_esm_ft_phase2_best.pt \
    --split test \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"

echo "=== 완료: $(date) ===" | tee -a "$LOG"
