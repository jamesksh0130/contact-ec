#!/bin/bash
# Price-149 ESMFold 구조 예측 → contact map 생성
# V2 훈련 완료 후 GPU가 해제된 직후 실행
set -e
cd /home/user/Desktop/unlv

LOG="/tmp/esmfold_price149.log"
echo "=== ESMFold Price-149 시작: $(date) ===" | tee "$LOG"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/prep_price149.py --step fold --gpu 1 2>&1 | tee -a "$LOG"

echo "=== ESMFold 완료: $(date) ===" | tee -a "$LOG"

# ESMFold 완료 후 V2/V3로 Price-149 재평가 (contact map 있는 버전)
echo "" | tee -a "$LOG"
echo "=== V2 Price-149 재평가 (ESMFold contact map) ===" | tee -a "$LOG"

if [ -f outputs/checkpoints/fusion_v2_ml_best.pt ]; then
  python evaluate.py \
    --model fusion_v2 \
    --checkpoint outputs/checkpoints/fusion_v2_ml_best.pt \
    --split price149 \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "=== V3 Price-149 재평가 (ESMFold contact map) ===" | tee -a "$LOG"

if [ -f outputs/checkpoints/fusion_v3_ml_best.pt ]; then
  python evaluate.py \
    --model fusion_v3 \
    --checkpoint outputs/checkpoints/fusion_v3_ml_best.pt \
    --split price149 \
    --hierarchical_inference \
    --gpu 1 \
    2>&1 | tee -a "$LOG"
fi

echo "=== 전체 완료: $(date) ===" | tee -a "$LOG"
