#!/bin/bash
# V2 30ep 완료 후 자동 실행: 테스트 평가 + 앙상블 재평가
set -e
cd /home/user/Desktop/unlv

LOG="outputs/results/v2_30ep_post_eval.log"
echo "=== V2 30ep 완료 후 평가 시작: $(date) ===" | tee "$LOG"

# ── 1. V2 30ep 단독 테스트 평가 ──────────────────────────────
echo "" | tee -a "$LOG"
echo "=== [1/3] FusionV2 30ep 테스트 평가 ===" | tee -a "$LOG"
python evaluate.py \
    --model fusion_v2 \
    --checkpoint outputs/checkpoints/fusion_v2_phase1_best.pt \
    --split test \
    --gpu 0 \
    2>&1 | tee -a "$LOG"

# ── 2. 앙상블 그리드 서치 (val set, 가중치 탐색) ──────────────
echo "" | tee -a "$LOG"
echo "=== [2/3] 앙상블 가중치 탐색 (Val set) ===" | tee -a "$LOG"
for w2 in 0.7 0.6 0.5 0.4 0.3; do
    w3=$(python3 -c "print(round(1.0 - $w2, 1))")
    echo "--- w2=$w2, w3=$w3 ---" | tee -a "$LOG"
    python scripts/ensemble_evaluate.py \
        --v2_ckpt outputs/checkpoints/fusion_v2_phase1_best.pt \
        --v3_ckpt outputs/checkpoints/fusion_v3_phase1_best.pt \
        --split val \
        --w2 "$w2" --w3 "$w3" \
        --hierarchical_inference \
        --gpu 0 \
        2>&1 | grep -E "level4|Micro F1|앙상블" | tee -a "$LOG"
done

# ── 3. 최적 가중치로 테스트 평가 ──────────────────────────────
echo "" | tee -a "$LOG"
echo "=== [3/3] 앙상블 V2(30ep)+V3(30ep) 테스트 평가 (w2=0.6, w3=0.4) ===" | tee -a "$LOG"
python scripts/ensemble_evaluate.py \
    --v2_ckpt outputs/checkpoints/fusion_v2_phase1_best.pt \
    --v3_ckpt outputs/checkpoints/fusion_v3_phase1_best.pt \
    --split test \
    --w2 0.6 --w3 0.4 \
    --hierarchical_inference \
    --gpu 0 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== 모든 평가 완료: $(date) ===" | tee -a "$LOG"
