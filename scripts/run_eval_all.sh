#!/bin/bash
# Fusion 완료 후 전체 모델 자동 평가 + B2 재학습 스크립트
# 사용법: bash scripts/run_eval_all.sh
# 또는:  bash scripts/run_eval_all.sh --skip_b2_retrain

set -e
ROOT="/home/user/Desktop/unlv"
LOG="$ROOT/outputs/results/eval_all.log"
GPU=1

cd "$ROOT"
mkdir -p outputs/results

echo "=======================================" | tee -a "$LOG"
echo "전체 평가 시작: $(date)" | tee -a "$LOG"
echo "=======================================" | tee -a "$LOG"

# ── 평가할 모델 목록 ──────────────────────────────────────
declare -A MODELS=(
    ["b0_cnn"]="outputs/checkpoints/b0_cnn_phase1_best.pt"
    ["b1_esm2_fc"]="outputs/checkpoints/b1_esm2_fc_phase1_best.pt"
    ["b2_esm2_hier"]="outputs/checkpoints/b2_esm2_hier_phase1_best.pt"
    ["b3_contact"]="outputs/checkpoints/b3_contact_phase1_best.pt"
    ["fusion"]="outputs/checkpoints/fusion_phase2_best.pt"
)

for model in b0_cnn b1_esm2_fc b3_contact fusion; do
    ckpt="${MODELS[$model]}"
    if [ ! -f "$ckpt" ]; then
        echo "[SKIP] $model — 체크포인트 없음: $ckpt" | tee -a "$LOG"
        continue
    fi
    echo "" | tee -a "$LOG"
    echo "--- [$model] TEST 평가 ---" | tee -a "$LOG"
    python evaluate.py \
        --model "$model" \
        --checkpoint "$ckpt" \
        --split test \
        --gpu "$GPU" 2>&1 | tee -a "$LOG"
done

# ── 결과 요약 출력 ────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "=======================================" | tee -a "$LOG"
echo "결과 JSON 파일 목록:" | tee -a "$LOG"
ls -la "$ROOT/outputs/results/"*_test_results.json 2>/dev/null | tee -a "$LOG"

# ── B2 재학습 (수정된 코드로) ─────────────────────────────
if [[ "$1" != "--skip_b2_retrain" ]]; then
    echo "" | tee -a "$LOG"
    echo "=======================================" | tee -a "$LOG"
    echo "B2 재학습 시작 (detach 버그 수정 버전): $(date)" | tee -a "$LOG"
    echo "=======================================" | tee -a "$LOG"
    python train.py \
        --model b2_esm2_hier \
        --phase 1 \
        --gpu "$GPU" \
        --epochs 30 2>&1 | tee -a /tmp/pipeline.log
    echo "B2 재학습 완료: $(date)" | tee -a "$LOG"

    echo "" | tee -a "$LOG"
    echo "--- [b2_esm2_hier] TEST 평가 (재학습 버전) ---" | tee -a "$LOG"
    python evaluate.py \
        --model b2_esm2_hier \
        --checkpoint outputs/checkpoints/b2_esm2_hier_phase1_best.pt \
        --split test \
        --gpu "$GPU" 2>&1 | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "모든 작업 완료: $(date)" | tee -a "$LOG"
