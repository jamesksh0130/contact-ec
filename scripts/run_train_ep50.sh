#!/bin/bash
# 전체 모델 50에폭 재학습
# GPU 0: B0 → B1 → B2 → B3 (순서대로)
# GPU 1: Fusion GCA Phase1(50ep) → Phase2(20ep) (자동 연결)
set -e
ROOT="/home/user/Desktop/unlv"
LOG="$ROOT/outputs/results/train_ep50.log"
cd "$ROOT"
mkdir -p outputs/results

echo "=======================================" | tee -a "$LOG"
echo "50에폭 전체 재학습 시작: $(date)" | tee -a "$LOG"
echo "GPU 0: B0→B1→B2→B3  |  GPU 1: Fusion GCA" | tee -a "$LOG"
echo "=======================================" | tee -a "$LOG"

# ── GPU 1: Fusion GCA (가장 오래 걸림, 먼저 백그라운드 시작) ──
echo "" | tee -a "$LOG"
echo "--- Fusion GCA Phase 1 (50ep, GPU 1) ---" | tee -a "$LOG"
nohup bash -c "
  python train.py --model fusion --phase 1 --gpu 1 --epochs 50 >> $LOG 2>&1 && \
  echo '--- Fusion GCA Phase 2 (20ep, GPU 1) ---' >> $LOG && \
  python train.py --model fusion --phase 2 --gpu 1 --epochs 20 \
    --resume $ROOT/outputs/checkpoints/fusion_phase1_best.pt >> $LOG 2>&1 && \
  echo 'Fusion 전체 완료: '$(date) >> $LOG
" &
FUSION_PID=$!
echo "Fusion PID: $FUSION_PID" | tee -a "$LOG"

# ── GPU 0: B0 → B1 → B2 → B3 순서 ──
for model in b0_cnn b1_esm2_fc b2_esm2_hier b3_contact; do
    echo "" | tee -a "$LOG"
    echo "--- [$model] Phase 1 (50ep, GPU 0) ---" | tee -a "$LOG"
    echo "시작: $(date)" | tee -a "$LOG"
    python train.py --model "$model" --phase 1 --gpu 0 --epochs 50 2>&1 | tee -a "$LOG"
    echo "완료: $(date)" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "GPU 0 학습 완료: $(date)" | tee -a "$LOG"
echo "Fusion(GPU 1) 완료 대기 중..." | tee -a "$LOG"
wait $FUSION_PID
echo "모든 학습 완료: $(date)" | tee -a "$LOG"
