#!/bin/bash
# Train one-seed B1/B3/Fusion models on stricter/looser Foldseek splits.
set -euo pipefail

cd /home/user/Desktop/unlv

PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/bin/python}"
CONFIG="${CONFIG:-config_foldseek.yaml}"
GPU="${GPU:-1}"
EPOCHS="${EPOCHS:-30}"
MAX_USED_MB="${MAX_USED_MB:-12000}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}"
LOG_DIR="outputs/results"
mkdir -p "$LOG_DIR"

wait_for_split() {
    local PREFIX="$1"
    while true; do
        if [ -s "data/ecbench/splits/${PREFIX}_train_ids.txt" ] \
            && [ -s "data/ecbench/splits/${PREFIX}_val_ids.txt" ] \
            && [ -s "data/ecbench/splits/${PREFIX}_test_ids.txt" ]; then
            break
        fi
        echo "Split ${PREFIX} is not ready; waiting ${CHECK_INTERVAL_SEC}s..."
        sleep "$CHECK_INTERVAL_SEC"
    done
}

wait_for_gpu() {
    while true; do
        USED_MB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
        echo "GPU ${GPU} used memory: ${USED_MB} MiB"
        if [ "$USED_MB" -lt "$MAX_USED_MB" ]; then
            break
        fi
        echo "GPU ${GPU} is busy; waiting ${CHECK_INTERVAL_SEC}s..."
        sleep "$CHECK_INTERVAL_SEC"
    done
}

train_eval_model() {
    local PREFIX="$1"
    local MODEL="$2"
    local TAG="$3"
    local BATCH="$4"
    local LOG="${LOG_DIR}/${TAG}.log"

    wait_for_gpu
    echo "=== Training ${MODEL} on ${PREFIX}: $(date) ===" | tee "$LOG"
    "$PYTHON_BIN" train.py \
        --config "$CONFIG" \
        --model "$MODEL" \
        --phase 1 \
        --epochs "$EPOCHS" \
        --gpu "$GPU" \
        --batch_size "$BATCH" \
        --split_prefix "${PREFIX}_" \
        --tag "$TAG" \
        --seed 42 \
        2>&1 | tee -a "$LOG"

    "$PYTHON_BIN" evaluate.py \
        --config "$CONFIG" \
        --model "$MODEL" \
        --checkpoint "outputs/checkpoints/${TAG}_best.pt" \
        --split "${PREFIX}_test" \
        --hierarchical_inference \
        --gpu "$GPU" \
        2>&1 | tee -a "$LOG"
}

for THR in 40 60; do
    PREFIX="foldseek_tmscore${THR}_cc"
    wait_for_split "$PREFIX"
    train_eval_model "$PREFIX" b1_esm2_fc "${PREFIX}_b1_ml" 512
    train_eval_model "$PREFIX" b3_contact "${PREFIX}_b3_ml" 256
    train_eval_model "$PREFIX" fusion "${PREFIX}_fusion_ml" 128
done

echo "=== Foldseek TM-score sweep training complete: $(date) ==="
