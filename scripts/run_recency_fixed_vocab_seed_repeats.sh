#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/user/anaconda3/bin/python}"
CONFIG="${CONFIG:-config_recency_fixed_vocab_sp2026.yaml}"
MODEL="${MODEL:-fusion_v2_flatfc}"
SEEDS="${SEEDS:-42 43 44}"
GPU_LIST="${GPU_LIST:-0 1}"
MIN_FREE_MB="${MIN_FREE_MB:-25000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-30}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-20}"
TRAIN_BATCH="${TRAIN_BATCH:-48}"
EVAL_BATCH="${EVAL_BATCH:-128}"

IDS_FULL="${IDS_FULL:-data/ecbench/splits/test_ids_full.txt}"
IDS_INTERSECTION="${IDS_INTERSECTION:-data/ecbench/splits/test_ids_recency_intersection.txt}"
META_OLD="${META_OLD:-data/ecbench/processed/test_meta_full.csv}"

LOG_DIR="${LOG_DIR:-outputs/logs/recency_fixed_vocab_seed_repeats}"
OUT_DIR="${OUT_DIR:-outputs/results/recency_fixed_vocab_seed_repeats}"
mkdir -p "$LOG_DIR" "$OUT_DIR" outputs/checkpoints

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

pick_gpu() {
  local gpu free
  for gpu in $GPU_LIST; do
    free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= MIN_FREE_MB )); then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

wait_for_gpu() {
  local gpu
  while true; do
    if gpu="$(pick_gpu)"; then
      echo "$gpu"
      return 0
    fi
    echo "[$(timestamp)] Waiting for GPU with >= ${MIN_FREE_MB} MiB free. Current:" >&2
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits >&2
    sleep "$SLEEP_SECONDS"
  done
}

run_seed() {
  local seed="$1"
  local gpu="$2"
  local phase1_tag="recency_fixed_vocab_sp2026_seed${seed}_phase1"
  local phase2_tag="recency_fixed_vocab_sp2026_seed${seed}_phase2"
  local phase1_ckpt="outputs/checkpoints/${phase1_tag}_best.pt"
  local phase2_ckpt="outputs/checkpoints/${phase2_tag}_best.pt"

  echo "[$(timestamp)] seed=${seed} gpu=${gpu} start"

  if [[ ! -f "$phase1_ckpt" ]]; then
    "$PYTHON" train.py \
      --config "$CONFIG" \
      --model "$MODEL" \
      --phase 1 \
      --epochs "$PHASE1_EPOCHS" \
      --gpu "$gpu" \
      --tag "$phase1_tag" \
      --batch_size "$TRAIN_BATCH" \
      --seed "$seed" \
      > "${LOG_DIR}/${phase1_tag}.log" 2>&1
  else
    echo "[$(timestamp)] Reuse $phase1_ckpt"
  fi

  if [[ ! -f "$phase2_ckpt" ]]; then
    "$PYTHON" train.py \
      --config "$CONFIG" \
      --model "$MODEL" \
      --phase 2 \
      --epochs "$PHASE2_EPOCHS" \
      --gpu "$gpu" \
      --tag "$phase2_tag" \
      --batch_size "$TRAIN_BATCH" \
      --seed "$seed" \
      --resume "$phase1_ckpt" \
      > "${LOG_DIR}/${phase2_tag}.log" 2>&1
  else
    echo "[$(timestamp)] Reuse $phase2_ckpt"
  fi

  "$PYTHON" evaluate.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --checkpoint "$phase2_ckpt" \
    --split "recency_fixed_vocab_full_seed${seed}" \
    --ids_file "$IDS_FULL" \
    --meta_csv "$META_OLD" \
    --gpu "$gpu" \
    --batch_size "$EVAL_BATCH" \
    --num_workers 0 \
    > "${LOG_DIR}/eval_full_seed${seed}.log" 2>&1
  cp "outputs/results/${MODEL}_recency_fixed_vocab_full_seed${seed}_results.json" \
     "${OUT_DIR}/seed${seed}_known124_results.json"

  "$PYTHON" evaluate.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --checkpoint "$phase2_ckpt" \
    --split "recency_fixed_vocab_intersection_seed${seed}" \
    --ids_file "$IDS_INTERSECTION" \
    --meta_csv "$META_OLD" \
    --gpu "$gpu" \
    --batch_size "$EVAL_BATCH" \
    --num_workers 0 \
    > "${LOG_DIR}/eval_intersection_seed${seed}.log" 2>&1
  cp "outputs/results/${MODEL}_recency_fixed_vocab_intersection_seed${seed}_results.json" \
     "${OUT_DIR}/seed${seed}_intersection99_results.json"

  "$PYTHON" scripts/collect_recency_fixed_vocab_eval.py
  echo "[$(timestamp)] seed=${seed} done"
}

echo "=== Fixed-vocabulary recency seed-repeat run started: $(timestamp) ==="
echo "CONFIG=$CONFIG MODEL=$MODEL SEEDS=$SEEDS GPU_LIST=$GPU_LIST MIN_FREE_MB=$MIN_FREE_MB TRAIN_BATCH=$TRAIN_BATCH"

for seed in $SEEDS; do
  gpu="$(wait_for_gpu)"
  run_seed "$seed" "$gpu" > "${LOG_DIR}/seed${seed}.master.log" 2>&1
done

"$PYTHON" scripts/collect_recency_fixed_vocab_eval.py
echo "=== Fixed-vocabulary recency seed-repeat run complete: $(timestamp) ==="
