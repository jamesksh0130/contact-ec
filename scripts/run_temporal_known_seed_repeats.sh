#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/user/anaconda3/bin/python}"
CONFIG="${CONFIG:-config_ecbench.yaml}"
SEEDS="${SEEDS:-42 43 44}"
GPU_LIST="${GPU_LIST:-0 1}"
MIN_FREE_MB="${MIN_FREE_MB:-42000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
EPOCHS="${EPOCHS:-30}"
TRAIN_BATCH="${TRAIN_BATCH:-64}"
EVAL_BATCH="${EVAL_BATCH:-128}"

IDS_FILE="${IDS_FILE:-data/ecbench/splits/test_ids_full.txt}"
META_CSV="${META_CSV:-data/ecbench/processed/test_meta_full.csv}"
OUT_DIR="outputs/results/temporal_known_seed_repeats"
LOG_DIR="outputs/logs/temporal_known_seed_repeats"
mkdir -p "$OUT_DIR" "$LOG_DIR"

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

train_and_eval() {
  local model="$1"
  local tag_prefix="$2"
  local seed="$3"
  local gpu="$4"
  local tag="${tag_prefix}_seed${seed}"
  local ckpt="outputs/checkpoints/${tag}_best.pt"
  local train_log="${LOG_DIR}/${tag}_train.log"
  local eval_log="${LOG_DIR}/${tag}_eval.log"
  local generic_json="outputs/results/${model}_test_hier_results.json"
  local final_json="${OUT_DIR}/${tag}_known124_hier_results.json"

  if [[ -f "$final_json" ]]; then
    echo "[$(timestamp)] Skip existing result: $final_json"
    return 0
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "[$(timestamp)] Train ${model} seed=${seed} gpu=${gpu}"
    "$PYTHON" train.py \
      --config "$CONFIG" \
      --model "$model" \
      --phase 1 \
      --epochs "$EPOCHS" \
      --gpu "$gpu" \
      --tag "$tag" \
      --batch_size "$TRAIN_BATCH" \
      --seed "$seed" \
      > "$train_log" 2>&1
  else
    echo "[$(timestamp)] Reuse checkpoint: $ckpt"
  fi

  echo "[$(timestamp)] Evaluate ${model} seed=${seed} gpu=${gpu}"
  "$PYTHON" evaluate.py \
    --config "$CONFIG" \
    --model "$model" \
    --checkpoint "$ckpt" \
    --split test \
    --ids_file "$IDS_FILE" \
    --meta_csv "$META_CSV" \
    --hierarchical_inference \
    --gpu "$gpu" \
    --batch_size "$EVAL_BATCH" \
    > "$eval_log" 2>&1

  cp "$generic_json" "$final_json"
  echo "[$(timestamp)] Saved $final_json"
}

echo "=== Temporal known-124 seed-repeat run started: $(timestamp) ==="
echo "CONFIG=$CONFIG SEEDS=$SEEDS GPU_LIST=$GPU_LIST MIN_FREE_MB=$MIN_FREE_MB"
echo "IDS_FILE=$IDS_FILE"
echo "META_CSV=$META_CSV"

for seed in $SEEDS; do
  gpu="$(wait_for_gpu)"
  train_and_eval "b1_esm2_fc" "temporal_known_b1_esm2_fc" "$seed" "$gpu"

  gpu="$(wait_for_gpu)"
  train_and_eval "b3_contact" "temporal_known_b3_contact" "$seed" "$gpu"

  gpu="$(wait_for_gpu)"
  train_and_eval "fusion_v2_flatfc" "temporal_known_fusion_v2_flatfc" "$seed" "$gpu"

  "$PYTHON" scripts/collect_temporal_known_seed_repeats.py
done

"$PYTHON" scripts/collect_temporal_known_seed_repeats.py
echo "=== Temporal known-124 seed-repeat run complete: $(timestamp) ==="
