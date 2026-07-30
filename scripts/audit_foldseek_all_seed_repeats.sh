#!/bin/bash
# Aggregate repeated Foldseek/TM-score-disjoint runs for B1, B3, and fusion.
set -euo pipefail

cd /home/user/Desktop/unlv

GPU="${1:-1}"
PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/bin/python}"

"$PYTHON_BIN" scripts/audit_foldseek_fusion_seed_repeats.py \
    --model b1_esm2_fc \
    --model-label B1_ESM2_only \
    --batch-size 512 \
    --checkpoint-pattern "foldseek_tmscore50_b1_ml*_best.pt" \
    --val-threshold 0.13 \
    --output-prefix foldseek_tmscore50_cc_b1_seed_repeats \
    --gpu "$GPU"

"$PYTHON_BIN" scripts/audit_foldseek_fusion_seed_repeats.py \
    --model b3_contact \
    --model-label B3_contact_only \
    --batch-size 256 \
    --checkpoint-pattern "foldseek_tmscore50_b3_ml*_best.pt" \
    --val-threshold 0.07 \
    --output-prefix foldseek_tmscore50_cc_b3_seed_repeats \
    --gpu "$GPU"

"$PYTHON_BIN" scripts/audit_foldseek_fusion_seed_repeats.py \
    --model fusion \
    --model-label Fusion_ESM2_contact \
    --batch-size 128 \
    --checkpoint-pattern "foldseek_tmscore50_fusion_ml*_best.pt" \
    --val-threshold 0.08 \
    --output-prefix foldseek_tmscore50_cc_fusion_seed_repeats \
    --gpu "$GPU"
