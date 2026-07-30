#!/bin/bash
# Build additional Foldseek/TM-score structure-disjoint splits for robustness.
set -euo pipefail

cd /home/user/Desktop/unlv

PYTHON_BIN="${PYTHON_BIN:-/home/user/anaconda3/bin/python}"
FOLDSEEK_BIN="${FOLDSEEK_BIN:-/home/user/anaconda3/bin/foldseek}"
THREADS="${THREADS:-16}"

for THR in 40 60; do
    SCORE="0.${THR}"
    PREFIX="foldseek_tmscore${THR}_cc"
    TMP_DIR="tmp_${PREFIX}"

    echo "=== Building ${PREFIX}: $(date) ==="
    "$PYTHON_BIN" scripts/foldseek_structure_split.py \
        --meta data/ecbench/processed/train_meta.csv \
        --pdb-dir data/raw/pdb \
        --split-dir data/ecbench/splits \
        --out-dir outputs/audit \
        --tmp-dir "$TMP_DIR" \
        --prefix "$PREFIX" \
        --seed 42 \
        --val-ratio 0.10 \
        --test-ratio 0.10 \
        --assignment-mode protein_balanced \
        --min-seq-id 0.30 \
        --coverage 0.80 \
        --tmscore-threshold "$SCORE" \
        --lddt-threshold 0.00 \
        --alignment-type 1 \
        --cluster-mode 1 \
        --sensitivity 7.5 \
        --threads "$THREADS" \
        --foldseek-bin "$FOLDSEEK_BIN"
done

echo "=== Foldseek TM-score sweep split build complete: $(date) ==="
