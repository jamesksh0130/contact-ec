#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="${ROOT}/outputs/baselines/homology"
THREADS="${THREADS:-16}"
SENSITIVITY="${SENSITIVITY:-7.5}"
SPLITS=("${@}")

if ! command -v mmseqs >/dev/null 2>&1; then
  echo "mmseqs binary not found in PATH" >&2
  exit 1
fi

if [ "${#SPLITS[@]}" -eq 0 ]; then
  SPLITS=(
    "foldseek_tmscore50_cc"
    "foldseek_tmscore40_cc"
    "foldseek_tmscore60_cc"
  )
fi

for split in "${SPLITS[@]}"; do
  train_fasta="${BASE_DIR}/${split}_train.fasta"
  test_fasta="${BASE_DIR}/${split}_test.fasta"
  out_tsv="${BASE_DIR}/${split}_mmseqs_top10.tsv"
  work_dir="${BASE_DIR}/mmseqs_${split}"
  tmp_dir="${work_dir}/tmp"
  train_db="${work_dir}/train_db"
  test_db="${work_dir}/test_db"
  result_db="${work_dir}/result_db"

  if [ ! -s "${train_fasta}" ] || [ ! -s "${test_fasta}" ]; then
    echo "Missing FASTA input for ${split}" >&2
    exit 1
  fi

  mkdir -p "${work_dir}" "${tmp_dir}"
  echo "[MMseqs2] ${split}: createdb"
  mmseqs createdb "${train_fasta}" "${train_db}" >/dev/null
  mmseqs createdb "${test_fasta}" "${test_db}" >/dev/null

  echo "[MMseqs2] ${split}: search"
  mmseqs search "${test_db}" "${train_db}" "${result_db}" "${tmp_dir}" \
    --threads "${THREADS}" -s "${SENSITIVITY}" >/dev/null

  echo "[MMseqs2] ${split}: convertalis"
  mmseqs convertalis "${test_db}" "${train_db}" "${result_db}" "${out_tsv}" \
    --format-output "query,target,pident,alnlen,evalue,bits" >/dev/null

  echo "[MMseqs2] ${split}: wrote ${out_tsv}"
done

python "${ROOT}/scripts/evaluate_mmseqs_homology_baseline.py" \
  --base-dir "${BASE_DIR}" \
  --splits "${SPLITS[@]}"
