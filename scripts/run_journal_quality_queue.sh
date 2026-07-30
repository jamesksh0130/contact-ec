#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHON="${PYTHON:-/home/user/anaconda3/bin/python}"

echo "=== Journal-quality experiment queue started: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

./scripts/run_temporal_known_seed_repeats.sh
./scripts/run_simple_fusion_baseline_seed_repeats.sh

echo "=== Journal-quality experiment queue complete: $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
