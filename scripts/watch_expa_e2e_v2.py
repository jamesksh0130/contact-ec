"""
Experiment A-E2E v2 자동 파이프라인 (Partial Cache 버전)

watch_expa_e2e.py 대체:
  1. Phase 1 완료 대기
  2. Layer 26 캐시 추출 완료 대기 (병렬로 먼저 시작해도 됨)
  3. Phase 2 캐시 학습 시작 (train_phase2_cached.py)
  4. Temporal test 평가 → expa_e2e_eval.json 저장
  5. finalize_e2e_figure.py 실행 (논문 자동 업데이트)

사용법:
  python scripts/watch_expa_e2e_v2.py &
  (watch_expa_e2e.py를 먼저 kill한 후 실행)
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT      = Path("/home/user/Desktop/unlv")
LOG_P1    = ROOT / "outputs/logs/expa_e2e_phase1.log"
CKPT_P1   = ROOT / "outputs/checkpoints/expa_e2e_phase1_best.pt"
CKPT_P2   = ROOT / "outputs/checkpoints/expa_e2e_phase2_cached_best.pt"
OUT_JSON  = ROOT / "outputs/results/expa_e2e_eval.json"
CACHE_DONE= ROOT / "data/processed/esm2_layer26/DONE"
P2_TAG    = "expa_e2e_phase2_cached"


def wait_for_file(path: Path, poll: int = 60, msg: str = ""):
    label = msg or str(path.name)
    print(f"대기: {label}", flush=True)
    while not path.exists():
        time.sleep(poll)
    print(f"감지: {label}", flush=True)


def wait_for_log(log_path: Path, pattern: str, poll: int = 60):
    print(f"로그 대기: {pattern!r} in {log_path.name}", flush=True)
    while True:
        if log_path.exists():
            if re.search(pattern, log_path.read_text(errors="replace")):
                print(f"감지 완료: {pattern!r}", flush=True)
                return
        time.sleep(poll)


# ── Step 0: Phase 1 완료 대기 ([10/10] 로그 패턴) ──────────────────────────
# 주의: 체크포인트 파일은 epoch 1 이후 바로 생성되므로 파일 존재 여부로 판단 불가.
# 반드시 로그에서 [10/10] 패턴을 확인해야 함.
if LOG_P1.exists() and re.search(r'\[0*10/10\]', LOG_P1.read_text(errors="replace")):
    print("Phase 1 이미 완료됨 (로그 확인).", flush=True)
else:
    wait_for_log(LOG_P1, r'\[0*10/10\]')
    print("Phase 1 완료 감지 (로그).", flush=True)

# 체크포인트 파일이 디스크에 flush될 때까지 잠시 대기
for _ in range(10):
    if CKPT_P1.exists():
        break
    time.sleep(6)

if not CKPT_P1.exists():
    print(f"ERROR: Phase 1 체크포인트를 찾을 수 없음: {CKPT_P1}", flush=True)
    sys.exit(1)

print(f"Phase 1 체크포인트 확인: {CKPT_P1.name}", flush=True)

# ── Step 1: Layer 26 캐시 추출 완료 대기 ─────────────────────────────────────
if CACHE_DONE.exists():
    print("Layer 26 캐시 이미 완료.", flush=True)
else:
    print("Layer 26 캐시 추출 완료 대기 중 (extract_layer26_cache.py 실행 필요)...", flush=True)
    wait_for_file(CACHE_DONE, poll=120,
                  msg="data/processed/esm2_layer26/DONE")

# ── Step 2: Phase 2 캐시 학습 시작 ───────────────────────────────────────────
print("\n=== Phase 2 시작 (ESM-2 end-to-end, Layer26 캐시) ===", flush=True)
p2 = subprocess.Popen(
    ["python", "scripts/train_phase2_cached.py",
     "--config",     "config_expa_e2e.yaml",
     "--resume",     str(CKPT_P1),
     "--epochs",     "10",
     "--batch_size", "16",
     "--grad_accum", "2",
     "--lr_esm",     "1e-5",
     "--lr_rest",    "1e-4",
     "--fp16",
     "--tag",        P2_TAG,
     "--gpu",        "0"],
    stdout=open(ROOT / f"outputs/logs/{P2_TAG}.log", "w"),
    stderr=subprocess.STDOUT,
    cwd=str(ROOT),
)
print(f"Phase 2 PID: {p2.pid}", flush=True)
p2.wait()
ret = p2.returncode
print(f"Phase 2 종료. 리턴코드: {ret}", flush=True)

# ── Step 3: Temporal test 평가 ───────────────────────────────────────────────
best = CKPT_P2 if CKPT_P2.exists() else CKPT_P1
print(f"\n=== Temporal test 평가: {best.name} ===", flush=True)

r = subprocess.run(
    ["python", "evaluate.py",
     "--config",     "config_expa_e2e.yaml",
     "--model",      "fusion_esm_ft",       # 평가는 원본 모델로 (forward가 같음)
     "--checkpoint", str(best),
     "--ids_file",   "data/ecbench/splits/test_ids_full.txt",
     "--meta_csv",   "data/ecbench/processed/test_meta_full_expa_enc.csv"],
    capture_output=True, text=True, cwd=str(ROOT),
)
print(r.stdout, flush=True)
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:], flush=True)

# ── 결과 파싱 & JSON 저장 ─────────────────────────────────────────────────────
micro = rare = None
for line in r.stdout.splitlines():
    m = re.search(r'micro[_\s]f1\s*[=:]\s*([0-9.]+)', line, re.I)
    if m: micro = float(m.group(1))
    m = re.search(r'rare[_\-]?ec[_\s]f1\s*[=:]\s*([0-9.]+)', line, re.I)
    if m: rare = float(m.group(1))

OUT_JSON.write_text(json.dumps({
    "experiment": "Experiment A-E2E v2 (Layer26 Partial Cache + E2E ESM-2 fine-tuning)",
    "checkpoint": str(best),
    "micro_f1": micro,
    "rare_ec_f1": rare,
    "method": "partial_cache_layer26",
}, indent=2))

print("\n" + "=" * 60, flush=True)
print("EXPERIMENT A-E2E (v2, CACHED) RESULTS", flush=True)
print("=" * 60, flush=True)
print(f"  Micro F1 (temporal) : {micro}", flush=True)
print(f"  Rare-EC F1          : {rare}", flush=True)
print(f"비교:", flush=True)
print(f"  Contact-EC (2018)   : 0.6032", flush=True)
print(f"  Contact-EC-ExpA     : 0.7209", flush=True)
print(f"  Contact-EC-E2E-v2   : {micro}", flush=True)
print(f"  HIT-EC              : 0.8471", flush=True)

# ── Step 4: 논문 자동 업데이트 ───────────────────────────────────────────────
print("\n=== 논문 자동 업데이트 ===", flush=True)
r2 = subprocess.run(
    ["python", "scripts/finalize_e2e_figure.py"],
    capture_output=True, text=True, cwd=str(ROOT),
)
print(r2.stdout, flush=True)
if r2.returncode != 0:
    print("finalize 오류:", r2.stderr[-1000:], flush=True)
else:
    print("논문 업데이트 완료.", flush=True)
