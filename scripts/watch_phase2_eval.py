"""Phase 2 완료 → 평가 → JSON 저장 → 논문 업데이트 자동 실행."""
import json, re, subprocess, sys, time
from pathlib import Path

ROOT    = Path("/home/user/Desktop/unlv")
LOG_P2  = ROOT / "outputs/logs/expa_e2e_phase2_cached.log"
CKPT_P2 = ROOT / "outputs/checkpoints/expa_e2e_phase2_cached_best.pt"
CKPT_P1 = ROOT / "outputs/checkpoints/expa_e2e_phase1_best.pt"
OUT_JSON= ROOT / "outputs/results/expa_e2e_eval.json"

print("Phase 2 완료 대기 중 ([010/10])...", flush=True)
while True:
    if LOG_P2.exists() and re.search(r'\[0*10/10\]', LOG_P2.read_text(errors="replace")):
        print("Phase 2 완료 감지!", flush=True)
        break
    time.sleep(60)

for _ in range(10):
    if CKPT_P2.exists(): break
    time.sleep(6)

best = CKPT_P2 if CKPT_P2.exists() else CKPT_P1
print(f"평가 체크포인트: {best.name}", flush=True)

r = subprocess.run(
    ["python", "evaluate.py",
     "--config",     "config_expa_e2e.yaml",
     "--model",      "fusion_esm_ft",
     "--checkpoint", str(best),
     "--ids_file",   "data/ecbench/splits/test_ids_full.txt",
     "--meta_csv",   "data/ecbench/processed/test_meta_full_expa_enc.csv"],
    capture_output=True, text=True, cwd=str(ROOT)
)
print(r.stdout, flush=True)
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:], flush=True)

micro = rare = None
for line in r.stdout.splitlines():
    m = re.search(r'micro[_\s]f1\s*[=:]\s*([0-9.]+)', line, re.I)
    if m: micro = float(m.group(1))
    m = re.search(r'rare[_\-]?ec[_\s]f1\s*[=:]\s*([0-9.]+)', line, re.I)
    if m: rare = float(m.group(1))

OUT_JSON.write_text(json.dumps({
    "experiment": "Experiment A-E2E (Layer26 cache + E2E fine-tuning)",
    "checkpoint": str(best),
    "micro_f1": micro, "rare_ec_f1": rare,
    "method": "partial_cache_layer26",
}, indent=2))
print(f"JSON 저장: micro_f1={micro}, rare_ec_f1={rare}", flush=True)

r2 = subprocess.run(["python", "scripts/finalize_e2e_figure.py"],
                    capture_output=True, text=True, cwd=str(ROOT))
print(r2.stdout, flush=True)
if r2.returncode != 0:
    print("finalize 오류:", r2.stderr[-1000:], flush=True)
print("완료!", flush=True)
