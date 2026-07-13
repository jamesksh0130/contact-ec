"""
Experiment A-E2E 자동 파이프라인:
  Phase 1 완료 → Phase 2 자동 시작 (ESM-2 last 6 layers unfreeze)
  Phase 2 완료 → temporal test 자동 평가
"""
import time, subprocess, re, json
from pathlib import Path

ROOT    = Path("/home/user/Desktop/unlv")
LOG_P1  = ROOT / "outputs/logs/expa_e2e_phase1.log"
LOG_P2  = ROOT / "outputs/logs/expa_e2e_phase2.log"
CKPT_P1 = ROOT / "outputs/checkpoints/expa_e2e_phase1_best.pt"
CKPT_P2 = ROOT / "outputs/checkpoints/expa_e2e_phase2_best.pt"
OUT_JSON= ROOT / "outputs/results/expa_e2e_eval.json"

def wait_for(log_path, pattern, poll=60):
    print(f"대기: {pattern} in {log_path.name}")
    while True:
        if log_path.exists():
            if re.search(pattern, log_path.read_text(errors="replace")):
                print(f"감지 완료: {pattern}")
                return
        time.sleep(poll)

# ── 1. Phase 1 완료 대기 ─────────────────────────────────────
wait_for(LOG_P1, r'\[0*10/10\]')

# ── 2. Phase 2 시작 (ESM-2 last 6 layers unfreeze) ───────────
print("\n=== Phase 2 시작 (ESM-2 end-to-end fine-tuning) ===")
p2 = subprocess.Popen(
    ["python", "train.py",
     "--config",    "config_expa_e2e.yaml",
     "--model",     "fusion_esm_ft",
     "--phase",     "2",
     "--epochs",    "10",
     "--batch_size","8",
     "--grad_accum","4",
     "--lr_esm",    "1e-5",
     "--tag",       "expa_e2e_phase2",
     "--resume",    str(CKPT_P1)],
    stdout=open(LOG_P2, "w"), stderr=subprocess.STDOUT, cwd=str(ROOT)
)
print(f"Phase 2 PID: {p2.pid}")
p2.wait()
print("Phase 2 완료")

# ── 3. Temporal test 평가 ────────────────────────────────────
best = CKPT_P2 if CKPT_P2.exists() else CKPT_P1
print(f"\n=== Temporal test 평가: {best.name} ===")
r = subprocess.run(
    ["python", "evaluate.py",
     "--config",     "config_expa_e2e.yaml",
     "--model",      "fusion_esm_ft",
     "--checkpoint", str(best),
     "--ids_file",   "data/ecbench/splits/test_ids_full.txt",
     "--meta_csv",   "data/ecbench/processed/test_meta_full_expa_enc.csv"],
    capture_output=True, text=True, cwd=str(ROOT)
)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:])

# 결과 파싱
micro = weighted = macro = rare = None
for line in r.stdout.splitlines():
    m = re.search(r'micro[_\s]f1\s*[=:]\s*([0-9.]+)', line, re.I)
    if m: micro = float(m.group(1))
    m = re.search(r'rare[_\-]?ec[_\s]f1\s*[=:]\s*([0-9.]+)', line, re.I)
    if m: rare = float(m.group(1))

OUT_JSON.write_text(json.dumps({
    "experiment": "Experiment A-E2E (full data + end-to-end ESM-2 fine-tuning)",
    "checkpoint": str(best),
    "micro_f1": micro, "rare_ec_f1": rare,
}, indent=2))

print("\n" + "="*60)
print("EXPERIMENT A-E2E RESULTS")
print("="*60)
print(f"  Micro F1 (temporal, N=99) : {micro}")
print(f"  Rare-EC F1                : {rare}")
print()
print("비교:")
print(f"  Contact-EC (2018, frozen) : 0.6032")
print(f"  Contact-EC-ExpA (full)    : 0.7209")
print(f"  Contact-EC-E2E  (full+ft) : {micro}")
print(f"  HIT-EC                    : 0.8471")
