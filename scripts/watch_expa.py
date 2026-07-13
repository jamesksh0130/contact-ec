"""
Experiment A 자동 파이프라인:
  1. Phase 1 완료(epoch 30/30) 감지 → Phase 2 자동 시작
  2. Phase 2 완료(epoch 20/20) 감지 → temporal test 자동 평가
  3. 평가 완료 → 결과 출력
"""
import time, subprocess, re, json
from pathlib import Path

ROOT     = Path("/home/user/Desktop/unlv")
LOG_P1   = ROOT / "outputs/logs/expa_flatfc_phase1.log"
LOG_P2   = ROOT / "outputs/logs/expa_flatfc_phase2.log"
CKPT_P1  = ROOT / "outputs/checkpoints/expa_flatfc_phase1_best.pt"
CKPT_P2  = ROOT / "outputs/checkpoints/expa_flatfc_phase2_best.pt"
OUT_JSON = ROOT / "outputs/results/expa_flatfc_eval.json"

def wait_for_pattern(log_path, pattern, poll=30):
    print(f"대기 중: {pattern} in {log_path.name}")
    while True:
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            if re.search(pattern, text):
                print(f"감지: {pattern}")
                return True
        time.sleep(poll)

# ── 1. Phase 1 완료 대기 ─────────────────────────────────────
wait_for_pattern(LOG_P1, r'\[0*30/30\]')

# ── 2. Phase 2 시작 ──────────────────────────────────────────
print("\n=== Phase 2 시작 ===")
p2_cmd = [
    "python", "train.py",
    "--config",  "config_expa.yaml",
    "--model",   "fusion_v2_flatfc",
    "--phase",   "2",
    "--epochs",  "20",
    "--tag",     "expa_flatfc_phase2",
    "--resume",  str(CKPT_P1),
]
with open(LOG_P2, "w") as f:
    proc = subprocess.Popen(p2_cmd, stdout=f, stderr=subprocess.STDOUT,
                            cwd=str(ROOT))
print(f"Phase 2 PID: {proc.pid}")
proc.wait()
print("Phase 2 완료")

# ── 3. temporal test 평가 ────────────────────────────────────
best_ckpt = CKPT_P2 if CKPT_P2.exists() else CKPT_P1
print(f"\n=== Temporal test 평가: {best_ckpt.name} ===")
eval_cmd = [
    "python", "evaluate.py",
    "--config",     "config_expa.yaml",
    "--model",      "fusion_v2_flatfc",
    "--checkpoint", str(best_ckpt),
    "--ids_file",   str(ROOT / "data/ecbench/splits/test_ids_full.txt"),
    "--meta_csv",   str(ROOT / "data/ecbench/processed/test_meta_full.csv"),
]
result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=str(ROOT))
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])

# ── 4. 결과 파싱 및 저장 ─────────────────────────────────────
micro = weighted = macro = rare = None
for line in result.stdout.splitlines():
    m = re.search(r'micro[_\s]f1\s*[=:]\s*([0-9.]+)', line, re.I)
    if m: micro = float(m.group(1))
    m = re.search(r'weighted[_\s]f1\s*[=:]\s*([0-9.]+)', line, re.I)
    if m: weighted = float(m.group(1))
    m = re.search(r'macro[_\s]f1\s*[=:]\s*([0-9.]+)', line, re.I)
    if m: macro = float(m.group(1))
    m = re.search(r'rare[_\-]?ec[_\s]f1\s*[=:]\s*([0-9.]+)', line, re.I)
    if m: rare = float(m.group(1))

summary = {
    "experiment": "Experiment A (full 270K data, equivalent to HIT-EC)",
    "checkpoint": str(best_ckpt),
    "train_proteins": 243198,
    "test_set": "EC-Bench temporal (N=124)",
    "micro_f1":    micro,
    "weighted_f1": weighted,
    "macro_f1":    macro,
    "rare_ec_f1":  rare,
}
OUT_JSON.write_text(json.dumps(summary, indent=2))

print("\n" + "="*60)
print("EXPERIMENT A RESULTS")
print("="*60)
print(f"  Train proteins  : 243,198 (full Swiss-Prot, equivalent to HIT-EC)")
print(f"  Micro F1        : {micro}")
print(f"  Weighted F1     : {weighted}")
print(f"  Rare-EC F1      : {rare}")
print()
print("비교:")
print(f"  Contact-EC (EC-Bench 2018) : 0.6032")
print(f"  HIT-EC (Swiss-Prot 2022)   : 0.8471")
print(f"  Contact-EC Exp A (full)    : {micro}")
print(f"결과 저장: {OUT_JSON}")
