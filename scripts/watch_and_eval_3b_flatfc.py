"""
3B flat FC 학습 완료(epoch 30/30) 감지 → EC-Bench temporal test 자동 평가
"""
import time, subprocess, json, re
from pathlib import Path

ROOT     = Path("/home/user/Desktop/unlv")
LOG_FILE = ROOT / "outputs/logs/ecbench_3b_flatfc.log"
CKPT     = ROOT / "outputs/checkpoints/ecbench_3b_flatfc_best.pt"
OUT_JSON = ROOT / "outputs/results/ecbench_3b_flatfc_eval.json"
POLL_SEC = 60

print("watcher 시작: 30/30 에포크 완료 대기 중...")

while True:
    if LOG_FILE.exists():
        text = LOG_FILE.read_text(errors="replace")
        # "[[030/30]" 또는 "[030/30]" 패턴 탐지
        if re.search(r'\[0*30/30\]', text):
            print("epoch 30 완료 감지! 평가 시작...")
            break
    time.sleep(POLL_SEC)

# ── 평가 실행 ─────────────────────────────────────────────────────
eval_cmd = [
    "python", "evaluate.py",
    "--config",     "config_ecbench_3b.yaml",
    "--model",      "fusion_v2_flatfc",
    "--checkpoint", str(CKPT),
    "--ids_file",   str(ROOT / "data/ecbench/splits/test_ids_full.txt"),
    "--meta_csv",   str(ROOT / "data/ecbench/processed/test_meta_full.csv"),
]

print("실행:", " ".join(eval_cmd))
result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=str(ROOT))

print("=== STDOUT ===")
print(result.stdout)
if result.returncode != 0:
    print("=== STDERR ===")
    print(result.stderr[-3000:])

# stdout에서 micro_f1 파싱
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
    "model": "Contact-EC-3B (flat FC)",
    "checkpoint": str(CKPT),
    "test_set": "EC-Bench temporal (N=124)",
    "micro_f1":    micro,
    "weighted_f1": weighted,
    "macro_f1":    macro,
    "rare_ec_f1":  rare,
    "stdout": result.stdout[-5000:],
}
OUT_JSON.write_text(json.dumps(summary, indent=2))
print(f"\n결과 저장: {OUT_JSON}")
print(f"\n{'='*50}")
print(f"Contact-EC-3B (flat FC)")
print(f"  Micro F1    : {micro}")
print(f"  Weighted F1 : {weighted}")
print(f"  Macro F1    : {macro}")
print(f"  Rare-EC F1  : {rare}")
print(f"{'='*50}")
print("\n비교:")
print("  Contact-EC (650M flat FC) : 0.6032")
print("  Contact-EC-Hier (650M)    : 0.5690")
print(f"  Contact-EC-3B  (flat FC)  : {micro}")

# ── 논문 자동 업데이트 ─────────────────────────────────────────────
if micro is not None:
    print("\n논문 업데이트 중...")
    paper_result = subprocess.run(
        ["python", "scripts/update_paper_3b_flatfc.py"],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    print(paper_result.stdout)
    if paper_result.returncode != 0:
        print("논문 업데이트 오류:")
        print(paper_result.stderr[-2000:])
else:
    print("\n경고: micro_f1 파싱 실패 — 논문 업데이트 건너뜀")
    print("수동으로 실행: python scripts/update_paper_3b_flatfc.py")
