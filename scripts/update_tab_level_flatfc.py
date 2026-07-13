"""
Contact-EC flat FC 학습 완료 후 tab:level 자동 업데이트 스크립트
1. evaluate.py 실행 (test set, hierarchical_inference)
2. 결과에서 L1-L4 micro F1 추출
3. main.tex tab:level에 Contact-EC flat FC 행 삽입
"""
import subprocess, json, re, sys
from pathlib import Path

ROOT = Path("/home/user/Desktop/unlv")
CKPT = ROOT / "outputs/checkpoints/flatfc_orig_best.pt"
RESULT_JSON = ROOT / "outputs/results/fusion_v2_flatfc_test_hier_results.json"
MAIN_TEX = ROOT / "paper/main.tex"

# ── 1. 체크포인트 확인 ───────────────────────────────────────────────────
if not CKPT.exists():
    print(f"❌ 체크포인트 없음: {CKPT}")
    sys.exit(1)
print(f"✅ 체크포인트 확인: {CKPT}")

# ── 2. evaluate.py 실행 (아직 결과 없을 때만) ──────────────────────────
if not RESULT_JSON.exists():
    print("▶ evaluate.py 실행 중...")
    cmd = [
        "python", str(ROOT / "evaluate.py"),
        "--model", "fusion_v2_flatfc",
        "--checkpoint", str(CKPT),
        "--split", "test",
        "--hierarchical_inference",
        "--gpu", "0",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
        sys.exit(1)

if not RESULT_JSON.exists():
    print("❌ 결과 파일 생성 실패")
    sys.exit(1)

# ── 3. 결과 로드 ───────────────────────────────────────────────────────
d = json.loads(RESULT_JSON.read_text())
l1 = d["level1"]["micro_f1"]
l2 = d["level2"]["micro_f1"]
l3 = d["level3"]["micro_f1"]
l4 = d["level4"]["micro_f1"]
print(f"\n=== Contact-EC flat FC 결과 ===")
print(f"  L1={l1:.4f}  L2={l2:.4f}  L3={l3:.4f}  L4={l4:.4f}")

# ── 4. 현재 best values 확인 (bold 처리 결정) ──────────────────────────
# 기존 값 (B1, B2, B3, Hier)
existing = {
    "B1":   (0.9158, 0.8405, 0.8141, 0.8121),
    "B2":   (0.9218, 0.8639, 0.8443, 0.8322),
    "B3":   (0.8801, 0.8602, 0.8628, 0.8987),
    "Hier": (0.9164, 0.8902, 0.8917, 0.9347),
    "FlatFC": (l1, l2, l3, l4),
}
all_vals = list(zip(*existing.values()))
bests = [max(v) for v in all_vals]
print(f"\n  Best per level: L1={bests[0]:.4f}  L2={bests[1]:.4f}  L3={bests[2]:.4f}  L4={bests[3]:.4f}")


def fmt(val, best, prec=4):
    s = f"{val:.{prec}f}"
    return f"\\textbf{{{s}}}" if abs(val - best) < 1e-9 else s


row = (
    f"Contact-EC flat FC          "
    f"& {fmt(l1, bests[0])} "
    f"& {fmt(l2, bests[1])} "
    f"& {fmt(l3, bests[2])} "
    f"& {fmt(l4, bests[3])} \\\\"
)
print(f"\n  LaTeX 행:\n  {row}")

# ── 5. main.tex 업데이트 ──────────────────────────────────────────────
tex = MAIN_TEX.read_text()

# Find the midrule separator before Contact-EC-Hier
# Old pattern: \midrule\nContact-EC-Hier
old_pattern = "\\midrule\nContact-EC-Hier"
new_pattern = f"\\midrule\n{row}\nContact-EC-Hier"

if old_pattern not in tex:
    print(f"\n❌ 삽입 위치를 찾지 못했습니다. 패턴: {repr(old_pattern)}")
    # Try alternate pattern
    old_pattern2 = "\\midrule\r\nContact-EC-Hier"
    if old_pattern2 in tex:
        new_pattern2 = f"\\midrule\r\n{row}\r\nContact-EC-Hier"
        tex = tex.replace(old_pattern2, new_pattern2, 1)
        MAIN_TEX.write_text(tex)
        print("✅ (CRLF 패턴으로) main.tex 업데이트 완료!")
    else:
        print("  검색할 문자열 주변 텍스트:")
        idx = tex.find("Contact-EC-Hier")
        print(repr(tex[max(0, idx-50):idx+80]))
        sys.exit(1)
else:
    if new_pattern in tex:
        print("\n⚠️  이미 Contact-EC flat FC 행이 있습니다. 스킵.")
    else:
        tex = tex.replace(old_pattern, new_pattern, 1)
        MAIN_TEX.write_text(tex)
        print("\n✅ main.tex tab:level 업데이트 완료!")

# ── 6. 텍스트 단락 업데이트 제안 ─────────────────────────────────────
gap_flatfc_vs_hier = (d["level4"]["micro_f1"] - 0.9347) * 100  # vs Hier
gap_flatfc_vs_b1   = (d["level4"]["micro_f1"] - 0.8121) * 100  # vs B1

print(f"\n=== 텍스트 업데이트 제안 ===")
print(f"  Contact-EC flat FC L4: {l4:.4f}")
print(f"  vs Contact-EC-Hier: {gap_flatfc_vs_hier:+.1f} pp")
print(f"  vs B1: {gap_flatfc_vs_b1:+.1f} pp")
print()
print("  단락 업데이트 (1092~1102 줄 근방)에서 flat FC 관련 문구 추가 필요:")
print(f"  'Contact-EC flat FC achieves L4={l4:.4f}, {abs(gap_flatfc_vs_hier):.1f} pp {'below' if gap_flatfc_vs_hier < 0 else 'above'} Contact-EC-Hier'")
