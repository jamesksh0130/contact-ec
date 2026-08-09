"""
전체 모델 아키텍처 시각화
출력: outputs/architecture.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis("off")
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

# ── 색상 팔레트 ──────────────────────────────────────────────
C_INPUT   = "#4ade80"   # 입력: 초록
C_SEQ     = "#60a5fa"   # 서열 경로: 파랑
C_STRUCT  = "#f472b6"   # 구조 경로: 핑크
C_FUSION  = "#fbbf24"   # 융합: 노랑
C_HEAD    = "#a78bfa"   # 헤드: 보라
C_OUTPUT  = "#fb923c"   # 출력: 주황
C_TEXT    = "#f1f5f9"   # 텍스트
C_ARROW   = "#94a3b8"   # 화살표

def box(ax, x, y, w, h, color, text, fontsize=9, alpha=0.9, radius=0.3,
        text2=None, text2_color="#94a3b8"):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          facecolor=color, edgecolor="white",
                          linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    ty = y if text2 is None else y + 0.15
    ax.text(x, ty, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", zorder=4)
    if text2:
        ax.text(x, y - 0.25, text2, ha="center", va="center",
                fontsize=7.5, color=text2_color, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, style="-|>", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)

def label(ax, x, y, text, color=C_TEXT, fontsize=8.5, ha="center"):
    ax.text(x, y, text, ha=ha, va="center",
            fontsize=fontsize, color=color, zorder=5)

# ══════════════════════════════════════════════════════════════
# 제목
ax.text(10, 13.4, "EC Number Prediction — Full Model Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=C_TEXT, zorder=5)
ax.text(10, 13.0, "ESM-2 (Sequence) + 2D ResNet-18 (Contact Map) → Hierarchical EC Head",
        ha="center", va="center", fontsize=9, color="#94a3b8", zorder=5)

# ══════════════════════════════════════════════════════════════
# ① 입력
box(ax, 5.0, 12.0, 3.4, 0.7, C_INPUT,
    "Amino Acid Sequence", 9,
    text2='"MKTAYIAKQRKLIF..." (len ≤ 1024)')
box(ax, 13.0, 12.0, 3.4, 0.7, C_INPUT,
    "UniProt ID → AlphaFold DB", 9,
    text2='.pdb  (Cα 3D coordinates)')

# 경로 레이블
label(ax, 5.0, 11.35, "▼  Path A: Sequence", C_SEQ, 8)
label(ax, 13.0, 11.35, "▼  Path B: Structure", C_STRUCT, 8)

# ══════════════════════════════════════════════════════════════
# ② ESM-2
arrow(ax, 5.0, 11.65, 5.0, 11.1)
box(ax, 5.0, 10.6, 3.6, 0.8, C_SEQ,
    "ESM-2  (frozen)", 10,
    text2="facebook/esm2_t33_650M_UR50D  |  650M params")
label(ax, 5.0, 10.05, "↓  CLS token  →  (1 × 1280)", C_SEQ, 8)

# ══════════════════════════════════════════════════════════════
# ② Contact Map 처리
arrow(ax, 13.0, 11.65, 13.0, 11.1)

# Cα 거리 계산
box(ax, 13.0, 10.6, 3.6, 0.8, C_STRUCT,
    "Distance Matrix + 8Å Threshold", 9,
    text2="Binary Contact Map  (N × N)")
arrow(ax, 13.0, 10.2, 13.0, 9.65)

# Resize
box(ax, 13.0, 9.3, 3.6, 0.65, C_STRUCT,
    "Resize  →  256 × 256", 9,
    text2="scipy.ndimage.zoom  |  float32")
label(ax, 13.0, 8.85, "↓  (1 × 256 × 256)", C_STRUCT, 8)

# ResNet-18
arrow(ax, 13.0, 9.0, 13.0, 8.45)
box(ax, 13.0, 8.05, 3.6, 0.75, C_STRUCT,
    "2D ResNet-18", 10,
    text2="in_channels=1  →  GlobalAvgPool  →  512-d")
label(ax, 13.0, 7.55, "↓  (1 × 512)", C_STRUCT, 8)

# ══════════════════════════════════════════════════════════════
# ESM-2 feature 화살표 아래로
arrow(ax, 5.0, 10.2, 5.0, 7.4)
box(ax, 5.0, 7.1, 3.6, 0.55, C_SEQ,
    "Sequence Feature  (1 × 1280)", 9, alpha=0.7)

# ══════════════════════════════════════════════════════════════
# ③ Concat / Fusion
arrow(ax, 6.8, 7.1, 8.35, 6.4)
arrow(ax, 13.0, 7.7, 11.55, 6.4)

box(ax, 9.95, 6.1, 3.8, 0.7, C_FUSION,
    "Concat  →  (1 × 1792)", 10, alpha=0.95)
arrow(ax, 9.95, 5.75, 9.95, 5.2)

box(ax, 9.95, 4.85, 3.8, 0.65, C_FUSION,
    "Linear(1792→1024) + ReLU + Dropout(0.3)", 8.5, alpha=0.95)
arrow(ax, 9.95, 4.52, 9.95, 4.0)

box(ax, 9.95, 3.72, 3.0, 0.5, C_FUSION,
    "Shared Feature  (1 × 1024)", 9, alpha=0.8)

# ══════════════════════════════════════════════════════════════
# ④ Hierarchical Heads
head_xs  = [4.0, 7.5, 12.3, 16.0]
head_cls = [6,   65,  242,  4647]
head_lv  = ["Level 1", "Level 2", "Level 3", "Level 4"]
head_eg  = ["1.-.-.−", "1.1.-.−", "1.1.1.−", "1.1.1.1"]
head_w   = [6, 65, 242, 4647]
head_col = ["#6366f1", "#8b5cf6", "#a855f7", "#c084fc"]

# 부채꼴 화살표
for hx in head_xs:
    arrow(ax, 9.95, 3.47, hx, 2.85, color="#fbbf24", lw=1.2)

for hx, nc, lv, eg, hc in zip(head_xs, head_cls, head_lv, head_eg, head_col):
    box(ax, hx, 2.55, 3.1, 0.58, hc,
        f"FC(1024→{nc})", 9,
        text2=f"{lv}  |  e.g. {eg}")

# 화살표 → loss
for hx in head_xs:
    arrow(ax, hx, 2.27, hx, 1.75, color="#94a3b8", lw=1.1)

# Loss 박스
for hx, lv in zip(head_xs, head_lv):
    w_map = {6: 0.1, 65: 0.1, 242: 0.2, 4647: 0.6}
    nc_map = {6: 6, 65: 65, 242: 242, 4647: 4647}
    nc = nc_map[{6:6,65:65,242:242,4647:4647}[
        [6,65,242,4647][head_lv.index(lv)]]]
    wt = [0.1, 0.1, 0.2, 0.6][head_lv.index(lv)]
    box(ax, hx, 1.48, 3.1, 0.45, "#334155",
        f"CE Loss  ×{wt}", 8,
        text2="Masked if label incomplete")

# 합산
for hx in head_xs:
    arrow(ax, hx, 1.25, 9.95, 0.78, color="#fb923c", lw=1.1)

box(ax, 9.95, 0.52, 5.0, 0.55, C_OUTPUT,
    "L = 0.1·L1 + 0.1·L2 + 0.2·L3 + 0.6·L4", 9.5, alpha=0.95,
    text2=None)

# ══════════════════════════════════════════════════════════════
# 범례
legend_items = [
    mpatches.Patch(color=C_INPUT,  label="Input"),
    mpatches.Patch(color=C_SEQ,    label="Sequence Path (ESM-2)"),
    mpatches.Patch(color=C_STRUCT, label="Structure Path (ResNet-18)"),
    mpatches.Patch(color=C_FUSION, label="Fusion Layer"),
    mpatches.Patch(color=C_HEAD,   label="Hierarchical Heads"),
    mpatches.Patch(color=C_OUTPUT, label="Loss"),
]
ax.legend(handles=legend_items, loc="lower left",
          fontsize=8, framealpha=0.3,
          facecolor="#0f172a", edgecolor="#475569",
          labelcolor=C_TEXT,
          bbox_to_anchor=(0.01, 0.01))

# 오른쪽 메모
note_lines = [
    "ESM-2  (650M)  → CLS token →  1280-d",
    "ResNet-18  (in_ch=1)  →  512-d",
    "Fusion dim  =  1280 + 512  =  1792  →  1024",
    "EC Classes: L1=6, L2=65, L3=242, L4=4647",
    "AlphaFold miss  →  zeros contact map",
    "Training: Phase1 frozen / Phase2 partial unfreeze",
]
for i, ln in enumerate(note_lines):
    ax.text(19.9, 3.5 + i*0.42, ln,
            ha="right", va="center", fontsize=7.5,
            color="#94a3b8", zorder=5,
            fontfamily="monospace")

plt.tight_layout()
plt.savefig("outputs/architecture.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("저장 완료: outputs/architecture.png")
