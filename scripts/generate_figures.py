"""
논문용 Figure 생성 스크립트
생성 목록:
  fig1_pipeline.png        — 전체 데이터 파이프라인
  fig2_architecture.png    — FusionV2 vs V3 아키텍처 비교
  fig3_ablation.png        — Ablation Study 결과 바차트
  fig5_underrepresented.png — Underrepresented EC 비교
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE   = "#2E86AB"
GREEN  = "#48A999"
ORANGE = "#F4845F"
PURPLE = "#7B4B94"
RED    = "#E63946"
GRAY   = "#6C757D"
GOLD   = "#F4A261"
DARK   = "#1D3557"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         11,
})


# ══════════════════════════════════════════════════════
# Fig 1: 전체 파이프라인
# ══════════════════════════════════════════════════════
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    def box(ax, x, y, w, h, text, color, fontsize=10, textcolor="white", subtext=None):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor="white", linewidth=2, zorder=3)
        ax.add_patch(rect)
        ty = y + h/2 + (0.15 if subtext else 0)
        ax.text(x + w/2, ty, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)
        if subtext:
            ax.text(x + w/2, y + h/2 - 0.22, subtext, ha="center", va="center",
                    fontsize=8, color=textcolor, alpha=0.85, zorder=4)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2),
                    zorder=5)

    # ── 입력 ──
    box(ax, 0.3, 5.2, 2.4, 1.3, "UniProt\nSwiss-Prot", DARK, fontsize=10,
        subtext="270,628 proteins")

    # ── 전처리 branch ──
    # 서열 branch
    arrow(ax, 2.7, 5.85, 4.0, 5.85)
    box(ax, 4.0, 5.2, 2.2, 1.3, "ESM-2\n(650M, frozen)", BLUE, fontsize=9,
        subtext="CLS token → 1280-dim")

    # 구조 branch
    arrow(ax, 1.5, 5.2, 1.5, 4.0)
    box(ax, 0.3, 3.0, 2.4, 1.1, "AlphaFold PDB\n→ Contact Map", GREEN, fontsize=9,
        subtext="8Å threshold, 256×256")

    # contact pair branch
    arrow(ax, 1.5, 3.0, 1.5, 1.8)
    box(ax, 0.3, 0.9, 2.4, 1.0, "Contact Pair\nEmbedding", PURPLE, fontsize=9,
        subtext="top-32 LR, (32,2,1280)")

    # ── 모델 ──
    arrow(ax, 6.2, 5.85, 7.5, 5.85)
    arrow(ax, 2.7, 3.55, 7.5, 3.55)

    box(ax, 7.5, 4.9, 2.4, 1.5, "FusionV2", ORANGE, fontsize=10,
        subtext="AttentionContactEncoder\n+ GCA + Hier.Transformer")

    arrow(ax, 2.7, 1.4, 7.5, 1.4)
    arrow(ax, 6.2, 5.2, 6.8, 1.4)

    box(ax, 7.5, 0.7, 2.4, 1.5, "FusionV3", PURPLE, fontsize=10,
        subtext="ContactPairEncoder\n+ GCA + Hier.Transformer")

    # ── 앙상블 ──
    arrow(ax, 9.9, 5.65, 11.2, 5.65)
    arrow(ax, 9.9, 1.45, 11.2, 1.45)

    box(ax, 11.2, 3.3, 2.4, 2.3, "Ensemble\n(6:4 Weighted\nAverage)", GOLD, fontsize=10,
        textcolor=DARK)

    # ── 추론 ──
    arrow(ax, 13.6, 4.45, 14.3, 4.45)
    box(ax, 14.3, 3.7, 1.5, 1.5, "Hierarchical\nConstraint\nInference", RED, fontsize=8)

    arrow(ax, 15.8, 4.45, 15.8, 4.45)  # dummy, extend right edge

    # ── EC 예측 ──
    box(ax, 14.3, 0.8, 1.5, 2.5, "EC\nPrediction\nL1~L4", DARK, fontsize=9)
    arrow(ax, 15.05, 3.7, 15.05, 3.3)

    # 레이블
    ax.text(5.1, 6.7, "Sequence Path", fontsize=9, color=BLUE,
            fontweight="bold", ha="center")
    ax.text(1.5, 4.25, "Structure Path", fontsize=9, color=GREEN,
            fontweight="bold", ha="center", rotation=90)

    ax.set_title("EC Number Prediction Pipeline", fontsize=14,
                 fontweight="bold", pad=10, color=DARK)

    fig.savefig(OUT / "fig1_pipeline.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("fig1_pipeline.png 저장 완료")


# ══════════════════════════════════════════════════════
# Fig 2: FusionV2 vs V3 아키텍처 비교
# ══════════════════════════════════════════════════════
def fig2_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor("#F8F9FA")

    for ax in axes:
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_facecolor("#F8F9FA")

    def box(ax, x, y, w, h, text, color, fontsize=9.5, textcolor="white", subtext=None):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                               facecolor=color, edgecolor="white", linewidth=1.8, zorder=3)
        ax.add_patch(rect)
        ty = y + h/2 + (0.12 if subtext else 0)
        ax.text(x + w/2, ty, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)
        if subtext:
            ax.text(x + w/2, y + h/2 - 0.18, subtext, ha="center", va="center",
                    fontsize=7.5, color=textcolor, alpha=0.88, zorder=4)

    def arr(ax, x1, y1, x2, y2, color=DARK):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8), zorder=5)

    # ──────── V2 (왼쪽) ────────
    ax = axes[0]
    ax.set_title("FusionV2  (55.3M params)", fontsize=13, fontweight="bold",
                 color=DARK, pad=8)

    # 입력
    box(ax, 1.0, 8.8, 2.5, 0.7, "ESM-2 CLS  (1280)", BLUE, fontsize=9)
    box(ax, 4.5, 8.8, 2.5, 0.7, "Contact Map  (3×256×256)", GREEN, fontsize=8.5)

    # AttentionContactEncoder
    arr(ax, 5.75, 8.8, 5.75, 7.85)
    box(ax, 3.8, 6.9, 3.4, 0.85, "ResNet-50 backbone", GREEN,
        subtext="ImageNet pretrained → (B,2048,8,8)")
    arr(ax, 5.75, 6.9, 5.75, 6.05)
    box(ax, 3.8, 5.2, 3.4, 0.75, "Spatial Attention", GREEN,
        subtext="1×1 conv → sigmoid → weighted sum")
    arr(ax, 5.75, 5.2, 5.75, 4.45)
    box(ax, 3.8, 3.7, 3.4, 0.65, "GAP → Linear(2048→512)", GREEN,
        fontsize=8.5, subtext="+ LayerNorm + ReLU  →  (B,512)")

    # GCA
    arr(ax, 2.25, 8.8, 2.25, 4.0)
    arr(ax, 5.0,  3.7, 5.0,  3.2)
    box(ax, 1.2, 2.45, 5.6, 0.65, "GCA Fusion: gate·contact + (1-gate)·esm  →  (B,1024)",
        ORANGE, fontsize=8.5)
    ax.text(0.35, 3.65, "gate =\nsigmoid(\nLinear\n(1792→1))", fontsize=7.5,
            color=ORANGE, ha="center", va="center")

    # Head
    arr(ax, 4.0, 2.45, 4.0, 1.7)
    box(ax, 1.2, 0.85, 5.6, 0.75, "HierarchicalTransformerHead",
        DARK, subtext="4 Level Tokens → 2-layer Transformer → L1/L2/L3/L4 logits")

    # ──────── V3 (오른쪽) ────────
    ax = axes[1]
    ax.set_title("FusionV3  (36.0M params)", fontsize=13, fontweight="bold",
                 color=DARK, pad=8)

    box(ax, 1.0, 8.8, 2.5, 0.7, "ESM-2 CLS  (1280)", BLUE, fontsize=9)
    box(ax, 4.0, 8.8, 3.2, 0.7, "Contact Pairs  (32,2,1280)", PURPLE, fontsize=8.5)

    # ContactPairEncoder
    arr(ax, 5.6, 8.8, 5.6, 7.85)
    box(ax, 3.5, 6.9, 3.6, 0.85, "pair_proj: Linear(2560→512)", PURPLE,
        subtext="float16→32 | view (B,32,2560)")
    arr(ax, 5.5, 6.9, 5.5, 6.05)
    box(ax, 3.5, 5.2, 3.6, 0.75, "2-layer TransformerEncoder", PURPLE,
        subtext="d=512, nhead=8, PreNorm  →  (B,32,512)")
    arr(ax, 5.5, 5.2, 5.5, 4.45)
    box(ax, 3.5, 3.7, 3.6, 0.65, "Attention Pooling → out_proj", PURPLE,
        fontsize=8.5, subtext="softmax(Linear(512→1)) → (B,512)")

    # GCA
    arr(ax, 2.25, 8.8, 2.25, 4.0)
    arr(ax, 5.3,  3.7, 5.3,  3.2)
    box(ax, 1.2, 2.45, 5.6, 0.65, "GCA Fusion: gate·contact + (1-gate)·esm  →  (B,1024)",
        ORANGE, fontsize=8.5)

    arr(ax, 4.0, 2.45, 4.0, 1.7)
    box(ax, 1.2, 0.85, 5.6, 0.75, "HierarchicalTransformerHead",
        DARK, subtext="4 Level Tokens → 2-layer Transformer → L1/L2/L3/L4 logits")

    fig.suptitle("Model Architecture Comparison", fontsize=14,
                 fontweight="bold", color=DARK, y=1.01)
    fig.savefig(OUT / "fig2_architecture.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("fig2_architecture.png 저장 완료")


# ══════════════════════════════════════════════════════
# Fig 3: Ablation Study 바차트
# ══════════════════════════════════════════════════════
def fig3_ablation():
    models = [
        "B2\n(cascade)",
        "B3\n(struct only)",
        "B1\n(seq only)",
        "FusionV2\n20ep",
        "FusionV3\n20ep",
        "FusionV3\n30ep",
        "Ensemble\nV2(20)+V3(30)",
    ]
    l4_f1 = [0.4421, 0.8608, 0.8853, 0.9209, 0.9226, 0.9384, 0.9437]
    colors = [GRAY, GREEN, BLUE, ORANGE, PURPLE, PURPLE, RED]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    bars = ax.bar(models, l4_f1, color=colors, width=0.6, zorder=3,
                  edgecolor="white", linewidth=1.5)

    # HIT-EC 기준선
    ax.axhline(0.9300, color=DARK, linestyle="--", linewidth=2, zorder=4)
    ax.text(6.5, 0.9315, "HIT-EC SOTA (0.9300)", fontsize=9.5,
            color=DARK, fontweight="bold", ha="right")

    # 값 레이블
    for bar, val in zip(bars, l4_f1):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=DARK)

    # 강조: SOTA 초과 표시
    for i, (bar, val) in enumerate(zip(bars, l4_f1)):
        if val > 0.9300:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_y() + 0.01, "★", ha="center",
                    fontsize=14, color="gold", zorder=5)

    ax.set_ylim(0.35, 0.98)
    ax.set_ylabel("Level 4 Micro F1", fontsize=12, fontweight="bold")
    ax.set_title("Ablation Study — Level 4 Micro F1 (Test Set)", fontsize=13,
                 fontweight="bold", color=DARK, pad=10)
    ax.tick_params(axis="x", labelsize=9)

    # 범례
    legend_patches = [
        mpatches.Patch(color=GRAY,   label="Baseline (cascade)"),
        mpatches.Patch(color=GREEN,  label="Structure-only (B3)"),
        mpatches.Patch(color=BLUE,   label="Sequence-only (B1)"),
        mpatches.Patch(color=ORANGE, label="FusionV2 (spatial attn)"),
        mpatches.Patch(color=PURPLE, label="FusionV3 (contact pair)"),
        mpatches.Patch(color=RED,    label="Ensemble V2+V3 (proposed)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9,
              framealpha=0.9, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_ablation.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("fig3_ablation.png 저장 완료")


# ══════════════════════════════════════════════════════
# Fig 5: Underrepresented EC 비교 + 전체 F1 scatter
# ══════════════════════════════════════════════════════
def fig5_underrepresented():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#F8F9FA")

    # ── 왼쪽: Underrepresented 바차트 ──
    ax = axes[0]
    ax.set_facecolor("#F8F9FA")

    labels = ["DeepECT", "B1\n(seq only)", "B3\n(struct)", "FusionV2",
              "FusionV3\n20ep", "FusionV3\n30ep", "Ensemble\nV2+V3",
              "CLEAN", "HIT-EC"]
    vals   = [0.47, 0.4557, 0.3810, 0.6422, 0.6118, 0.6923, 0.7035, 0.73, 0.77]
    clrs   = [GRAY, BLUE, GREEN, ORANGE, PURPLE, PURPLE, RED, GRAY, DARK]
    hatch  = ["", "", "", "", "", "", "", "//", "//"]  # 선행연구

    bars = ax.bar(labels, vals, color=clrs, width=0.65, zorder=3,
                  edgecolor="white", linewidth=1.5)
    for bar, h in zip(bars, hatch):
        bar.set_hatch(h)

    ax.axhline(0.77, color=DARK,  linestyle="--", lw=1.8, zorder=4)
    ax.axhline(0.73, color=GRAY,  linestyle=":",  lw=1.5, zorder=4)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=DARK)

    ax.set_ylim(0.30, 0.86)
    ax.set_ylabel("Underrepresented Micro F1 (N≤25)", fontsize=10.5, fontweight="bold")
    ax.set_title("Underrepresented EC Performance", fontsize=12,
                 fontweight="bold", color=DARK, pad=8)
    ax.tick_params(axis="x", labelsize=8)

    ax.text(7.35, 0.775, "HIT-EC", fontsize=8, color=DARK, fontweight="bold")
    ax.text(7.35, 0.735, "CLEAN",  fontsize=8, color=GRAY, fontweight="bold")

    # ── 오른쪽: Overall vs Underrepresented 산점도 ──
    ax2 = axes[1]
    ax2.set_facecolor("#F8F9FA")

    pts = {
        "B1 (seq)":        (0.8853, 0.4557, BLUE,   "o"),
        "B3 (struct)":     (0.8608, 0.3810, GREEN,  "o"),
        "FusionV2":        (0.9209, 0.6422, ORANGE, "s"),
        "FusionV3 20ep":   (0.9226, 0.6118, PURPLE, "s"),
        "FusionV3 30ep":   (0.9384, 0.6923, PURPLE, "D"),
        "Ensemble (ours)": (0.9437, 0.7035, RED,    "*"),
        "HIT-EC":          (0.9300, 0.7700, DARK,   "^"),
        "CLEAN":           (None,   0.7300, GRAY,   "^"),
    }

    for name, (x, y, c, m) in pts.items():
        if x is None:
            continue
        sz = 200 if m == "*" else 100
        ax2.scatter(x, y, color=c, marker=m, s=sz, zorder=5,
                    edgecolors="white", linewidths=1.2)
        offx, offy = 0.001, 0.008
        if name == "HIT-EC":
            offx, offy = 0.001, -0.018
        if name == "Ensemble (ours)":
            offx, offy = -0.006, 0.012
        ax2.text(x + offx, y + offy, name, fontsize=8, color=c, fontweight="bold")

    # 이상적 방향 화살표
    ax2.annotate("", xy=(0.948, 0.79), xytext=(0.870, 0.36),
                 arrowprops=dict(arrowstyle="-|>", color="lightgray", lw=1.5))
    ax2.text(0.872, 0.38, "Better →", fontsize=8, color="lightgray", rotation=60)

    ax2.set_xlabel("L4 Micro F1 (Overall)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Underrepresented Micro F1", fontsize=10.5, fontweight="bold")
    ax2.set_title("Overall vs Underrepresented EC", fontsize=12,
                  fontweight="bold", color=DARK, pad=8)
    ax2.set_xlim(0.85, 0.955)
    ax2.set_ylim(0.33, 0.82)

    fig.suptitle("Underrepresented EC Analysis", fontsize=13,
                 fontweight="bold", color=DARK, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_underrepresented.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("fig5_underrepresented.png 저장 완료")


# ══════════════════════════════════════════════════════
# Fig 6: Training Curves (V3 20ep vs 30ep)
# ══════════════════════════════════════════════════════
def fig6_training_curves():
    import re
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]

    def parse_log(path):
        epochs, vals = [], []
        for line in Path(path).read_text().splitlines():
            m = re.search(r'\[(\d+)/\d+\].*micro_f1=([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                vals.append(float(m.group(2)))
        return epochs, vals

    logs = {
        "FusionV3 (20 ep)": ROOT / "outputs/results/fusion_v3.log",
        "FusionV3 (30 ep)": ROOT / "outputs/results/fusion_v3_30ep.log",
        "FusionV2 (20 ep)": ROOT / "outputs/results/fusion_v2.log",
        "FusionV2 (30 ep, ongoing)": ROOT / "outputs/results/fusion_v2_30ep.log",
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    palette = {
        "FusionV3 (20 ep)":        (PURPLE, "--", 1.6),
        "FusionV3 (30 ep)":        (PURPLE, "-",  2.2),
        "FusionV2 (20 ep)":        (ORANGE, "--", 1.6),
        "FusionV2 (30 ep, ongoing)": (ORANGE, "-", 2.2),
    }

    for name, path in logs.items():
        if not Path(path).exists():
            continue
        ep, vf = parse_log(path)
        if not ep:
            continue
        color, ls, lw = palette[name]
        ax.plot(ep, vf, color=color, linestyle=ls, linewidth=lw,
                label=name, marker="o", markersize=3.5, markevery=2)
        # 마지막 점에 현재 best 표시
        ax.scatter(ep[-1], max(vf), color=color, s=60, zorder=6,
                   edgecolors="white", linewidths=1.2)

    ax.axhline(0.9300, color=DARK, linestyle=":", linewidth=1.8)
    ax.text(1, 0.933, "HIT-EC (0.9300)", fontsize=9, color=DARK, fontweight="bold")

    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Val Micro F1", fontsize=12, fontweight="bold")
    ax.set_title("Training Curves — Val Micro F1", fontsize=13,
                 fontweight="bold", color=DARK, pad=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(0.60, 0.97)

    fig.tight_layout()
    fig.savefig(OUT / "fig6_training_curves.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("fig6_training_curves.png 저장 완료")


if __name__ == "__main__":
    print("Figure 생성 시작...")
    fig1_pipeline()
    fig2_architecture()
    fig3_ablation()
    fig5_underrepresented()
    fig6_training_curves()
    print(f"\n모든 Figure 저장 완료: {OUT}")
