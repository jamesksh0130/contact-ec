"""
저널 품질 figure 재생성 스크립트

재생성 대상:
  1. learning_curves_all_models.png  (EC-Bench hard val 학습 곡선)
  2. level_breakdown_bar.png          (Level별 micro F1 막대 그래프)
  3. contact_map_3channel.png         (3채널 Contact Map, 제목 수정)
  4. gap_decomp.pdf                   (Gap 분해 figure, E2E 결과 포함)

저널 스타일 기준:
  - Bioinformatics / Nature Communications 서식
  - 300 DPI, Arial 폰트
  - 색약자 배려 팔레트 (Wong 8-color)
  - 상단/우측 spine 제거 (Tufte style)
  - 두 칼럼 기준: 6.75인치 너비
"""
import re
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "outputs" / "results"
PAPER= ROOT / "paper"

# ── Wong 8-color (colorblind-safe) ─────────────────────────────────────────
WONG = {
    "black":  "#000000",
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "red":    "#D55E00",
    "purple": "#CC79A7",
}

# ── 공통 rcParams ───────────────────────────────────────────────────────────
def set_journal_style():
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "DejaVu Sans"],
        "font.size":          9,
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "0.8",
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "grid.linewidth":     0.5,
        "grid.alpha":         0.4,
        "lines.linewidth":    1.5,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Learning Curves (EC-Bench hard val)
# ══════════════════════════════════════════════════════════════════════════════
def parse_log(path, max_epochs=30):
    """에폭별 micro_f1 추출."""
    vals = []
    pat  = re.compile(r"\[(\d+)/\d+\].*?micro_f1=([\d.]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                ep, v = int(m.group(1)), float(m.group(2))
                if ep <= max_epochs:
                    vals.append((ep, v))
    vals.sort()
    if not vals:
        return [], []
    epochs = [x[0] for x in vals]
    micro  = [x[1] for x in vals]
    return epochs, micro


def plot_learning_curves():
    set_journal_style()
    LOG = ROOT / "outputs" / "logs"

    # ── 데이터 로드 ──
    ep_b1,  f1_b1  = parse_log(LOG / "ecbench_b1.log",  30)
    ep_b2,  f1_b2  = parse_log(LOG / "ecbench_b2.log",  30)
    # B3: main log만 사용 (resume은 서로 다른 체크포인트에서 시작해 불연속)
    ep_b3,  f1_b3  = parse_log(LOG / "ecbench_b3.log",  29)

    # Contact-EC-Hier: Phase1(24→7 resume) + Phase2(20)
    ep_fv2,  f1_fv2  = parse_log(LOG / "ecbench_fv2_phase1.log", 24)
    ep_fv2r, f1_fv2r = parse_log(LOG / "ecbench_fv2_phase1_resume.log", 7)
    last_fv2 = max(ep_fv2) if ep_fv2 else 24
    for e, v in zip(ep_fv2r, f1_fv2r):
        ep_fv2.append(last_fv2 + e)
        f1_fv2.append(v)
    ep_fv2p2, f1_fv2p2 = parse_log(LOG / "ecbench_fv2_phase2.log", 20)
    phase2_offset = max(ep_fv2) if ep_fv2 else 31
    for e, v in zip(ep_fv2p2, f1_fv2p2):
        ep_fv2.append(phase2_offset + e)
        f1_fv2.append(v)

    fig, ax = plt.subplots(figsize=(6.75, 3.6))

    # ── 색상 및 스타일 ──
    c_b1  = WONG["blue"]
    c_b2  = WONG["red"]
    c_b3  = WONG["green"]
    c_fv2 = WONG["orange"]

    ax.plot(ep_b1, f1_b1,   color=c_b1,  ls="-",  marker="o", ms=3,
            markevery=5, label="B1 (ESM-2, flat FC)")
    ax.plot(ep_b2, f1_b2,   color=c_b2,  ls="--", marker="s", ms=3,
            markevery=5, label="B2 (ESM-2, hier. FC)")
    ax.plot(ep_b3, f1_b3,   color=c_b3,  ls="-.", marker="^", ms=3,
            markevery=5, label="B3 (Contact map only)")
    ax.plot(ep_fv2, f1_fv2, color=c_fv2, ls="-",  marker="D", ms=3,
            markevery=5, lw=2.0, label="Contact-EC-Hier (ours)")

    # Phase 2 시작 구분선
    p2_start = phase2_offset + 1
    ax.axvline(p2_start, color="0.5", ls=":", lw=1.0)
    ax.text(p2_start + 0.5, 0.30, "Phase 2\nstarts",
            fontsize=7, color="0.5", va="bottom")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val L4 Micro F1\n(EC-Bench hard val, $\\leq$30% seq. identity)")
    ax.set_xlim(0, max(ep_fv2) + 1)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(True, which="major", axis="y")
    ax.legend(loc="lower right", ncol=1, frameon=True)

    # Phase annotation
    ax.text(phase2_offset / 2, 0.97, "Phase 1 (frozen ESM-2)",
            ha="center", va="top", fontsize=7, color="0.4")
    ax.text(phase2_offset + (max(ep_fv2) - phase2_offset) / 2, 0.97,
            "Phase 2 (partial unfreeze)",
            ha="center", va="top", fontsize=7, color="0.4")

    fig.tight_layout()
    out_path = OUT / "learning_curves_all_models.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[1] 저장: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Level-by-Level Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_level_breakdown():
    set_journal_style()

    # Tab. 5 (tab:level) 수치 (random-split test)
    models = ["B1\n(ESM-2)", "B2\n(ESM-2+hier.FC)", "B3\n(Contact)", "Contact-EC\n(flat FC)", "Contact-EC-Hier\n(ours)"]
    levels = ["L1", "L2", "L3", "L4"]
    data   = np.array([
        # B1       B2        B3        flat FC   Contact-EC-Hier
        [0.9158,  0.9218,   0.8801,   0.8500,   0.9164],   # L1
        [0.8405,  0.8639,   0.8602,   0.7599,   0.8902],   # L2
        [0.8141,  0.8443,   0.8628,   0.7329,   0.8917],   # L3
        [0.8121,  0.8322,   0.8987,   0.8075,   0.9347],   # L4
    ])  # shape: (4 levels, 5 models)

    colors = [WONG["blue"], WONG["red"], WONG["green"], WONG["sky"], WONG["orange"]]
    n_models = len(models)
    n_levels = len(levels)
    x = np.arange(n_levels)
    width = 0.15
    offsets = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * width

    fig, ax = plt.subplots(figsize=(7.5, 3.4))

    bars_list = []
    for i, (mdl, col, off) in enumerate(zip(models, colors, offsets)):
        vals = data[:, i]
        bars = ax.bar(x + off, vals, width, color=col, alpha=0.85,
                      edgecolor="white", linewidth=0.5, zorder=3)
        bars_list.append(bars)
        # 값 레이블 (L4만)
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if j == 3:  # L4 only
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6, color="0.2")

    ax.set_ylabel("Micro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0.70, 0.97)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.025))
    ax.grid(True, which="major", axis="y", zorder=0)

    legend_handles = [
        mpatches.Patch(color=colors[i], alpha=0.85, label=models[i].replace("\n", " "))
        for i in range(n_models)
    ]
    ax.legend(handles=legend_handles, loc="lower right", ncol=2, frameon=True,
              handlelength=1.0, handleheight=0.8, fontsize=7)

    # 가장 높은 bar에 별표 (L4 Contact-EC-Hier)
    ax.annotate("★", xy=(x[3] + offsets[4], data[3, 4] + 0.005),
                ha="center", va="bottom", fontsize=9, color=WONG["orange"])

    fig.tight_layout()
    out_path = OUT / "level_breakdown_bar.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[2] 저장: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: 3-Channel Contact Map (제목 수정 + 레이아웃 개선)
# ══════════════════════════════════════════════════════════════════════════════
def plot_contact_maps():
    set_journal_style()

    cmap_dir = ROOT / "data" / "processed" / "contact_maps"
    meta_csv = ROOT / "data" / "expa" / "dataset_meta_reenc.csv"

    if not cmap_dir.exists():
        print("[3] contact_maps 디렉토리 없음, 스킵")
        return
    if not meta_csv.exists():
        print("[3] dataset_meta_reenc.csv 없음, 스킵")
        return

    import pandas as pd
    meta = pd.read_csv(meta_csv)

    ec_classes = {
        1: ("Oxidoreductase",  "1."),
        2: ("Transferase",     "2."),
        3: ("Hydrolase",       "3."),
        4: ("Lyase",           "4."),
        5: ("Isomerase",       "5."),
        6: ("Ligase",          "6."),
        7: ("Translocase",     "7."),
    }

    selected = {}
    for lvl, (name, prefix) in ec_classes.items():
        sub = meta[meta["ec_chosen"].astype(str).str.startswith(prefix)]
        for _, row in sub.iterrows():
            uid = row["accession"]
            f = cmap_dir / f"{uid}.npy"
            if f.exists():
                arr = np.load(f).astype(np.float32)
                if arr.shape == (256, 256) and arr.sum() > 100:
                    ec_str = str(row["ec_chosen"])
                    selected[lvl] = (uid, name, ec_str, arr)
                    break

    if not selected:
        print("[3] 충분한 contact map 없음, 스킵")
        return

    n_rows = len(selected)
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(6.75, 1.55 * n_rows))
    gs  = GridSpec(n_rows, 3, figure=fig,
                   wspace=0.06, hspace=0.10,
                   left=0.22, right=0.98, top=0.94, bottom=0.02)

    col_titles = [
        "Ch1: All contacts (d < 8 A)",
        "Ch2: Short-range (|i-j| < 12)",
        "Ch3: Long-range (|i-j| >= 12)",
    ]
    cmaps_ch = ["Blues", "Greens", "Reds"]

    idx     = np.arange(256)
    diff    = np.abs(idx[:, None] - idx[None, :])
    short_m = (diff < 12).astype(np.float32)
    long_m  = 1.0 - short_m

    ax_grid = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(n_rows)]

    for col, ct in enumerate(col_titles):
        ax_grid[0][col].set_title(ct, fontsize=7.5, pad=3, fontweight="bold")

    for row_i, (lvl_id, (uid, ec_name, ec_str, binary)) in enumerate(sorted(selected.items())):
        channels = [binary, binary * short_m, binary * long_m]

        for col, (ch, cm) in enumerate(zip(channels, cmaps_ch)):
            ax = ax_grid[row_i][col]
            ax.imshow(ch, cmap=cm, vmin=0, vmax=1,
                      interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_visible(True)

        ax_grid[row_i][0].text(
            -0.30, 0.5, "EC " + ec_str + "\n(" + ec_name + ")",
            transform=ax_grid[row_i][0].transAxes,
            fontsize=6.8, va="center", ha="right",
            multialignment="right", linespacing=1.4,
        )

    fig.suptitle(
        "Three-Channel Contact Maps --- Representative Enzymes (EC Level 1-7)",
        fontsize=8.5, fontweight="bold"
    )

    out_path = OUT / "contact_map_3channel.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[3] 저장: {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Gap Decomposition (optional E2E value)
# ══════════════════════════════════════════════════════════════════════════════
def plot_gap_decomp(e2e_val=None):
    """
    e2e_val: float or None.
      None keeps the E2E bar as an explicitly unlabeled optional result.
    """
    set_journal_style()

    # 데이터
    labels = [
        "Contact-EC\n(SP-2018)",
        "Contact-EC-ExpA\n(SP-2026, 243K)",
        "Contact-EC-E2E\n(SP-2026, E2E ft)",
        "HIT-EC\n(temporal test)",
    ]
    values  = [0.6032, 0.7209, e2e_val if e2e_val is not None else 0.0, 0.8471]
    colors  = [WONG["blue"], WONG["blue"], WONG["sky"], WONG["red"]]
    hatches = ["", "//", "xx", ""]
    alphas  = [0.85, 0.85, 0.85, 0.85]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    x = np.arange(len(labels))
    bar_w = 0.5
    bars = []
    for i, (v, c, h, a) in enumerate(zip(values, colors, hatches, alphas)):
        if v == 0.0 and e2e_val is None:
            bar = ax.bar(x[i], 0.01, bar_w, color="0.85", hatch=h,
                         edgecolor="0.6", linewidth=0.8, alpha=0.5, zorder=3)
            ax.text(x[i], 0.535, "not run\n(optional)", ha="center", va="bottom",
                    fontsize=7, color="0.5", style="italic")
        else:
            bar = ax.bar(x[i], v, bar_w, color=c, hatch=h,
                         edgecolor="white", linewidth=0.8, alpha=a, zorder=3)
            # 레이블 위치: bar 내부 상단 (overlap 방지)
            label_y = v - 0.018 if v > 0.55 else v + 0.010
            va_str  = "top" if v > 0.55 else "bottom"
            ax.text(x[i], label_y, f"{v:.4f}", ha="center", va=va_str,
                    fontsize=9, fontweight="bold", color="white" if v > 0.55 else "black")
        bars.append(bar)

    # 화살표 + 주석
    def draw_arrow(x0, x1, y0, y1, label, col, text_offset_y=0.0):
        ax.annotate("", xy=(x1, y1 + 0.005), xytext=(x0, y0 + 0.005),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=1.8, mutation_scale=12))
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        delta  = y1 - y0
        sign   = "+" if delta >= 0 else ""
        ax.text(mid_x, mid_y + 0.055 + text_offset_y,
                f"{sign}{delta*100:.1f} pp\n{label}",
                ha="center", va="bottom", fontsize=7.5, color=col,
                fontweight="semibold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    draw_arrow(0, 1, values[0], values[1], "(data recency)", WONG["green"])
    if e2e_val is not None and e2e_val > 0:
        draw_arrow(1, 2, values[1], values[2], "(E2E fine-tuning)", WONG["sky"])
        draw_arrow(2, 3, values[2], values[3], "(residual gap)", WONG["red"])
    else:
        # E2E optional result omitted: ExpA → HIT-EC direct dashed arrow
        ax.annotate("", xy=(x[3], values[3] + 0.005),
                    xytext=(x[1], values[1] + 0.005),
                    arrowprops=dict(arrowstyle="-|>", color="#8B4513",
                                    lw=1.8, mutation_scale=12,
                                    linestyle="dashed"))
        gap = values[3] - values[1]
        ax.text((x[1]+x[3])/2, (values[1]+values[3])/2 + 0.08,
                f"+{gap*100:.1f} pp\n(architecture gap)",
                ha="center", va="bottom", fontsize=7.5, color="#8B4513",
                fontweight="semibold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # 축 설정
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Temporal Micro F1 (EC-Bench 2023-01)", fontsize=9)
    ax.set_ylim(0.50, 0.97)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.grid(True, which="major", axis="y", zorder=0)

    # 범례
    legend_els = [
        mpatches.Patch(color=WONG["blue"],  label="Contact-EC (ours)"),
        mpatches.Patch(color=WONG["blue"],  hatch="//", label="Contact-EC-ExpA (matched corpus)"),
        mpatches.Patch(color=WONG["sky"],   hatch="xx", label="Contact-EC-E2E (end-to-end ft)"),
        mpatches.Patch(color=WONG["red"],   label="HIT-EC (external temporal reference)"),
    ]
    ax.legend(handles=legend_els, loc="upper left", fontsize=7,
              ncol=1, frameon=True)

    title = "Gap Decomposition: Contact-EC vs. HIT-EC on Temporal Test"
    if e2e_val is None:
        title += " [E2E optional result omitted]"
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)

    fig.tight_layout()
    out_path = PAPER / "gap_decomp.pdf"
    fig.savefig(out_path, format="pdf", dpi=300)
    plt.close(fig)
    print(f"[4] 저장: {out_path}")

    # PNG 백업
    fig2, ax2 = plt.subplots(figsize=(6.0, 3.6))
    plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e_val", type=float, default=None,
                        help="E2E temporal micro F1 (업데이트 시 입력)")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="건너뛸 figure 번호 목록 (예: --skip 3)")
    args = parser.parse_args()

    if "1" not in args.skip:
        plot_learning_curves()
    if "2" not in args.skip:
        plot_level_breakdown()
    if "3" not in args.skip:
        plot_contact_maps()
    if "4" not in args.skip:
        plot_gap_decomp(args.e2e_val)

    print("\n모든 figure 재생성 완료.")
