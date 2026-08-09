"""
Paper figures v3 — journal quality, all issues from review fixed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = "/home/user/Desktop/unlv/paper"

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       12,
    "axes.labelsize":  13,
    "axes.titlesize":  13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":  1.1,
    "figure.dpi":      150,
})

MC = {
    "B1":  "#4878CF",
    "B3":  "#E8812A",
    "CE":  "#2CA02C",
    "HIT": "#D62728",
    "MM":  "#9467BD",
}
CC = {
    "Original": "#2C7BB6",
    "Shuffled": "#D7191C",
    "Diagonal": "#F4A636",
    "Random":   "#7B2D8B",
}


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1  Overview: EC hierarchy | contact map | pipeline
# ═══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 6.0))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40,
                        left=0.02, right=0.98, top=0.95, bottom=0.05)

# ── Panel A  EC hierarchy (top-down, arrows pointing DOWN) ───────────────
ax_a = fig.add_subplot(gs[0])
ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 11)
ax_a.axis("off")

# Panel label outside plot, upper-left
ax_a.text(-0.04, 1.02, "A", transform=ax_a.transAxes,
          fontsize=18, fontweight="bold", va="top")

ax_a.text(5, 10.6, "EC Number Hierarchy", ha="center",
          fontsize=13, fontweight="bold")

# Boxes: top = most general (EC 1.–.–.–), bottom = most specific (EC 1.1.3.4)
# Arrows go DOWNWARD (parent → child)
BOX_W, BOX_H = 8.6, 1.25
levels = [
    # (ec_str, description, level_tag, y_center, bg, border)
    ("EC 1.–.–.–",  "Oxidoreductases\n(7 main classes)",
     "Level 1",  9.0, "#C7DCF0", "#2C5F8A"),
    ("EC 1.1.–.–",  "Acting on C–OH donor\n(~30 sub-classes)",
     "Level 2",  7.1, "#A8D4F5", "#2874A6"),
    ("EC 1.1.3.–",  "With O₂ as acceptor\n(~12 sub-sub-classes)",
     "Level 3",  5.2, "#D6EAF8", "#1A5276"),
    ("EC 1.1.3.4",  "Glucose oxidase\n(specific enzyme)",
     "Level 4",  3.3, "#EBF5FB", "#154360"),
]
for i, (ec, desc, lvl, yc, bg, bc) in enumerate(levels):
    bx = 0.7
    by = yc - BOX_H / 2
    rect = FancyBboxPatch((bx, by), BOX_W, BOX_H,
                          boxstyle="round,pad=0.15",
                          linewidth=1.8, edgecolor=bc, facecolor=bg)
    ax_a.add_patch(rect)
    # EC number — large bold on left
    ax_a.text(bx + 0.3, yc + 0.08, ec, fontsize=11.5, fontweight="bold",
              va="center", color="#111")
    # Description — right side, smaller
    ax_a.text(bx + BOX_W - 0.15, yc + 0.08, desc,
              fontsize=9.5, va="center", ha="right", color="#333",
              linespacing=1.3)
    # Level tag — right outside
    ax_a.text(bx + BOX_W + 0.25, yc + 0.08, lvl,
              fontsize=9, va="center", color=bc, fontstyle="italic")
    # Downward arrow connecting this box to the next (not for last)
    if i < len(levels) - 1:
        ax_a.annotate("",
            xy    =(5.0, yc - BOX_H / 2 - 0.55),   # tip = top of next box
            xytext=(5.0, yc - BOX_H / 2 - 0.02),   # tail = bottom of this box
            arrowprops=dict(arrowstyle="-|>", lw=1.8,
                            color="#555", mutation_scale=18))

# Example label at bottom, clearly separate
ax_a.add_patch(FancyBboxPatch((0.7, 1.55), BOX_W, 0.95,
               boxstyle="round,pad=0.1", linewidth=1.2,
               edgecolor="#888", facecolor="#FFFDE7"))
ax_a.text(5.0, 2.02,
    "Example: Glucose oxidase  (Aspergillus niger,  UniProt P13006)",
    ha="center", va="center", fontsize=9.5, color="#444", fontstyle="italic")


# ── Panel B  Contact map ─────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[1])

# Panel label: inside axes, top-left, white background
ax_b.text(0.03, 0.97, "B", transform=ax_b.transAxes,
          fontsize=18, fontweight="bold", va="top",
          bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

N = 80
np.random.seed(42)
cm = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        d = abs(i - j)
        if d == 0:
            cm[i, j] = 1.0
        elif d < 5:
            cm[i, j] = 0.85 * np.exp(-d / 3)
        elif d < 15:
            cm[i, j] = 0.3 * np.exp(-d / 8)
for (r, c) in [(15,60),(16,61),(17,62),(18,63),(20,55),(21,56),(40,75),(41,76)]:
    if r < N and c < N:
        cm[r, c] = cm[c, r] = 0.75 + 0.2 * np.random.rand()

im = ax_b.imshow(cm, cmap="Blues", vmin=0, vmax=1,
                 origin="upper", interpolation="nearest")
plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04,
             label="Contact probability")
ax_b.set_xlabel("Residue index  j", fontsize=12)
ax_b.set_ylabel("Residue index  i", fontsize=12)
ax_b.set_title("Residue contact map  (8 Å threshold)", fontsize=12, pad=8)

# Annotations with white background boxes
ann_kw = dict(fontsize=10, va="center",
              bbox=dict(boxstyle="round,pad=0.3", fc="white",
                        ec="#555", alpha=0.92, lw=1.0))
ax_b.annotate("Short-range\n|i − j| < 12\n(secondary structure)",
              xy=(6, 6), xytext=(22, 18), **ann_kw,
              arrowprops=dict(arrowstyle="->", color="#1565C0",
                              lw=1.5, relpos=(0, 0.5)))
ax_b.annotate("Long-range contacts\n(tertiary fold topology)",
              xy=(61, 16), xytext=(38, 35), **ann_kw,
              arrowprops=dict(arrowstyle="->", color="#7B2D8B",
                              lw=1.5, relpos=(1, 0.5)))


# ── Panel C  Pipeline with example sequence ──────────────────────────────
ax_c = fig.add_subplot(gs[2])
ax_c.set_xlim(0, 10); ax_c.set_ylim(0, 12)
ax_c.axis("off")
ax_c.text(-0.04, 1.02, "C", transform=ax_c.transAxes,
          fontsize=18, fontweight="bold", va="top")
ax_c.text(5, 11.65, "Contact-EC: sequence–structure fusion",
          ha="center", fontsize=11.5, fontweight="bold")

# Helper: draw a rounded box with text that always fits inside
def draw_box(ax, cx, cy, w, h, txt, fc, ec_col, fontsize=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle="round,pad=0.12",
                                linewidth=1.6, edgecolor=ec_col, facecolor=fc))
    fw = "bold" if bold else "normal"
    ax.text(cx, cy, txt, ha="center", va="center",
            fontsize=fontsize, fontweight=fw,
            color="#111", multialignment="center", linespacing=1.35,
            wrap=False)

# Helper: consistent downward arrow
def arrow_down(ax, x, y1, y2, col="#555"):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.6,
                                color=col, mutation_scale=16))

# Example input sequence (top) — truncated to fit
draw_box(ax_c, 5, 11.05, 9.2, 0.85,
         "Input sequence:  M K T A Y I A K Q R Q I S F V K S H F S R Q L E E R L G …  (1,438 residues)",
         "#F5F5F5", "#888", fontsize=10.5)

# ESM-2 branch (left) and ResNet branch (right)
arrow_down(ax_c, 3.0, 10.6, 9.8)
arrow_down(ax_c, 7.0, 10.6, 9.8)

ax_c.text(2.2, 10.2, "Sequence", ha="center", fontsize=8.5, color="#555")
ax_c.text(7.8, 10.2, "Contact map\n(from AlphaFold)", ha="center",
          fontsize=8.5, color="#555", linespacing=1.3)

draw_box(ax_c, 3.0, 9.25, 4.2, 1.15,
         "ESM-2 650M\n(protein language model)\nfrozen weights",
         "#E8F5E9", "#2E7D32", fontsize=9.5)
draw_box(ax_c, 7.0, 9.25, 4.2, 1.15,
         "ResNet-50\n(2-D conv encoder)\n256 × 256 input",
         "#FFF3E0", "#E65100", fontsize=9.5)

arrow_down(ax_c, 3.0, 8.68, 8.00)
arrow_down(ax_c, 7.0, 8.68, 8.00)

ax_c.text(1.8, 8.35, "1 280-d", ha="center", fontsize=8.5,
          color="#2E7D32", fontweight="bold")
ax_c.text(8.2, 8.35, "512-d", ha="center", fontsize=8.5,
          color="#E65100", fontweight="bold")

draw_box(ax_c, 5.0, 7.45, 8.0, 1.10,
         "Gated cross-attention fusion  (GCA)\n[1 792-d → 512-d]",
         "#F3E5F5", "#6A1B9A", fontsize=9.5)

arrow_down(ax_c, 5.0, 6.90, 6.15)

draw_box(ax_c, 5.0, 5.62, 8.0, 1.05,
         "Fully-connected + Sigmoid\n1 938 EC classes  (multi-label)",
         "#FCE4EC", "#880E4F", fontsize=9.5)

arrow_down(ax_c, 5.0, 5.10, 4.30)

# Output prediction box
draw_box(ax_c, 5.0, 3.78, 8.0, 1.05,
         "Prediction:  EC 1.1.3.4  (0.94)\n"
         "Glucose oxidase  [glucose + O₂ → gluconolactone + H₂O₂]",
         "#FFFDE7", "#F57F17", fontsize=9.5)

# Performance annotation
ax_c.add_patch(FancyBboxPatch((0.4, 1.5), 9.2, 1.85,
               boxstyle="round,pad=0.15", linewidth=1.3,
               edgecolor="#2CA02C", facecolor="#F0FFF0"))
ax_c.text(5, 2.95, "SP-2023-01  (N = 124):", ha="center",
          fontsize=9.5, fontweight="bold", color="#1A5276")
ax_c.text(5, 2.42,
    "Contact-EC  F1 = 0.624   vs   HIT-EC  F1 = 0.847",
    ha="center", fontsize=9.5, color="#333")
ax_c.text(5, 1.92,
    "SP-2024 (N = 1 226):  Contact-EC  F1 = 0.682   vs   HIT-EC  F1 = 0.458",
    ha="center", fontsize=9.2, color="#333")

fig.savefig(f"{OUT}/fig1_overview.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig1_overview.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig1 done")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2  Temporal bar charts — fixed error bars + no confusing green band
# ═══════════════════════════════════════════════════════════════════════════
models = ["MMseqs2\n(top-hit)", "HIT-EC", "Contact-EC\n(fusion)",
          "B3\n(contact)", "B1\n(ESM-2)"]
model_colors = [MC["MM"], MC["HIT"], MC["CE"], MC["B3"], MC["B1"]]

data_23 = dict(means=[0.5852, 0.8471, 0.6241, 0.4244, 0.4508],
               errs =[0,      0,      0.0170, 0.0207, 0.0203])
data_24 = dict(means=[0.7080, 0.4578, 0.6819, 0.4388, 0.3892],
               errs =[0,      0,      0.0026, 0.0166, 0.0121])

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 5.5))
fig.subplots_adjust(left=0.19, right=0.97, wspace=0.38, top=0.87, bottom=0.28)

y = np.arange(len(models))
h = 0.60

def horiz_bars(ax, data, title, panel_ch, show_ylab=True):
    # Bars
    ax.barh(y, data["means"], h,
            color=model_colors, edgecolor="white", linewidth=0.9, zorder=3)
    # Error bars drawn separately for full control
    for i, (v, e) in enumerate(zip(data["means"], data["errs"])):
        if e > 0:
            ax.errorbar(v, i, xerr=e, fmt="none",
                        elinewidth=2.5, ecolor="black",
                        capsize=9, capthick=2.5, zorder=5)
    # Value labels
    for i, (v, e) in enumerate(zip(data["means"], data["errs"])):
        ax.text(v + e + 0.018, i, f"{v:.3f}",
                va="center", ha="left", fontsize=11.5, fontweight="bold",
                color="#111")
    ax.set_xlim(0, 1.15)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.55, len(models) - 0.45)
    ax.set_yticks(y)
    if show_ylab:
        ax.set_yticklabels(models, fontsize=12)
    else:
        ax.set_yticklabels([""] * len(models))
    ax.set_xlabel("Level-4 micro F1", fontsize=13, labelpad=5)
    ax.set_title(title, fontsize=12.5, pad=10, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    # Bold green border around Contact-EC bar to mark it as the main model
    ce_idx = 2
    ax.barh(ce_idx, data["means"][ce_idx], h,
            color="none", edgecolor=MC["CE"], linewidth=2.8, zorder=4)

horiz_bars(ax_a, data_23, "SP-2023-01 temporal holdout  (N = 124)",
           "A", show_ylab=True)
horiz_bars(ax_b, data_24, "SP-2024 temporal holdout  (N = 1,226)",
           "B", show_ylab=False)

# Panel letters placed clearly ABOVE each subplot, left-aligned
ax_a.text(-0.14, 1.06, "A", transform=ax_a.transAxes,
          fontsize=20, fontweight="bold", va="top")
ax_b.text(-0.06, 1.06, "B", transform=ax_b.transAxes,
          fontsize=20, fontweight="bold", va="top")

# Shared colour legend (needed since panel B has no y labels)
legend_patches = [
    mpatches.Patch(color=MC["MM"],  label="MMseqs2 (top-hit)"),
    mpatches.Patch(color=MC["HIT"], label="HIT-EC"),
    mpatches.Patch(color=MC["CE"],  label="Contact-EC (fusion)  ← main model"),
    mpatches.Patch(color=MC["B3"],  label="B3 (contact-map only)"),
    mpatches.Patch(color=MC["B1"],  label="B1 (ESM-2 only)"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           fontsize=11, framealpha=0.95, bbox_to_anchor=(0.58, 0.01),
           borderpad=0.8, handlelength=1.5)
fig.text(0.58, -0.04,
    "Error bars: ±1 s.d. across three random seeds (42/43/44).  "
    "HIT-EC and MMseqs2 have no seed variance (deterministic).",
    ha="center", fontsize=10, color="#555", style="italic")

fig.savefig(f"{OUT}/fig2_temporal_bars.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig2_temporal_bars.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig2 done")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3  Reversal slope chart — legend moved to lower-left (avoids 0.847)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9.0, 5.8))
fig.subplots_adjust(left=0.10, right=0.68, top=0.90, bottom=0.13)

traces = [
    ("MMseqs2",      0.5852, 0.7080, MC["MM"],  1.8, ":",  "D"),
    ("HIT-EC",       0.8471, 0.4578, MC["HIT"], 3.0, "-",  "s"),
    ("Contact-EC",   0.6241, 0.6819, MC["CE"],  3.0, "-",  "o"),
    ("B3 (contact)", 0.4244, 0.4388, MC["B3"],  1.8, "--", "^"),
    ("B1 (ESM-2)",   0.4508, 0.3892, MC["B1"],  1.8, "--", "v"),
]

for (label, v23, v24, col, lw, ls, mk) in traces:
    ax.plot([0, 1], [v23, v24], color=col, linewidth=lw, linestyle=ls,
            marker=mk, markersize=10, markerfacecolor=col,
            markeredgecolor="white", markeredgewidth=1.5, zorder=4,
            solid_capstyle="round")

# ── Left labels with generous spreading ──
def spread_labels(vals, min_gap=0.032):
    pos = list(vals)
    for _ in range(400):
        for i in range(1, len(pos)):
            if pos[i-1] - pos[i] < min_gap:
                mid = (pos[i-1] + pos[i]) / 2
                pos[i-1] = mid + min_gap / 2
                pos[i]   = mid - min_gap / 2
    return pos

left_data  = sorted(traces, key=lambda t: t[1], reverse=True)
left_pos   = spread_labels([t[1] for t in left_data])
right_data = sorted(traces, key=lambda t: t[2], reverse=True)
right_pos  = spread_labels([t[2] for t in right_data])

for i, (label, v23, v24, col, lw, ls, mk) in enumerate(left_data):
    ax.text(-0.06, left_pos[i], f"{v23:.3f}",
            ha="right", va="center", fontsize=11, color=col,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="none", alpha=0.85))

for i, (label, v23, v24, col, lw, ls, mk) in enumerate(right_data):
    delta = v24 - v23
    sign  = "+" if delta >= 0 else "−"
    ax.text(1.06, right_pos[i],
            f"{v24:.3f}  ({sign}{abs(delta)*100:.1f} pp)",
            ha="left", va="center", fontsize=11, color=col,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="none", alpha=0.85))

# Reversal region
ax.fill_between([0.80, 1.20],
                [0.4578] * 2, [0.6819] * 2,
                color=MC["CE"], alpha=0.09)
ax.text(1.0, 0.570, "+22.4 pp\nreversal", ha="center", va="center",
        fontsize=10, color=MC["CE"], fontweight="bold",
        bbox=dict(fc="white", ec=MC["CE"],
                  boxstyle="round,pad=0.3", lw=1.2, alpha=0.9))

ax.set_xticks([0, 1])
ax.set_xticklabels(["SP-2023-01\n(N = 124)", "SP-2024\n(N = 1,226)"],
                   fontsize=13)
ax.set_xlim(-0.32, 1.30)
ax.set_ylim(0.27, 0.98)
ax.set_ylabel("Level-4 micro F1", fontsize=13, labelpad=6)
ax.set_title("Model trajectories across temporal horizons",
             fontsize=13, pad=10)
ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# Legend at LOWER-LEFT — clear of the 0.847 HIT-EC point in upper-left
legend_elements = [
    plt.Line2D([0],[0], color=MC["CE"],  lw=3.0, ls="-",  marker="o",
               markersize=9, label="Contact-EC (fusion)"),
    plt.Line2D([0],[0], color=MC["HIT"], lw=3.0, ls="-",  marker="s",
               markersize=9, label="HIT-EC"),
    plt.Line2D([0],[0], color=MC["MM"],  lw=1.8, ls=":",  marker="D",
               markersize=9, label="MMseqs2 (top-hit)"),
    plt.Line2D([0],[0], color=MC["B3"],  lw=1.8, ls="--", marker="^",
               markersize=9, label="B3 (contact only)"),
    plt.Line2D([0],[0], color=MC["B1"],  lw=1.8, ls="--", marker="v",
               markersize=9, label="B1 (ESM-2 only)"),
]
ax.legend(handles=legend_elements, loc="lower left",
          framealpha=0.94, fontsize=10.5, borderpad=0.8)

fig.savefig(f"{OUT}/fig3_reversal_slope.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig3_reversal_slope.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig3 done")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4  Perturbation (unchanged — already good)
# ═══════════════════════════════════════════════════════════════════════════
conditions = ["Original", "Shuffled\n(row/col)", "Diagonal-only\n(|i−j|<12)",
              "Density-matched\nrandom"]
cond_cols  = [CC["Original"], CC["Shuffled"], CC["Diagonal"], CC["Random"]]
cond_short = ["Original", "Shuffled", "Diagonal", "Random"]

panel_data = [
    ("A   B3  (contact-map only)",
     [0.3646, 0.0000, 0.0000, 0.0000], MC["B3"]),
    ("B   Contact-EC  (flat FC)",
     [0.6032, 0.0625, 0.5242, 0.0982], MC["CE"]),
    ("C   Contact-EC-Hier",
     [0.5690, 0.5702, 0.5776, 0.5641], "#6A5ACD"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), sharey=True)
fig.subplots_adjust(wspace=0.05, top=0.88, bottom=0.25)

x = np.arange(len(conditions))
for ax, (title, vals, mc_col) in zip(axes, panel_data):
    ax.bar(x, vals, width=0.6, color=cond_cols,
           edgecolor="white", linewidth=0.9, zorder=3)
    for xi, v in zip(x, vals):
        if v >= 0.02:
            ax.text(xi, v + 0.016, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=12,
                    fontweight="bold", color="#111")
        else:
            ax.text(xi, 0.022, "0.000",
                    ha="center", va="bottom", fontsize=12,
                    fontweight="bold", color="#999")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_short, fontsize=11.5)
    ax.set_ylim(0, 0.83)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=12, pad=8, fontweight="bold")

axes[0].set_ylabel("Level-4 micro F1", fontsize=13)

legend_patches = [
    mpatches.Patch(color=CC["Original"], label="Original contact map"),
    mpatches.Patch(color=CC["Shuffled"], label="Shuffled (topology destroyed)"),
    mpatches.Patch(color=CC["Diagonal"], label="Diagonal-only (secondary structure)"),
    mpatches.Patch(color=CC["Random"],   label="Density-matched random"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=4,
           fontsize=11, framealpha=0.95, bbox_to_anchor=(0.5, -0.06),
           borderpad=0.8)
fig.suptitle("Contact-map perturbation controls  (SP-2023-01, N = 124)",
             fontsize=13, fontweight="bold", y=1.0)

fig.savefig(f"{OUT}/fig4_perturbation.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig4_perturbation.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig4 done")

print("\nAll figures saved to", OUT)
