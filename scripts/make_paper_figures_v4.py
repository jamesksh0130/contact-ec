"""Paper figures v4 — full redesign."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

OUT = "/home/user/Desktop/unlv/paper"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "figure.dpi": 150,
})

MC = {"Seq": "#4878CF", "Str": "#E8812A", "CE": "#2CA02C",
      "HIT": "#D62728", "MM": "#9467BD"}
CC = {"Original": "#2C7BB6", "Permuted": "#D7191C",
      "Secondary": "#F4A636", "Random": "#7B2D8B"}
AA_COL = {
    'M': '#B8B8B8', 'K': '#6495ED', 'T': '#6BC975', 'A': '#C8C8C8',
    'Y': '#FFD700', 'I': '#B8B8B8', 'Q': '#6BC975', 'R': '#6495ED',
    'S': '#6BC975', 'F': '#C8C880', 'V': '#B8B8B8', 'N': '#6BC975',
    'G': '#FFD700', 'E': '#FF8888', 'D': '#FF8888', 'H': '#6495ED',
    'L': '#B8B8B8', 'W': '#C8C880', 'P': '#C8A0C8', 'C': '#FFD700',
}


def draw_box(ax, cx, cy, bw, bh, fc, ec, lw=2.4):
    ax.add_patch(FancyBboxPatch(
        (cx - bw / 2, cy - bh / 2), bw, bh,
        boxstyle="round,pad=0.15", lw=lw, edgecolor=ec, facecolor=fc))


def arrow(ax, x1, y1, x2, y2, col="#666", lw=2.4, ms=22, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=col,
                                mutation_scale=ms,
                                connectionstyle=f"arc3,rad={rad}"))


def T(ax, x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("fontsize", 14)
    kw.setdefault("color", "#111")
    ax.text(x, y, s, **kw)


def SH(ax, x, y, s):
    ax.text(x, y, s, ha="center", fontsize=17, color="#999",
            fontstyle="italic", fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 36))
gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.9, 2.0],
                       hspace=0.05, left=0.04, right=0.96,
                       top=0.97, bottom=0.01)
gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],
                                          wspace=0.30, width_ratios=[1, 1.1])
ax_a = fig.add_subplot(gs_top[0])
ax_b = fig.add_subplot(gs_top[1])
ax_c = fig.add_subplot(gs[1])

# ── Panel A ───────────────────────────────────────────────────────────────
ax_a.set_xlim(0, 10); ax_a.set_ylim(0, 15); ax_a.axis("off")
ax_a.text(-0.06, 1.04, "A", transform=ax_a.transAxes,
          fontsize=30, fontweight="bold", va="top")
ax_a.text(5.0, 14.5, "Enzyme Commission (EC) Classification",
          ha="center", fontsize=22, fontweight="bold")

BOX_W = 9.0; BOX_H = 2.4; GAP = 0.85
step = BOX_H + GAP
yc = [13.0 - i * step for i in range(4)]

levels = [
    ("EC  1.–.–.–", "Oxidoreductases",
     "Level 1  ·  7 classes", "#C7DCF0", "#1D4ED8"),
    ("EC  1.1.–.–", "CH–OH group donors",
     "Level 2  ·  ~30 sub-classes", "#A8D4F5", "#1E40AF"),
    ("EC  1.1.3.–", "With O₂ as acceptor",
     "Level 3  ·  ~12 sub-sub-classes", "#D6EAF8", "#0369A1"),
    ("EC  1.1.3.4", "Glucose oxidase",
     "Level 4  ·  specific enzyme", "#EBF5FB", "#0C4A6E"),
]
for i, (ec, desc, lvl, bg, bc) in enumerate(levels):
    ax_a.add_patch(FancyBboxPatch((0.5, yc[i] - BOX_H / 2), BOX_W, BOX_H,
                                  boxstyle="round,pad=0.18", lw=2.4,
                                  edgecolor=bc, facecolor=bg))
    ax_a.text(0.95, yc[i] + 0.50, ec,
              fontsize=32, fontweight="bold", va="center", color="#111")
    ax_a.text(0.95, yc[i] - 0.50, desc,
              fontsize=24, va="center", color="#333")
    ax_a.text(9.3, yc[i] + 0.50, lvl,
              fontsize=16, va="center", ha="right", color=bc, fontstyle="italic")
    if i < 3:
        ax_a.annotate("",
                      xy=(5, yc[i + 1] + BOX_H / 2 + 0.08),
                      xytext=(5, yc[i] - BOX_H / 2 - 0.08),
                      arrowprops=dict(arrowstyle="-|>", lw=2.8,
                                     color="#667", mutation_scale=28))

# ── Panel B ───────────────────────────────────────────────────────────────
N = 80; np.random.seed(42)
cm_arr = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        d = abs(i - j)
        if d == 0:   cm_arr[i, j] = 1.0
        elif d < 5:  cm_arr[i, j] = 0.85 * np.exp(-d / 3)
        elif d < 15: cm_arr[i, j] = 0.30 * np.exp(-d / 8)
for r, c in [(15,60),(16,61),(17,62),(18,63),(20,55),(40,75),(41,76)]:
    if r < N and c < N:
        cm_arr[r, c] = cm_arr[c, r] = 0.75 + 0.2 * np.random.rand()

im = ax_b.imshow(cm_arr, cmap="Blues", vmin=0, vmax=1,
                 origin="upper", interpolation="nearest")
cbar = plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
cbar.set_label("Contact probability", fontsize=14)
cbar.ax.tick_params(labelsize=13)
ax_b.set_xlabel("Residue index  j", fontsize=16)
ax_b.set_ylabel("Residue index  i", fontsize=16)
ax_b.tick_params(labelsize=14)

ax_b.text(-0.06, 1.04, "B", transform=ax_b.transAxes,
          fontsize=30, fontweight="bold", va="top")
ax_b.text(0.5, 1.02, "Residue–residue contact map  (8 Å threshold)",
          transform=ax_b.transAxes, ha="center", va="bottom",
          fontsize=20, fontweight="bold")

ax_b.annotate("Short-range\ncontacts",
              xy=(5, 5), xytext=(26, 18),
              fontsize=17, color="#1565C0", fontweight="bold",
              arrowprops=dict(arrowstyle="->", lw=2.0, color="#1565C0",
                              connectionstyle="arc3,rad=0.25"))
ax_b.annotate("Long-range\ncontacts",
              xy=(60, 16), xytext=(42, 4),
              fontsize=17, color="#7B2D8B", fontweight="bold", ha="center",
              arrowprops=dict(arrowstyle="->", lw=2.0, color="#7B2D8B",
                              connectionstyle="arc3,rad=-0.25"))

# ── Panel C: 6-stage vertical pipeline ───────────────────────────────────
ax_c.set_xlim(0, 18); ax_c.set_ylim(0, 30); ax_c.axis("off")
ax_c.text(-0.02, 1.03, "C", transform=ax_c.transAxes,
          fontsize=30, fontweight="bold", va="top")
ax_c.text(9, 29.3, "Contact-EC: Sequence–Structure Fusion Pipeline",
          ha="center", fontsize=23, fontweight="bold")

BW_SIDE  = 7.5   # branch box width
BH       = 2.80  # standard box height
BW_CTR   = 11.0  # center box width

# Stage y-centers — generous spacing so section headers fit in gaps
Y_IN   = 26.5   # INPUT
Y_PRE  = 21.5   # PREPROCESSING
Y_ENC  = 16.5   # ENCODING
Y_FUS  = 11.5   # FUSION
Y_CLS  =  7.0   # CLASSIFICATION
Y_OUT  =  2.8   # OUTPUT

# X-centers for branches and center
XL = 4.5; XR = 13.5; XC = 9.0

# ── INPUTS ────────────────────────────────────────────────────────────────
SH(ax_c, XC, 28.5, "INPUTS")

T(ax_c, XL, Y_IN + 0.7, "Protein Sequence",
  fontsize=23, fontweight="bold", color="#1565C0")
seq = "MKTAYIAKQR"; bw_aa = 0.52
x0_aa = XL - len(seq) * bw_aa / 2
for idx, aa in enumerate(seq):
    xb = x0_aa + idx * bw_aa
    ax_c.add_patch(Rectangle((xb, Y_IN - 0.22), bw_aa - 0.04, 0.55,
                              fc=AA_COL.get(aa, "#CCC"), ec="white", lw=1.2))
    ax_c.text(xb + (bw_aa - 0.04) / 2, Y_IN + 0.06, aa,
              ha="center", va="center", fontsize=10, fontweight="bold", color="#111")
ax_c.text(x0_aa + len(seq) * bw_aa + 0.08, Y_IN + 0.06, "…",
          ha="left", va="center", fontsize=18, color="#888")
T(ax_c, XL, Y_IN - 0.8, "1,438 amino acid residues",
  fontsize=19, color="#555")

T(ax_c, XR, Y_IN + 0.6, "Protein 3D Structure",
  fontsize=23, fontweight="bold", color="#E65100")
T(ax_c, XR, Y_IN - 0.05, "UniProt ID → AlphaFold Database",
  fontsize=19, color="#555")
T(ax_c, XR, Y_IN - 0.82, "Cα atomic coordinates (PDB)",
  fontsize=17, color="#888")

arrow(ax_c, XL, Y_IN - BH/2, XL, Y_PRE + BH/2, col="#1565C0")
arrow(ax_c, XR, Y_IN - BH/2, XR, Y_PRE + BH/2, col="#E65100")

# ── PREPROCESSING ─────────────────────────────────────────────────────────
SH(ax_c, XC, Y_IN - BH/2 - 0.70, "PREPROCESSING")

T(ax_c, XL, Y_PRE + 0.6, "Tokenization",
  fontsize=23, fontweight="bold", color="#1E40AF")
T(ax_c, XL, Y_PRE - 0.25, "BPE encoding",
  fontsize=19, color="#555")
T(ax_c, XL, Y_PRE - 0.92, "Max length: 1,024 tokens",
  fontsize=17, color="#888")

T(ax_c, XR, Y_PRE + 0.6, "Contact Map Extraction",
  fontsize=23, fontweight="bold", color="#C2410C")
T(ax_c, XR, Y_PRE - 0.15, "8 Å threshold  ·  256 × 256",
  fontsize=19, color="#555")
T(ax_c, XR, Y_PRE - 0.85, "Cα pairwise distance matrix",
  fontsize=17, color="#888")

arrow(ax_c, XL, Y_PRE - BH/2, XL, Y_ENC + BH/2, col="#1E40AF")
arrow(ax_c, XR, Y_PRE - BH/2, XR, Y_ENC + BH/2, col="#C2410C")

# ── ENCODING ──────────────────────────────────────────────────────────────
SH(ax_c, XC, Y_PRE - BH/2 - 0.70, "ENCODING")

T(ax_c, XL, Y_ENC + 0.6, "ESM-2 650M",
  fontsize=23, fontweight="bold", color="#1D4ED8")
T(ax_c, XL, Y_ENC - 0.1, "Protein language model  (frozen)",
  fontsize=19, color="#555")
T(ax_c, XL, Y_ENC - 0.85, "→  1,280-d sequence embedding",
  fontsize=19, fontweight="bold", color="#1D4ED8")

T(ax_c, XR, Y_ENC + 0.6, "ResNet-50",
  fontsize=23, fontweight="bold", color="#EA580C")
T(ax_c, XR, Y_ENC - 0.1, "Convolutional encoder  (trainable)",
  fontsize=19, color="#555")
T(ax_c, XR, Y_ENC - 0.85, "→  512-d structural feature",
  fontsize=19, fontweight="bold", color="#EA580C")

# Converging arrows to fusion
arrow(ax_c, XL, Y_ENC - BH/2, 7.0, Y_FUS + BH/2,
      col="#1D4ED8", rad=-0.18)
arrow(ax_c, XR, Y_ENC - BH/2, 11.0, Y_FUS + BH/2,
      col="#EA580C", rad=0.18)

# ── FUSION ────────────────────────────────────────────────────────────────
SH(ax_c, XC, Y_ENC - BH/2 - 0.70, "FUSION")

T(ax_c, XC, Y_FUS + 0.6, "Gated Cross-Attention Fusion",
  fontsize=23, fontweight="bold", color="#065F46")
T(ax_c, XC, Y_FUS - 0.1, "Structure features gate sequence attention",
  fontsize=19, color="#444")
T(ax_c, XC, Y_FUS - 0.85, "1,792-d  →  512-d fused representation",
  fontsize=19, fontweight="bold", color="#065F46")

arrow(ax_c, XC, Y_FUS - BH/2, XC, Y_CLS + BH/2, col="#444")

# ── CLASSIFICATION ────────────────────────────────────────────────────────
SH(ax_c, XC, Y_FUS - BH/2 - 0.70, "CLASSIFICATION HEAD")

T(ax_c, XC, Y_CLS + 0.6, "FC Head  +  Sigmoid",
  fontsize=23, fontweight="bold", color="#3730A3")
T(ax_c, XC, Y_CLS - 0.1, "Multi-label binary classification",
  fontsize=19, color="#444")
T(ax_c, XC, Y_CLS - 0.85, "1,938 EC class labels",
  fontsize=19, fontweight="bold", color="#3730A3")

arrow(ax_c, XC, Y_CLS - BH/2, XC, Y_OUT + BH/2,
      col="#166534", lw=2.6)

# ── OUTPUT ────────────────────────────────────────────────────────────────
SH(ax_c, XC, Y_CLS - BH/2 - 0.70, "PREDICTED ENZYME FUNCTION")

T(ax_c, XC, Y_OUT + 0.6, "EC 1.1.3.4  —  Glucose oxidase",
  fontsize=23, fontweight="bold", color="#166534")
T(ax_c, XC, Y_OUT - 0.1, "Oxidoreductase  ·  CH–OH donors  ·  O₂ as acceptor",
  fontsize=19, color="#333")
T(ax_c, XC, Y_OUT - 0.82, "Confidence: p = 0.94   (Aspergillus niger,  UniProt P13006)",
  fontsize=17, color="#666", fontstyle="italic")

fig.savefig(f"{OUT}/fig1_overview.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig1_overview.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig1 done")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2
# ═══════════════════════════════════════════════════════════════════════════
MODELS = ["MMseqs2", "HIT-EC", "Contact-EC", "Structure-only", "Sequence-only"]
M_COLS = [MC["MM"], MC["HIT"], MC["CE"], MC["Str"], MC["Seq"]]
d23 = dict(means=[0.5852, 0.8471, 0.6241, 0.4244, 0.4508],
           errs=[0, 0, 0.0170, 0.0207, 0.0203])
d24 = dict(means=[0.7080, 0.4578, 0.6819, 0.4388, 0.3892],
           errs=[0, 0, 0.0026, 0.0166, 0.0121])

fig, (ax_a2, ax_b2) = plt.subplots(1, 2, figsize=(19, 7.0))
fig.subplots_adjust(left=0.17, right=0.97, wspace=0.44, top=0.87, bottom=0.20)

y = np.arange(len(MODELS)); h = 0.60


def horiz_bars(ax, data, title, panel_ch):
    ax.barh(y, data["means"], h, color=M_COLS,
            edgecolor="white", linewidth=0.9, zorder=3)
    for i, (v, e) in enumerate(zip(data["means"], data["errs"])):
        if e > 0:
            ax.errorbar(v, i, xerr=e, fmt="none",
                        elinewidth=2.4, ecolor="#111",
                        capsize=8, capthick=2.4, zorder=5)
        ax.text(v + max(e, 0) + 0.018, i, f"{v:.3f}",
                va="center", ha="left", fontsize=15, fontweight="bold", color="#111")
    ax.set_xlim(0, 1.20)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=14)
    ax.set_ylim(-0.55, len(MODELS) - 0.45)
    ax.set_yticks(y)
    ax.set_yticklabels(MODELS, fontsize=16)
    ax.set_xlabel("Level-4 micro F1", fontsize=16, labelpad=5)
    ax.set_title(title, fontsize=14, pad=10, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.barh(2, data["means"][2], h, color="none",
            edgecolor=MC["CE"], linewidth=2.8, zorder=4)
    ax.text(-0.14, 1.06, panel_ch, transform=ax.transAxes,
            fontsize=24, fontweight="bold", va="top")


horiz_bars(ax_a2, d23, "A   SP-2023-01 temporal holdout  (N = 124)", "A")
horiz_bars(ax_b2, d24, "B   SP-2024 temporal holdout  (N = 1,226)", "B")

legend_patches = [
    mpatches.Patch(color=MC["MM"],  label="MMseqs2"),
    mpatches.Patch(color=MC["HIT"], label="HIT-EC"),
    mpatches.Patch(color=MC["CE"],  label="Contact-EC  (ours)"),
    mpatches.Patch(color=MC["Str"], label="Structure-only"),
    mpatches.Patch(color=MC["Seq"], label="Sequence-only"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=5,
           fontsize=13, frameon=False, bbox_to_anchor=(0.57, -0.01))

fig.savefig(f"{OUT}/fig2_temporal_bars.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig2_temporal_bars.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig2 done")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 — no green shading, no annotation text
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9.5, 6.0))
fig.subplots_adjust(left=0.10, right=0.68, top=0.90, bottom=0.28)

traces = [
    ("MMseqs2",        0.5852, 0.7080, MC["MM"],  1.8, ":",  "D"),
    ("HIT-EC",         0.8471, 0.4578, MC["HIT"], 3.0, "-",  "s"),
    ("Contact-EC",     0.6241, 0.6819, MC["CE"],  3.0, "-",  "o"),
    ("Structure-only", 0.4244, 0.4388, MC["Str"], 1.8, "--", "^"),
    ("Sequence-only",  0.4508, 0.3892, MC["Seq"], 1.8, "--", "v"),
]

for (lbl, v23, v24, col, lw, ls, mk) in traces:
    ax.plot([0, 1], [v23, v24], color=col, linewidth=lw, linestyle=ls,
            marker=mk, markersize=10, markerfacecolor=col,
            markeredgecolor="white", markeredgewidth=1.5, zorder=4,
            solid_capstyle="round")


def spread(vals, gap=0.034):
    p = list(vals)
    for _ in range(500):
        for i in range(1, len(p)):
            if p[i - 1] - p[i] < gap:
                mid = (p[i - 1] + p[i]) / 2
                p[i - 1] = mid + gap / 2; p[i] = mid - gap / 2
    return p


L = sorted(traces, key=lambda t: t[1], reverse=True)
R = sorted(traces, key=lambda t: t[2], reverse=True)
lp = spread([t[1] for t in L])
rp = spread([t[2] for t in R])

for i, (lbl, v23, v24, col, lw, ls, mk) in enumerate(L):
    ax.text(-0.06, lp[i], f"{v23:.3f}", ha="right", va="center",
            fontsize=12, color=col, fontweight="bold")
for i, (lbl, v23, v24, col, lw, ls, mk) in enumerate(R):
    delta = v24 - v23
    sign = "+" if delta >= 0 else "−"
    ax.text(1.06, rp[i],
            f"{v24:.3f}  ({sign}{abs(delta) * 100:.1f} pp)",
            ha="left", va="center", fontsize=12, color=col, fontweight="bold")

ax.set_xticks([0, 1])
ax.set_xticklabels(["SP-2023-01\n(N = 124)", "SP-2024\n(N = 1,226)"], fontsize=14)
ax.set_xlim(-0.32, 1.30)
ax.set_ylim(0.27, 0.98)
ax.set_ylabel("Level-4 micro F1", fontsize=15, labelpad=6)
ax.set_title("Model trajectories across temporal horizons", fontsize=15, pad=10)
ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

legend_elements = [
    plt.Line2D([0], [0], color=MC["CE"],  lw=3.0, ls="-",  marker="o",
               markersize=9, label="Contact-EC"),
    plt.Line2D([0], [0], color=MC["HIT"], lw=3.0, ls="-",  marker="s",
               markersize=9, label="HIT-EC"),
    plt.Line2D([0], [0], color=MC["MM"],  lw=1.8, ls=":",  marker="D",
               markersize=9, label="MMseqs2"),
    plt.Line2D([0], [0], color=MC["Str"], lw=1.8, ls="--", marker="^",
               markersize=9, label="Structure-only"),
    plt.Line2D([0], [0], color=MC["Seq"], lw=1.8, ls="--", marker="v",
               markersize=9, label="Sequence-only"),
]
ax.legend(handles=legend_elements,
          loc="upper center", bbox_to_anchor=(0.5, -0.20),
          ncol=5, frameon=False, fontsize=12)

fig.savefig(f"{OUT}/fig3_reversal_slope.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig3_reversal_slope.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig3 done")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 — tight legend below bars
# ═══════════════════════════════════════════════════════════════════════════
cond_labels = ["Original", "Permuted", "Secondary\nstructure", "Random"]
cond_cols = [CC["Original"], CC["Permuted"], CC["Secondary"], CC["Random"]]

panels = [
    ("A   Structure-only",      [0.3646, 0.0000, 0.0000, 0.0000]),
    ("B   Contact-EC",          [0.6032, 0.0625, 0.5242, 0.0982]),
    ("C   Contact-EC  (gated)", [0.5690, 0.5702, 0.5776, 0.5641]),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
fig.subplots_adjust(wspace=0.06, top=0.87, bottom=0.30, left=0.08, right=0.98)

x = np.arange(4)
for ax, (title, vals) in zip(axes, panels):
    ax.bar(x, vals, width=0.62, color=cond_cols,
           edgecolor="white", linewidth=0.9, zorder=3)
    ax.axhline(y=0, color="black", lw=1.8, zorder=6)
    for xi, v in zip(x, vals):
        if v >= 0.015:
            ax.text(xi, v + 0.016, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=13,
                    fontweight="bold", color="#111")
        else:
            ax.text(xi, 0.022, "0.000",
                    ha="center", va="bottom", fontsize=13,
                    fontweight="bold", color="#999")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=13)
    ax.set_ylim(0, 0.84)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=14, pad=8, fontweight="bold")

axes[0].set_ylabel("Level-4 micro F1", fontsize=15)

legend_patches = [
    mpatches.Patch(color=CC["Original"],  label="Original contact map"),
    mpatches.Patch(color=CC["Permuted"],  label="Row/column permutation"),
    mpatches.Patch(color=CC["Secondary"], label="Secondary structure only  (|i−j| < 12)"),
    mpatches.Patch(color=CC["Random"],    label="Density-matched random"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=4,
           fontsize=12.5, frameon=False, bbox_to_anchor=(0.53, 0.01))
fig.suptitle("Contact-map perturbation controls  (SP-2023-01, N = 124)",
             fontsize=15, fontweight="bold", y=1.0)

fig.savefig(f"{OUT}/fig4_perturbation.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig4_perturbation.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig4 done")

print(f"\nAll figures → {OUT}")
