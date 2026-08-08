"""Paper figures v5 — print-ready for Bioinformatics (170 mm double column, 300 DPI).

All font sizes are set as final print sizes (pt).
"""
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
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "figure.dpi": 300,
})

AA_COL = {
    'M': '#B8B8B8', 'K': '#6495ED', 'T': '#6BC975', 'A': '#C8C8C8',
    'Y': '#FFD700', 'I': '#B8B8B8', 'Q': '#6BC975', 'R': '#6495ED',
    'S': '#6BC975', 'F': '#C8C880', 'V': '#B8B8B8', 'N': '#6BC975',
    'G': '#FFD700', 'E': '#FF8888', 'D': '#FF8888', 'H': '#6495ED',
    'L': '#B8B8B8', 'W': '#C8C880', 'P': '#C8A0C8', 'C': '#FFD700',
}
MC = {"Seq": "#4878CF", "Str": "#E8812A", "CE": "#2CA02C",
      "HIT": "#D62728", "MM": "#9467BD"}
CC = {"Original": "#2C7BB6", "Permuted": "#D7191C",
      "Secondary": "#F4A636", "Random": "#7B2D8B"}


def arr(ax, x1, y1, x2, y2, col="#555", lw=0.9, ms=8, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=col,
                                mutation_scale=ms,
                                connectionstyle=f"arc3,rad={rad}"))


def T(ax, x, y, s, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center")
    kw.setdefault("fontsize", 7); kw.setdefault("color", "#111")
    ax.text(x, y, s, **kw)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1  (print-ready: 6.7 × 12.0 inches = 170 × 305 mm at 300 DPI)
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(6.7, 12.0))
gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.9, 2.0],
                       hspace=0.14, left=0.05, right=0.95,
                       top=0.97, bottom=0.01)
gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],
                                          wspace=0.32, width_ratios=[1, 1.1])
ax_a = fig.add_subplot(gs_top[0])
ax_b = fig.add_subplot(gs_top[1])
ax_c = fig.add_subplot(gs[1])

# ── Panel A ───────────────────────────────────────────────────────────────
ax_a.set_xlim(0, 10); ax_a.set_ylim(2.5, 14.8); ax_a.axis("off")
ax_a.text(-0.06, 1.15, "A", transform=ax_a.transAxes,
          fontsize=11, fontweight="bold", va="top")
ax_a.text(0.5, 1.02, "Enzyme Commission (EC) Classification",
          transform=ax_a.transAxes, ha="center", va="bottom",
          fontsize=9, fontweight="bold")

BOX_W = 9.0; BOX_H = 2.1; GAP = 0.70
yc = [13.0 - i * (BOX_H + GAP) for i in range(4)]
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
                                  boxstyle="round,pad=0.18", lw=0.9,
                                  edgecolor=bc, facecolor=bg))
    ax_a.text(0.85, yc[i] + 0.38, ec,
              fontsize=11, fontweight="bold", va="center", color="#111")
    ax_a.text(0.85, yc[i] - 0.40, desc,
              fontsize=8, va="center", color="#333")
    ax_a.text(9.3, yc[i] + 0.38, lvl,
              fontsize=5.5, va="center", ha="right", color=bc, fontstyle="italic")
    if i < 3:
        ax_a.annotate("",
                      xy=(5, yc[i + 1] + BOX_H / 2 + 0.08),
                      xytext=(5, yc[i] - BOX_H / 2 - 0.08),
                      arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                     color="#667", mutation_scale=10))

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
                 origin="upper", interpolation="nearest", aspect="auto")
cbar = plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
cbar.set_label("Contact probability", fontsize=6)
cbar.ax.tick_params(labelsize=5.5)
ax_b.set_xlabel("Residue index  j", fontsize=7)
ax_b.set_ylabel("Residue index  i", fontsize=7)
ax_b.tick_params(labelsize=6)

ax_b.text(-0.06, 1.15, "B", transform=ax_b.transAxes,
          fontsize=11, fontweight="bold", va="top")
ax_b.text(0.5, 1.02, "Residue–residue contact map  (8 Å threshold)",
          transform=ax_b.transAxes, ha="center", va="bottom",
          fontsize=9, fontweight="bold")

ax_b.annotate("Short-range\ncontacts",
              xy=(5, 5), xytext=(24, 16),
              fontsize=6, color="#1565C0", fontweight="bold",
              arrowprops=dict(arrowstyle="->", lw=0.8, color="#1565C0",
                              connectionstyle="arc3,rad=0.25"))
ax_b.annotate("Long-range\ncontacts",
              xy=(60, 16), xytext=(42, 4),
              fontsize=6, color="#7B2D8B", fontweight="bold", ha="center",
              arrowprops=dict(arrowstyle="->", lw=0.8, color="#7B2D8B",
                              connectionstyle="arc3,rad=-0.25"))

# ── Panel C ───────────────────────────────────────────────────────────────
ax_c.set_xlim(0, 18); ax_c.set_ylim(0, 30); ax_c.axis("off")
ax_c.text(-0.02, 1.03, "C", transform=ax_c.transAxes,
          fontsize=11, fontweight="bold", va="top")
ax_c.text(9, 29.3, "Contact-EC: Sequence–Structure Fusion Pipeline",
          ha="center", fontsize=9, fontweight="bold")

BH_H = 1.7; BH_SEC = BH_H * 2; BW_FULL = 17.0; BW_CTR = 12.0
Y_IN = 26.0; Y_PRE = 21.2; Y_ENC = 16.4
Y_FUS = 11.5; Y_CLS = 7.0; Y_OUT = 2.8
XL = 4.5; XR = 13.5; XC = 9.0

DARK  = "#0F172A"; MID = "#555555"; LIGHT = "#888888"
ECBOX = "#374151"

def sec_box(cy, bw=BW_FULL, cx=XC, fc="white", ec=ECBOX, label=None):
    ax_c.add_patch(FancyBboxPatch((cx - bw/2, cy - BH_H), bw, BH_SEC,
                                   boxstyle="round,pad=0.1", lw=0.8,
                                   edgecolor=ec, facecolor=fc, zorder=2))
    if label:
        ax_c.text(cx, cy + BH_H - 0.20, label,
                  ha="center", va="top", fontsize=5.5,
                  fontweight="bold", color=ECBOX, fontstyle="italic")

# ── INPUTS ────────────────────────────────────────────────────────────────
sec_box(Y_IN, label="INPUTS")

T(ax_c, XL, Y_IN + 0.72, "Protein Sequence", fontsize=8, fontweight="bold", color=DARK)
seq = "MKTAYIAKQR"; bw_aa = 0.60
x0_aa = XL - len(seq) * bw_aa / 2
for idx, aa in enumerate(seq):
    xb = x0_aa + idx * bw_aa
    ax_c.add_patch(Rectangle((xb, Y_IN - 0.30), bw_aa - 0.04, 0.80,
                              fc=AA_COL.get(aa, "#CCC"), ec="white", lw=0.6, zorder=3))
    ax_c.text(xb + (bw_aa - 0.04)/2, Y_IN + 0.10, aa,
              ha="center", va="center", fontsize=5, fontweight="bold",
              color="#111", zorder=4)
ax_c.text(x0_aa + len(seq)*bw_aa + 0.05, Y_IN + 0.10, "…",
          ha="left", va="center", fontsize=8, color="#888")
T(ax_c, XL, Y_IN - 0.88, "1,438 amino acid residues", fontsize=6, color=MID)

T(ax_c, XR, Y_IN + 0.55, "Protein 3D Structure", fontsize=8, fontweight="bold", color=DARK)
T(ax_c, XR, Y_IN - 0.12, "UniProt ID → AlphaFold Database", fontsize=6, color=MID)
T(ax_c, XR, Y_IN - 0.75, "Cα atomic coordinates (PDB)", fontsize=5.5, color=LIGHT)

arr(ax_c, XL, Y_IN - BH_H, XL, Y_PRE + BH_H)
arr(ax_c, XR, Y_IN - BH_H, XR, Y_PRE + BH_H)

# ── PREPROCESSING ─────────────────────────────────────────────────────────
sec_box(Y_PRE, label="PREPROCESSING")

T(ax_c, XL, Y_PRE + 0.55, "Tokenization", fontsize=8, fontweight="bold", color=DARK)
T(ax_c, XL, Y_PRE - 0.10, "BPE encoding", fontsize=6, color=MID)
T(ax_c, XL, Y_PRE - 0.72, "Max length: 1,024 tokens", fontsize=5.5, color=LIGHT)

# Contact Map Extraction: title → subtitle → grid stacked vertically, centered at XR
T(ax_c, XR, Y_PRE + 0.55, "Contact Map Extraction", fontsize=8, fontweight="bold", color=DARK)
T(ax_c, XR, Y_PRE + 0.05, "8 Å threshold  ·  256 × 256", fontsize=6, color=MID)

NM_CM = 8; GSZ = 1.10; CS_CM = GSZ / NM_CM
mini_cm = np.zeros((NM_CM, NM_CM))
for ri in range(NM_CM):
    for ci in range(NM_CM):
        d = abs(ri - ci)
        if d == 0:   mini_cm[ri, ci] = 1.0
        elif d == 1: mini_cm[ri, ci] = 0.75
        elif d == 2: mini_cm[ri, ci] = 0.42
        elif d == 3: mini_cm[ri, ci] = 0.18
mini_cm[0][7] = mini_cm[7][0] = 0.82
mini_cm[1][6] = mini_cm[6][1] = 0.68
mini_cm[2][5] = mini_cm[5][2] = 0.52
mini_cm[1][5] = mini_cm[5][1] = 0.40
# grid centered at XR, below subtitle
cm_x0 = XR - GSZ / 2; cm_y0 = Y_PRE - 0.22 - GSZ
for ri in range(NM_CM):
    for ci in range(NM_CM):
        v = mini_cm[ri, ci]
        ax_c.add_patch(Rectangle(
            (cm_x0 + ci * CS_CM, cm_y0 + (NM_CM - 1 - ri) * CS_CM),
            CS_CM - 0.012, CS_CM - 0.012,
            fc=plt.cm.Blues(v * 0.85 + 0.08), ec="white", lw=0.5, zorder=4))
# i / j axis labels
cm_cx = cm_x0 + GSZ / 2
cm_cy = cm_y0 + GSZ / 2
ax_c.text(cm_cx, cm_y0 - 0.09, "j", ha="center", va="top",
          fontsize=5.5, color="#444", zorder=5)
ax_c.text(cm_x0 - 0.07, cm_cy, "i", ha="right", va="center",
          fontsize=5.5, color="#444", rotation=90, zorder=5)

arr(ax_c, XL, Y_PRE - BH_H, XL, Y_ENC + BH_H)
arr(ax_c, XR, Y_PRE - BH_H, XR, Y_ENC + BH_H)

# ── ENCODING ──────────────────────────────────────────────────────────────
sec_box(Y_ENC, label="ENCODING")

T(ax_c, XL, Y_ENC + 0.55, "ESM-2 650M", fontsize=8, fontweight="bold", color=DARK)
T(ax_c, XL, Y_ENC - 0.10, "Protein language model  (frozen)", fontsize=6, color=MID)
T(ax_c, XL, Y_ENC - 0.75, "→  1,280-d sequence embedding",
  fontsize=6.5, fontweight="bold", color=DARK)

T(ax_c, XR, Y_ENC + 0.55, "ResNet-50", fontsize=8, fontweight="bold", color=DARK)
T(ax_c, XR, Y_ENC - 0.10, "Convolutional encoder  (trainable)", fontsize=6, color=MID)
T(ax_c, XR, Y_ENC - 0.75, "→  512-d structural feature",
  fontsize=6.5, fontweight="bold", color=DARK)

arr(ax_c, XL, Y_ENC - BH_H, 7.0, Y_FUS + BH_H, rad=-0.18)
arr(ax_c, XR, Y_ENC - BH_H, 11.0, Y_FUS + BH_H, rad=0.18)

# ── FUSION ────────────────────────────────────────────────────────────────
sec_box(Y_FUS, bw=BW_CTR, label="FUSION")

T(ax_c, XC, Y_FUS + 0.45, "Gated Cross-Attention Fusion",
  fontsize=8, fontweight="bold", color="#065F46")
T(ax_c, XC, Y_FUS - 0.18, "Structure features gate sequence attention",
  fontsize=6, color=MID)
T(ax_c, XC, Y_FUS - 0.82, "1,792-d  →  512-d fused representation",
  fontsize=6.5, fontweight="bold", color="#065F46")

arr(ax_c, XC, Y_FUS - BH_H, XC, Y_CLS + BH_H)

# ── CLASSIFICATION HEAD ───────────────────────────────────────────────────
sec_box(Y_CLS, bw=BW_CTR, label="CLASSIFICATION HEAD")

T(ax_c, XC, Y_CLS + 0.45, "FC Head  +  Sigmoid",
  fontsize=8, fontweight="bold", color="#3730A3")
T(ax_c, XC, Y_CLS - 0.18, "Multi-label binary classification",
  fontsize=6, color=MID)
T(ax_c, XC, Y_CLS - 0.82, "1,938 EC class labels",
  fontsize=6.5, fontweight="bold", color="#3730A3")

arr(ax_c, XC, Y_CLS - BH_H, XC, Y_OUT + BH_H, lw=1.1)

# ── PREDICTED ENZYME FUNCTION ─────────────────────────────────────────────
sec_box(Y_OUT, bw=BW_CTR, label="PREDICTED ENZYME FUNCTION")

T(ax_c, XC, Y_OUT + 0.45, "EC 1.1.3.4  —  Glucose oxidase",
  fontsize=8, fontweight="bold", color="#166534")
T(ax_c, XC, Y_OUT - 0.18, "Oxidoreductase  ·  CH–OH donors  ·  O₂ as acceptor",
  fontsize=6, color="#333")
T(ax_c, XC, Y_OUT - 0.82, "Confidence: p = 0.94   (Aspergillus niger,  UniProt P13006)",
  fontsize=5.5, color=MID, fontstyle="italic")

fig.savefig(f"{OUT}/fig1_overview_print.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig1_overview_print.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig1 print done")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2
# ═══════════════════════════════════════════════════════════════════════════
MODELS = ["MMseqs2", "HIT-EC", "Contact-EC", "Structure-only", "Sequence-only"]
M_COLS = [MC["MM"], MC["HIT"], MC["CE"], MC["Str"], MC["Seq"]]
d23 = dict(means=[0.5852, 0.8471, 0.6241, 0.4244, 0.4508],
           errs=[0, 0, 0.0170, 0.0207, 0.0203])
d24 = dict(means=[0.7080, 0.4578, 0.6819, 0.4388, 0.3892],
           errs=[0, 0, 0.0026, 0.0166, 0.0121])

fig, (ax_a2, ax_b2) = plt.subplots(1, 2, figsize=(6.7, 3.2))
fig.subplots_adjust(left=0.20, right=0.97, wspace=0.44, top=0.84, bottom=0.28)

y = np.arange(len(MODELS)); h = 0.60

def horiz_bars(ax, data, title, panel_ch):
    ax.barh(y, data["means"], h, color=M_COLS,
            edgecolor="white", linewidth=0.5, zorder=3)
    for i, (v, e) in enumerate(zip(data["means"], data["errs"])):
        if e > 0:
            ax.errorbar(v, i, xerr=e, fmt="none",
                        elinewidth=1.2, ecolor="#111",
                        capsize=4, capthick=1.2, zorder=5)
        ax.text(v + max(e, 0) + 0.018, i, f"{v:.3f}",
                va="center", ha="left", fontsize=6.5, fontweight="bold", color="#111")
    ax.set_xlim(0, 1.20)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6)
    ax.set_ylim(-0.55, len(MODELS) - 0.45)
    ax.set_yticks(y)
    ax.set_yticklabels(MODELS, fontsize=7)
    ax.set_xlabel("Level-4 micro F1", fontsize=7, labelpad=4)
    ax.set_title(title, fontsize=7, pad=6, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.barh(2, data["means"][2], h, color="none",
            edgecolor=MC["CE"], linewidth=1.4, zorder=4)
    ax.text(-0.14, 1.06, panel_ch, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")

horiz_bars(ax_a2, d23, "A   SP-2023-01  (N = 124)", "A")
horiz_bars(ax_b2, d24, "B   SP-2024  (N = 1,226)", "B")

legend_patches = [
    mpatches.Patch(color=MC["MM"],  label="MMseqs2"),
    mpatches.Patch(color=MC["HIT"], label="HIT-EC"),
    mpatches.Patch(color=MC["CE"],  label="Contact-EC  (ours)"),
    mpatches.Patch(color=MC["Str"], label="Structure-only"),
    mpatches.Patch(color=MC["Seq"], label="Sequence-only"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           fontsize=6, frameon=False, bbox_to_anchor=(0.57, -0.05))

fig.savefig(f"{OUT}/fig2_temporal_bars_print.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig2_temporal_bars_print.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig2 print done")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.3, 2.6))
fig.subplots_adjust(left=0.12, right=0.66, top=0.88, bottom=0.30)

traces = [
    ("MMseqs2",        0.5852, 0.7080, MC["MM"],  1.4, ":",  "D"),
    ("HIT-EC",         0.8471, 0.4578, MC["HIT"], 2.0, "-",  "s"),
    ("Contact-EC",     0.6241, 0.6819, MC["CE"],  2.0, "-",  "o"),
    ("Structure-only", 0.4244, 0.4388, MC["Str"], 1.4, "--", "^"),
    ("Sequence-only",  0.4508, 0.3892, MC["Seq"], 1.4, "--", "v"),
]
for (lbl, v23, v24, col, lw, ls, mk) in traces:
    ax.plot([0, 1], [v23, v24], color=col, linewidth=lw, linestyle=ls,
            marker=mk, markersize=5, markerfacecolor=col,
            markeredgecolor="white", markeredgewidth=0.8, zorder=4,
            solid_capstyle="round")

def spread(vals, gap=0.034):
    p = list(vals)
    for _ in range(500):
        for i in range(1, len(p)):
            if p[i-1] - p[i] < gap:
                mid = (p[i-1] + p[i]) / 2
                p[i-1] = mid + gap/2; p[i] = mid - gap/2
    return p

L = sorted(traces, key=lambda t: t[1], reverse=True)
R = sorted(traces, key=lambda t: t[2], reverse=True)
lp = spread([t[1] for t in L])
rp = spread([t[2] for t in R])

for i, (lbl, v23, v24, col, lw, ls, mk) in enumerate(L):
    ax.text(-0.06, lp[i], f"{v23:.3f}", ha="right", va="center",
            fontsize=5.5, color=col, fontweight="bold")
for i, (lbl, v23, v24, col, lw, ls, mk) in enumerate(R):
    delta = v24 - v23
    sign = "+" if delta >= 0 else "−"
    ax.text(1.06, rp[i],
            f"{v24:.3f}  ({sign}{abs(delta)*100:.1f} pp)",
            ha="left", va="center", fontsize=5.5, color=col, fontweight="bold")

ax.set_xticks([0, 1])
ax.set_xticklabels(["SP-2023-01\n(N = 124)", "SP-2024\n(N = 1,226)"], fontsize=7)
ax.set_xlim(-0.32, 1.30)
ax.set_ylim(0.27, 0.98)
ax.set_ylabel("Level-4 micro F1", fontsize=7, labelpad=4)
ax.set_title("Model trajectories across temporal horizons", fontsize=7, pad=6)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

legend_elements = [
    plt.Line2D([0],[0], color=MC["CE"],  lw=2.0, ls="-",  marker="o",
               markersize=4, label="Contact-EC"),
    plt.Line2D([0],[0], color=MC["HIT"], lw=2.0, ls="-",  marker="s",
               markersize=4, label="HIT-EC"),
    plt.Line2D([0],[0], color=MC["MM"],  lw=1.4, ls=":",  marker="D",
               markersize=4, label="MMseqs2"),
    plt.Line2D([0],[0], color=MC["Str"], lw=1.4, ls="--", marker="^",
               markersize=4, label="Structure-only"),
    plt.Line2D([0],[0], color=MC["Seq"], lw=1.4, ls="--", marker="v",
               markersize=4, label="Sequence-only"),
]
ax.legend(handles=legend_elements,
          loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=3, frameon=False, fontsize=6)

fig.savefig(f"{OUT}/fig3_reversal_slope_print.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig3_reversal_slope_print.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig3 print done")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4
# ═══════════════════════════════════════════════════════════════════════════
cond_labels = ["Original", "Permuted", "Secondary\nstructure", "Random"]
cond_cols = [CC["Original"], CC["Permuted"], CC["Secondary"], CC["Random"]]
panels = [
    ("A   Structure-only",      [0.3646, 0.0000, 0.0000, 0.0000]),
    ("B   Contact-EC",          [0.6032, 0.0625, 0.5242, 0.0982]),
    ("C   Contact-EC  (gated)", [0.5690, 0.5702, 0.5776, 0.5641]),
]

fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.6), sharey=True)
fig.subplots_adjust(wspace=0.06, top=0.84, bottom=0.32, left=0.10, right=0.98)

x = np.arange(4)
for ax, (title, vals) in zip(axes, panels):
    ax.bar(x, vals, width=0.62, color=cond_cols,
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(y=0, color="black", lw=0.8, zorder=6)
    for xi, v in zip(x, vals):
        if v >= 0.015:
            ax.text(xi, v + 0.016, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6,
                    fontweight="bold", color="#111")
        else:
            ax.text(xi, 0.022, "0.000",
                    ha="center", va="bottom", fontsize=6,
                    fontweight="bold", color="#999")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=6)
    ax.set_ylim(0, 0.84)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=7, pad=5, fontweight="bold")

axes[0].set_ylabel("Level-4 micro F1", fontsize=7)

legend_patches = [
    mpatches.Patch(color=CC["Original"],  label="Original"),
    mpatches.Patch(color=CC["Permuted"],  label="Row/col permutation"),
    mpatches.Patch(color=CC["Secondary"], label="Secondary structure  (|i−j| < 12)"),
    mpatches.Patch(color=CC["Random"],    label="Density-matched random"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=4,
           fontsize=6, frameon=False, bbox_to_anchor=(0.53, 0.0))
fig.suptitle("Contact-map perturbation controls  (SP-2023-01, N = 124)",
             fontsize=7, fontweight="bold", y=1.0)

fig.savefig(f"{OUT}/fig4_perturbation_print.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/fig4_perturbation_print.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig4 print done")

print(f"\nAll print figures → {OUT}")
