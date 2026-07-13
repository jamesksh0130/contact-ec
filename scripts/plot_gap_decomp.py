"""
Gap Decomposition Figure for Contact-EC paper.
Shows: data recency (+11.8pp) vs architecture gap (-12.6pp)
relative to HIT-EC temporal baseline (0.8471).
Automatically adds E2E bar if outputs/results/expa_e2e_eval.json exists.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ROOT = Path("/home/user/Desktop/unlv")
# expa_e2e_full_eval.json (verified eval, flatfc_e2e_cached_best.pt, N=124)
E2E_JSON = ROOT / "outputs/results/expa_e2e_full_eval.json"

# ── Load E2E result if available ──────────────────────────────────────────────
e2e_f1 = None
if E2E_JSON.exists():
    data = json.loads(E2E_JSON.read_text())
    # expa_e2e_full_eval.json has nested structure: {"e2e": {"micro_f1": ...}}
    if "e2e" in data:
        e2e_f1 = data["e2e"].get("micro_f1")
    else:
        e2e_f1 = data.get("micro_f1")
    print(f"E2E result loaded: micro_f1={e2e_f1}")
else:
    print("E2E result not yet available — plotting without E2E bar.")

# ── Data ──────────────────────────────────────────────────────────────────────
models = [
    "Contact-EC\n(SP-2018)",
    "Contact-EC-ExpA\n(SP-2026, 243K)",
]
scores = [0.6032, 0.7209]
COLORS = ["#4878CF", "#4878CF"]
EDGE   = ["#2B5490", "#2B5490"]
HATCH  = ["", "///"]

if e2e_f1 is not None:
    models.append("Contact-EC-E2E\n(SP-2026, E2E ft)")
    scores.append(e2e_f1)
    COLORS.append("#2E6FAF")
    EDGE.append("#1B4880")
    HATCH.append("xxx")

models.append("HIT-EC\n(temporal test)")
scores.append(0.8471)
COLORS.append("#C44E52")
EDGE.append("#8B1A1A")
HATCH.append("")

# ── Figure setup ──────────────────────────────────────────────────────────────
fig_w = 8.0 if e2e_f1 is not None else 6.5
fig, ax = plt.subplots(figsize=(fig_w, 4.2))
fig.subplots_adjust(left=0.11, right=0.97, top=0.88, bottom=0.18)

x = np.arange(len(models))
bars = ax.bar(x, scores, width=0.52, color=COLORS, edgecolor=EDGE,
              linewidth=1.2, hatch=HATCH, zorder=3)

# ── F1 value labels on top of bars ───────────────────────────────────────────
for bar, val in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + 0.006, f"{val:.4f}",
            ha="center", va="bottom", fontsize=9.0, fontweight="bold",
            color="#1a1a1a")

# ── Arrow: data recency (Contact-EC → Contact-EC-ExpA) ───────────────────────
y0, y1 = scores[0], scores[1]
pp_data = (y1 - y0) * 100
ax.annotate("", xy=(x[1], y1 + 0.012), xytext=(x[0], y0 + 0.012),
            arrowprops=dict(arrowstyle="<->", color="#2B7D2F", lw=1.6))
ax.text((x[0] + x[1]) / 2, max(y0, y1) + 0.028,
        f"+{pp_data:.1f} pp\n(data recency)",
        ha="center", va="bottom", fontsize=8.5,
        color="#2B7D2F", fontweight="bold")

# ── Arrow: E2E fine-tuning (ExpA → E2E) or architecture gap ──────────────────
if e2e_f1 is not None:
    y_e2e = e2e_f1
    pp_e2e = (y_e2e - y1) * 100
    sign = "+" if pp_e2e >= 0 else ""
    ax.annotate("", xy=(x[2], y_e2e + 0.012), xytext=(x[1], y1 + 0.012),
                arrowprops=dict(arrowstyle="<->", color="#1B6B9E", lw=1.6))
    ax.text((x[1] + x[2]) / 2, max(y1, y_e2e) + 0.028,
            f"{sign}{pp_e2e:.1f} pp\n(E2E fine-tuning)",
            ha="center", va="bottom", fontsize=8.5,
            color="#1B6B9E", fontweight="bold")
    # remaining architecture gap (E2E → HIT-EC)
    y_hitec = scores[-1]
    pp_arch = (y_hitec - y_e2e) * 100
    sign2 = "+" if pp_arch >= 0 else ""
    ax.annotate("", xy=(x[3], y_hitec + 0.012), xytext=(x[2], y_e2e + 0.012),
                arrowprops=dict(arrowstyle="<->", color="#B8500A", lw=1.6))
    ax.text((x[2] + x[3]) / 2, max(y_e2e, y_hitec) + 0.028,
            f"{sign2}{pp_arch:.1f} pp\n(architecture)",
            ha="center", va="bottom", fontsize=8.5,
            color="#B8500A", fontweight="bold")
else:
    # no E2E: single architecture arrow ExpA → HIT-EC
    y2 = scores[-1]
    pp_arch = (y2 - y1) * 100
    ax.annotate("", xy=(x[-1], y2 + 0.012), xytext=(x[1], y1 + 0.012),
                arrowprops=dict(arrowstyle="<->", color="#B8500A", lw=1.6))
    ax.text((x[1] + x[-1]) / 2, y2 + 0.028,
            f"{pp_arch:+.1f} pp\n(architecture)",
            ha="center", va="bottom", fontsize=8.5,
            color="#B8500A", fontweight="bold")

# ── Axes styling ──────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9.0)
ax.set_ylabel("Temporal Micro F1  (EC-Bench 2023-01)", fontsize=10)
ax.set_ylim(0.50, 0.96)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
p1 = mpatches.Patch(facecolor="#4878CF", edgecolor="#2B5490", lw=1.2,
                    label="Contact-EC (ours, SP-2018)")
p2 = mpatches.Patch(facecolor="#4878CF", edgecolor="#2B5490", lw=1.2,
                    hatch="///", label="Contact-EC-ExpA (matched corpus)")
patches = [p1, p2]
if e2e_f1 is not None:
    p_e2e = mpatches.Patch(facecolor="#2E6FAF", edgecolor="#1B4880", lw=1.2,
                            hatch="xxx", label="Contact-EC-E2E (end-to-end ft)")
    patches.append(p_e2e)
p3 = mpatches.Patch(facecolor="#C44E52", edgecolor="#8B1A1A", lw=1.2,
                    label="HIT-EC (SOTA, temporal)")
patches.append(p3)
ax.legend(handles=patches, fontsize=8.0,
          loc="upper left", framealpha=0.9, edgecolor="#cccccc")

ax.set_title(
    "Gap Decomposition: Contact-EC vs. HIT-EC on Temporal Test",
    fontsize=10.5, fontweight="bold", pad=8,
)

out = ROOT / "paper/gap_decomp.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
