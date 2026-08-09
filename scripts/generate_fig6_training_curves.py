"""
supplementary.tex S3 (Training Dynamics) figure — val-hard Level-4 micro F1
vs. epoch for B1 / B2 / B3 / Contact-EC-Hier.

Source-of-truth notes (see outputs/logs/):
  - B1: ecbench_b1.log (30 epochs, final 0.7655 matches Table dynamics
    exactly). An older ecbench_b1_resume.log (5 epochs, 2026-07-02) predates
    this full run (2026-07-04) and is superseded by it — not used.
  - B2: ecbench_b2.log (30 epochs, only log available).
  - B3: ecbench_b3.log stalls/diverges after epoch 10 (continues to a
    higher, unreported 0.7781 by epoch 29) and was superseded by
    ecbench_b3_phase1_resume.log (2026-07-02 05:59, ~2h after b3.log's
    03:55), which resumes from the epoch-10 checkpoint and reaches 0.7650
    at its own epoch 10 — this exactly matches the reported table value,
    so the plotted curve is b3.log[:10] + b3_phase1_resume (offset +10).
  - Contact-EC-Hier: ecbench_fv2_phase1.log (24 epochs, frozen ESM-2) +
    ecbench_fv2_phase2.log (20 epochs, partial unfreeze, offset +24).
    Phase 2 epoch 1 (0.8601) picks up almost exactly where Phase 1 epoch 24
    (0.8585) left off, confirming Phase 2 continued from Phase 1's final
    checkpoint. A separate ecbench_fv2_phase1_resume.log restarts from an
    earlier, lower checkpoint (0.7859) and was not the run Phase 2 actually
    continued from — not used. Phase 2 peaks at epoch 16 with micro_f1
    =0.8698 ("학습 완료! 베스트 Micro F1: 0.8698" in the log), matching
    Table dynamics.

Contact-EC (flat FC)'s per-epoch history is not plotted: no training log
for checkpoint ecbench_b4_flatfc_best.pt was found anywhere in the repo
(only the final-eval JSONs survive), so its curve cannot be reconstructed.
Its final value remains in the Table dynamics as a single number.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "outputs" / "logs"
OUT = ROOT / "paper" / "fig6_training_curves.png"

DARK = "#1D3557"
BLUE = "#2E86AB"
GRAY = "#6C757D"
GREEN = "#48A999"
PURPLE = "#7B4B94"


def parse(path, epoch_offset=0):
    epochs, vals = [], []
    for line in Path(path).read_text().splitlines():
        m = re.search(r"^\[(\d+)/\d+\].*micro_f1=([\d.]+)", line)
        if m:
            epochs.append(int(m.group(1)) + epoch_offset)
            vals.append(float(m.group(2)))
    return epochs, vals


def main():
    b1_e, b1_v = parse(LOGDIR / "ecbench_b1.log")
    b2_e, b2_v = parse(LOGDIR / "ecbench_b2.log")

    b3a_e, b3a_v = parse(LOGDIR / "ecbench_b3.log")
    b3b_e, b3b_v = parse(LOGDIR / "ecbench_b3_phase1_resume.log", epoch_offset=10)
    b3_e, b3_v = b3a_e[:10] + b3b_e, b3a_v[:10] + b3b_v

    hier1_e, hier1_v = parse(LOGDIR / "ecbench_fv2_phase1.log")
    hier2_e, hier2_v = parse(LOGDIR / "ecbench_fv2_phase2.log", epoch_offset=24)
    hier_e, hier_v = hier1_e + hier2_e, hier1_v + hier2_v
    phase_boundary = 24

    print("B1 final:", b1_v[-1], "expect 0.7655")
    print("B2 final:", b2_v[-1], "peak:", max(b2_v), "expect ~0.4204")
    print("B3 final:", b3_v[-1], "expect 0.7650")
    print("Hier final:", hier_v[-1], "peak:", max(hier_v), "expect 0.8698")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    })

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    ax.plot(b1_e, b1_v, color=BLUE, lw=2.0, marker="o", markersize=3,
            label="B1 (ESM-2, flat FC)")
    ax.plot(b2_e, b2_v, color=GRAY, lw=2.0, marker="o", markersize=3,
            label="B2 (ESM-2, hier. FC)")
    ax.plot(b3_e, b3_v, color=GREEN, lw=2.0, marker="o", markersize=3,
            label="B3 (Contact map only)")
    ax.plot(hier_e, hier_v, color=PURPLE, lw=2.4, marker="o", markersize=3,
            label="Contact-EC-Hier")

    ax.axvline(phase_boundary + 0.5, color=DARK, linestyle=":",
                linewidth=1.3, alpha=0.7)
    ax.text(phase_boundary + 0.7, 0.12, "Phase 1 → Phase 2\n(Contact-EC-Hier)",
            fontsize=8.5, color=DARK, va="bottom")

    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Val-hard Level-4 micro F1", fontsize=12, fontweight="bold")
    ax.set_title("Training Dynamics on EC-Bench Hard Validation", fontsize=13,
                 fontweight="bold", color=DARK, pad=10)
    ax.legend(fontsize=10, loc="center right")
    ax.set_ylim(0, 0.95)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("saved", OUT)


if __name__ == "__main__":
    main()
