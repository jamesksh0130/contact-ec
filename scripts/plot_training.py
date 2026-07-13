"""
학습 로그 파싱 → 다양한 피규어 자동 생성

생성 파일 목록 (outputs/results/):
  overview_micro_f1.png   — 전체 모델 Micro F1 비교
  overview_val_loss.png   — 전체 모델 Val Loss 비교
  overview_macro_f1.png   — 전체 모델 Macro F1 비교
  overview_time.png       — 전체 모델 에폭당 시간 비교
  summary_bar.png         — 모델별 베스트 F1 막대 그래프
  {model_key}_loss.png    — 모델별 Loss 곡선
  {model_key}_f1.png      — 모델별 F1 곡선
  {model_key}_time.png    — 모델별 에폭당 시간

사용법:
  python scripts/plot_training.py           # 현재 로그 즉시 시각화
  python scripts/plot_training.py --watch   # 120초마다 자동 업데이트
"""
import re, time, argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
_CANDIDATE_LOGS = [Path("/tmp/pipeline.log"), ROOT / "outputs" / "pipeline.log"]
LOG_PATH = next((p for p in _CANDIDATE_LOGS if p.exists()), _CANDIDATE_LOGS[0])
OUT_DIR  = ROOT / "outputs" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "b0_cnn":       "#e74c3c",
    "b1_esm2_fc":   "#3498db",
    "b2_esm2_hier": "#2ecc71",
    "b3_contact":   "#f39c12",
    "fusion":       "#9b59b6",
}
MODEL_LABELS = {
    "b0_cnn":       "B0: 1D CNN",
    "b1_esm2_fc":   "B1: ESM-2 + FC",
    "b2_esm2_hier": "B2: ESM-2 Hier.",
    "b3_contact":   "B3: Contact ResNet",
    "fusion":       "Fusion (Ours)",
}

EPOCH_RE    = re.compile(
    r"\[(\d+)/\d+\]\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+"
    r"micro_f1=([\d.]+)\s+macro_f1=([\d.]+)\s+\((\d+)s\)"
)
MODEL_START = re.compile(r"모델: (\S+)\s+\|.*Phase: (\d+)")


# ── 로그 파싱 ────────────────────────────────────────────────
def parse_log(log_path: Path) -> dict:
    data = defaultdict(list)
    cur_model, cur_phase = None, "1"
    if not log_path.exists():
        return data
    for line in log_path.read_text(errors="ignore").splitlines():
        m = MODEL_START.search(line)
        if m:
            cur_model, cur_phase = m.group(1), m.group(2)
            continue
        m = EPOCH_RE.search(line)
        if m and cur_model:
            key = f"{cur_model}_p{cur_phase}"
            data[key].append({
                "epoch":      int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss":   float(m.group(3)),
                "micro_f1":   float(m.group(4)),
                "macro_f1":   float(m.group(5)),
                "sec":        int(m.group(6)),
            })
    return dict(data)


def _label(key: str) -> str:
    model = key.split("_p")[0]
    phase = key.split("_p")[1]
    l = MODEL_LABELS.get(model, model)
    return l if phase == "1" else f"{l} P{phase}"


def _color(key: str) -> str:
    return MODEL_COLORS.get(key.split("_p")[0], "#888")


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{time.strftime('%H:%M:%S')}] 저장: {path.name}")


# ── 1. 전체 모델 Micro F1 비교 ───────────────────────────────
def plot_overview_micro_f1(data: dict):
    fig, ax = plt.subplots(figsize=(11, 5))
    for key, rows in data.items():
        epochs = [r["epoch"] for r in rows]
        vals   = [r["micro_f1"] for r in rows]
        best   = max(vals) if vals else 0
        ax.plot(epochs, vals, "-o", markersize=4, linewidth=2,
                color=_color(key), label=f"{_label(key)}  (best={best:.4f})")
    ax.axhline(0.93, color="red", linestyle="--", linewidth=1.5,
               alpha=0.8, label="HIT-EC SOTA (0.93)")
    ax.set_title("Validation Micro F1 — All Models", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Micro F1")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, OUT_DIR / "overview_micro_f1.png")


# ── 2. 전체 모델 Val Loss 비교 ──────────────────────────────
def plot_overview_val_loss(data: dict):
    fig, ax = plt.subplots(figsize=(11, 5))
    for key, rows in data.items():
        epochs = [r["epoch"] for r in rows]
        vals   = [r["val_loss"] for r in rows]
        ax.plot(epochs, vals, "-o", markersize=4, linewidth=2,
                color=_color(key), label=_label(key))
    ax.set_title("Validation Loss — All Models", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, OUT_DIR / "overview_val_loss.png")


# ── 3. 전체 모델 Macro F1 비교 ──────────────────────────────
def plot_overview_macro_f1(data: dict):
    fig, ax = plt.subplots(figsize=(11, 5))
    for key, rows in data.items():
        epochs = [r["epoch"] for r in rows]
        vals   = [r["macro_f1"] for r in rows]
        best   = max(vals) if vals else 0
        ax.plot(epochs, vals, "-o", markersize=4, linewidth=2,
                color=_color(key), label=f"{_label(key)}  (best={best:.4f})")
    ax.set_title("Validation Macro F1 — All Models", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, OUT_DIR / "overview_macro_f1.png")


# ── 4. 전체 모델 에폭당 시간 비교 ───────────────────────────
def plot_overview_time(data: dict):
    fig, ax = plt.subplots(figsize=(11, 5))
    for key, rows in data.items():
        epochs = [r["epoch"] for r in rows]
        mins   = [r["sec"] / 60 for r in rows]
        avg    = np.mean(mins) if mins else 0
        ax.plot(epochs, mins, "-o", markersize=4, linewidth=2,
                color=_color(key), label=f"{_label(key)}  (avg={avg:.1f}min)")
    ax.set_title("Training Time per Epoch — All Models", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Minutes / Epoch")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, OUT_DIR / "overview_time.png")


# ── 5. 베스트 F1 막대 그래프 ─────────────────────────────────
def plot_summary_bar(data: dict):
    keys   = list(data.keys())
    labels = [_label(k) for k in keys]
    micro  = [max(r["micro_f1"] for r in data[k]) for k in keys]
    macro  = [max(r["macro_f1"] for r in data[k]) for k in keys]
    colors = [_color(k) for k in keys]

    x = np.arange(len(keys))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 2), 5))
    bars1 = ax.bar(x - w/2, micro, w, label="Micro F1", color=colors, alpha=0.85)
    bars2 = ax.bar(x + w/2, macro, w, label="Macro F1", color=colors, alpha=0.45,
                   edgecolor=colors, linewidth=1.2)
    ax.axhline(0.93, color="red", linestyle="--", linewidth=1.5,
               alpha=0.8, label="HIT-EC SOTA (0.93)")

    for bar, val in zip(bars1, micro):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar, val in zip(bars2, macro):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_title("Best Validation F1 — Model Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.08)
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, OUT_DIR / "summary_bar.png")


# ── 6. 모델별 개별 Loss 곡선 ─────────────────────────────────
def plot_model_loss(key: str, rows: list):
    color = _color(key); label = _label(key)
    epochs     = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss   = [r["val_loss"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, "-",  color=color, linewidth=2, label="Train Loss")
    ax.plot(epochs, val_loss,   "--", color=color, linewidth=2, alpha=0.7, label="Val Loss")
    ax.set_title(f"{label} — Loss Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, OUT_DIR / f"{key}_loss.png")


# ── 7. 모델별 개별 F1 곡선 ───────────────────────────────────
def plot_model_f1(key: str, rows: list):
    color = _color(key); label = _label(key)
    epochs   = [r["epoch"] for r in rows]
    micro_f1 = [r["micro_f1"] for r in rows]
    macro_f1 = [r["macro_f1"] for r in rows]
    best     = max(micro_f1) if micro_f1 else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, micro_f1, "-o", color=color, markersize=4,
            linewidth=2, label="Micro F1")
    ax.plot(epochs, macro_f1, "--s", color=color, markersize=4,
            linewidth=1.5, alpha=0.6, label="Macro F1")
    ax.axhline(0.93, color="red", linestyle=":", linewidth=1.2,
               alpha=0.6, label="HIT-EC (0.93)")
    ax.set_title(f"{label} — F1 Curve  (best={best:.4f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, OUT_DIR / f"{key}_f1.png")


# ── 8. 모델별 개별 에폭당 시간 ───────────────────────────────
def plot_model_time(key: str, rows: list):
    color = _color(key); label = _label(key)
    epochs = [r["epoch"] for r in rows]
    mins   = [r["sec"] / 60 for r in rows]
    avg    = np.mean(mins) if mins else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(epochs, mins, color=color, alpha=0.65, width=0.8)
    ax.axhline(avg, color="gray", linestyle="--", linewidth=1.5,
               label=f"Avg {avg:.1f} min")
    ax.set_title(f"{label} — Time per Epoch  (avg {avg:.1f} min)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Minutes")
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, OUT_DIR / f"{key}_time.png")


# ── 전체 실행 ─────────────────────────────────────────────────
def plot_all(data: dict):
    if not data:
        print("아직 학습 로그 없음 — 나중에 다시 시도하세요.")
        return

    # Overview 피규어들
    plot_overview_micro_f1(data)
    plot_overview_val_loss(data)
    plot_overview_macro_f1(data)
    plot_overview_time(data)
    plot_summary_bar(data)

    # 모델별 개별 피규어
    for key, rows in data.items():
        plot_model_loss(key, rows)
        plot_model_f1(key, rows)
        plot_model_time(key, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch",    action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()

    while True:
        data = parse_log(LOG_PATH)
        plot_all(data)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
