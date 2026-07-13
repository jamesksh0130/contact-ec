"""
에러 분석 스크립트 — FusionV3 30ep 테스트셋 예측 저장 후 분석
생성 Figure:
  fig4a_l1_confusion.png    — L1 (7×7) 혼동 행렬
  fig4b_longtail.png        — EC class 빈도 vs F1 (롱테일 분석)
  fig4c_per_l1_f1.png       — L1 클래스별 L4 Micro F1 비교 (우리 vs B1)
"""
import sys, pickle, yaml
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

from models.dataset import ContactPairDataset, collate_fn_v3, ProteinDataset, collate_fn
import evaluate as ev   # build_model 재사용

# ── 경로 ────────────────────────────────────────────────────────
CKPT      = ROOT / "outputs/checkpoints/fusion_v3_phase1_best.pt"
CKPT_B1   = ROOT / "outputs/checkpoints/b1_esm2_fc_phase1_best.pt"
ENC_PATH  = ROOT / "data/label_encoders.pkl"
META_CSV  = ROOT / CFG["paths"]["meta_csv"]
SPLITS    = ROOT / CFG["paths"]["splits_dir"]
PAIR_DIR  = ROOT / "data" / "processed" / "contact_pair_embs"
OUT       = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE   = "#2E86AB"; GREEN  = "#48A999"; ORANGE = "#F4845F"
PURPLE = "#7B4B94"; RED    = "#E63946"; DARK   = "#1D3557"
GRAY   = "#6C757D"; GOLD   = "#F4A261"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
})

EC_CLASS_NAMES = {
    0: "1. Oxidoreductases",
    1: "2. Transferases",
    2: "3. Hydrolases",
    3: "4. Lyases",
    4: "5. Isomerases",
    5: "6. Ligases",
    6: "7. Translocases",
}

# ────────────────────────────────────────────────────────────────
def load_encoders():
    with open(ENC_PATH, "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    return encoders, n_classes


def run_inference_v3(device, batch_size=128):
    encoders, n_classes = load_encoders()
    ckpt  = torch.load(CKPT, map_location=device)
    model = ev.build_model("fusion_v3", n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"FusionV3 로드 완료 (val_micro_f1={ckpt.get('micro_f1','?')})")

    ds = ContactPairDataset(
        ids_file     = str(SPLITS / "test_ids.txt"),
        meta_csv     = str(META_CSV),
        embed_dir    = str(ROOT / CFG["paths"]["embed_dir"]),
        cmap_dir     = str(ROOT / CFG["paths"]["cmap_dir"]),
        label_enc_pkl= str(ENC_PATH),
        pair_emb_dir = str(PAIR_DIR),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn_v3, pin_memory=True)

    all_preds  = [[], [], [], []]
    all_labels = [[], [], [], []]
    all_masks  = []

    with torch.no_grad():
        for batch in loader:
            esm, cmap, seqs, labels, masks, l4mh, _, pair = batch
            esm  = esm.to(device)
            pair = pair.to(device)
            logits = model(esm, pair)
            for i in range(4):
                all_preds[i].extend(logits[i].argmax(dim=1).cpu().tolist())
            all_labels[0].extend(labels[:, 0].tolist())
            all_labels[1].extend(labels[:, 1].tolist())
            all_labels[2].extend(labels[:, 2].tolist())
            all_labels[3].extend(labels[:, 3].tolist())
            all_masks.extend(masks.tolist())

    del model; torch.cuda.empty_cache()
    return (
        [np.array(all_preds[i])  for i in range(4)],
        [np.array(all_labels[i]) for i in range(4)],
        np.array(all_masks),
        encoders, n_classes,
    )


def run_inference_b1(device, n_classes, batch_size=256):
    ckpt  = torch.load(CKPT_B1, map_location=device)
    model = ev.build_model("b1_esm2_fc", n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"B1 로드 완료 (val_micro_f1={ckpt.get('micro_f1','?')})")

    ds = ProteinDataset(
        ids_file     = str(SPLITS / "test_ids.txt"),
        meta_csv     = str(META_CSV),
        embed_dir    = str(ROOT / CFG["paths"]["embed_dir"]),
        cmap_dir     = str(ROOT / CFG["paths"]["cmap_dir"]),
        label_enc_pkl= str(ENC_PATH),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    all_preds  = [[], [], [], []]
    all_labels = [[], [], [], []]
    all_masks  = []

    with torch.no_grad():
        for batch in loader:
            esm, cmap, seqs, labels, masks, l4mh, _ = batch
            esm = esm.to(device)
            logits = model(esm, cmap.to(device))
            for i in range(4):
                all_preds[i].extend(logits[i].argmax(1).cpu().tolist())
            for i in range(4):
                all_labels[i].extend(labels[:, i].tolist())
            all_masks.extend(masks.tolist())

    del model; torch.cuda.empty_cache()
    return (
        [np.array(all_preds[i])  for i in range(4)],
        [np.array(all_labels[i]) for i in range(4)],
        np.array(all_masks),
    )


# ── Fig 4a: L1 Confusion Matrix ──────────────────────────────────
def fig4a_confusion(preds, labels, masks):
    valid = masks[:, 0] == 1
    p1 = preds[0][valid]
    l1 = labels[0][valid]

    cm = confusion_matrix(l1, p1, labels=list(range(7)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    cmap_colors = LinearSegmentedColormap.from_list(
        "wb", ["#F8F9FA", BLUE], N=256)
    im = ax.imshow(cm_norm, cmap=cmap_colors, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    short = ["Oxidored.", "Transfer.", "Hydrolase", "Lyase",
             "Isomerase", "Ligase", "Translocase"]
    ax.set_xticks(range(7)); ax.set_xticklabels(short, rotation=30, ha="right", fontsize=9.5)
    ax.set_yticks(range(7)); ax.set_yticklabels(short, fontsize=9.5)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True",      fontsize=12, fontweight="bold")

    for i in range(7):
        for j in range(7):
            val = cm_norm[i, j]
            cnt = cm[i, j]
            color = "white" if val > 0.5 else DARK
            ax.text(j, i, f"{val:.2f}\n({cnt:,})",
                    ha="center", va="center", fontsize=7.5,
                    color=color, fontweight="bold" if i == j else "normal")

    micro = f1_score(l1, p1, average="micro")
    macro = f1_score(l1, p1, average="macro")
    ax.set_title(f"Level-1 Confusion Matrix  (Micro F1={micro:.4f}, Macro F1={macro:.4f})",
                 fontsize=12, fontweight="bold", color=DARK, pad=10)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig4a_l1_confusion.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("fig4a_l1_confusion.png 저장 완료")


# ── Fig 4b: 롱테일 — 학습 빈도 vs Per-class F1 ─────────────────
def fig4b_longtail(preds, labels, masks, encoders):
    # 학습 셋 빈도 계산
    train_ids = (SPLITS / "train_ids.txt").read_text().splitlines()
    meta = pd.read_csv(META_CSV)
    train_meta = meta[meta["accession"].isin(set(train_ids)) & (meta["m4"] == 1)]
    train_counts = train_meta["l4_idx"].value_counts().to_dict()

    # 유효 샘플만
    valid = masks[:, 3] == 1
    p4 = preds[3][valid]
    l4 = labels[3][valid]
    valid_l4 = l4[l4 >= 0]
    valid_p4 = p4[l4 >= 0]

    classes = np.unique(valid_l4)
    per_cls_f1 = {}
    for c in classes:
        mask_c = valid_l4 == c
        if mask_c.sum() < 2:
            continue
        f1 = f1_score(valid_l4[mask_c], valid_p4[mask_c], average="micro", zero_division=0)
        cnt = train_counts.get(int(c), 0)
        per_cls_f1[int(c)] = (cnt, f1)

    counts = np.array([v[0] for v in per_cls_f1.values()])
    f1s    = np.array([v[1] for v in per_cls_f1.values()])

    # 구간별 색상
    bins   = [0, 5, 25, 100, 500, 10000]
    labels_bin = ["1–5\n(very rare)", "6–25\n(rare)", "26–100\n(uncommon)",
                  "101–500\n(moderate)", ">500\n(frequent)"]
    bin_colors = [RED, ORANGE, GOLD, GREEN, BLUE]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#F8F9FA")

    # ── 왼쪽: scatter ──
    ax = axes[0]
    ax.set_facecolor("#F8F9FA")
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        sel = (counts >= lo) & (counts < hi)
        ax.scatter(counts[sel] + 0.5, f1s[sel],
                   alpha=0.4, s=18, color=bin_colors[i], zorder=3)

    # 이동 평균
    sort_idx = np.argsort(counts)
    sc, sf = counts[sort_idx], f1s[sort_idx]
    w = 60
    if len(sc) > w:
        ma = np.convolve(sf, np.ones(w)/w, mode="valid")
        ax.plot(sc[w//2:-w//2+1] + 0.5, ma, color=DARK, linewidth=2.2,
                label=f"Moving avg (w={w})", zorder=5)

    ax.axvline(25,  color=ORANGE, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(100, color=GREEN,  linestyle="--", linewidth=1.5, alpha=0.6)
    ax.text(26,  0.05, "N=25", color=ORANGE, fontsize=8, fontweight="bold")
    ax.text(101, 0.05, "N=100", color=GREEN, fontsize=8, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Training Sample Count (log scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Per-class Micro F1",               fontsize=11, fontweight="bold")
    ax.set_title("Long-tail Analysis: Sample Count vs F1",
                 fontsize=12, fontweight="bold", color=DARK, pad=8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)

    patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(bin_colors, labels_bin)]
    ax.legend(handles=patches, fontsize=8, loc="lower right", ncol=2, framealpha=0.9)

    # ── 오른쪽: 구간별 평균 F1 박차트 ──
    ax2 = axes[1]
    ax2.set_facecolor("#F8F9FA")
    bin_means, bin_ns = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (counts >= lo) & (counts < hi)
        if sel.sum() > 0:
            bin_means.append(f1s[sel].mean())
            bin_ns.append(sel.sum())
        else:
            bin_means.append(0)
            bin_ns.append(0)

    bars = ax2.bar(labels_bin, bin_means, color=bin_colors, width=0.6,
                   edgecolor="white", linewidth=1.5, zorder=3)
    for bar, mean, n in zip(bars, bin_means, bin_ns):
        ax2.text(bar.get_x() + bar.get_width()/2, mean + 0.012,
                 f"{mean:.3f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=DARK)
        ax2.text(bar.get_x() + bar.get_width()/2, 0.02,
                 f"n={n}", ha="center", va="bottom",
                 fontsize=7.5, color="white", fontweight="bold")

    ax2.axhline(0.9384, color=PURPLE, linestyle="--", linewidth=1.8)
    ax2.text(4.4, 0.942, "Overall\n0.9384", color=PURPLE, fontsize=8, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Average Per-class F1", fontsize=11, fontweight="bold")
    ax2.set_title("Mean F1 by Training Frequency Bin",
                  fontsize=12, fontweight="bold", color=DARK, pad=8)

    fig.suptitle("Long-tail Distribution Analysis (FusionV3 30ep)",
                 fontsize=13, fontweight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4b_longtail.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("fig4b_longtail.png 저장 완료")


# ── Fig 4c: L1 클래스별 L4 F1 비교 (Ensemble vs B1) ─────────────
def fig4c_per_l1(preds_ens, labels_ens, masks_ens,
                 preds_b1,  labels_b1,  masks_b1):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    x = np.arange(7)
    w = 0.28
    ens_f1s, b1_f1s, hit_f1s = [], [], []

    # HIT-EC per-class L4 (approximated from paper if available, else None)
    hit_approx = [None]*7  # 논문에 per-class 없음 → 표시 안 함

    for cls in range(7):
        for model_name, preds, labels, masks in [
            ("ens", preds_ens, labels_ens, masks_ens),
            ("b1",  preds_b1,  labels_b1,  masks_b1),
        ]:
            # L1 cls에 속하면서 L4 유효한 샘플
            valid_l4 = (masks[:, 3] == 1) & (labels[0] == cls) & (labels[3] >= 0)
            if valid_l4.sum() < 5:
                val = 0.0
            else:
                val = f1_score(labels[3][valid_l4], preds[3][valid_l4],
                               average="micro", zero_division=0)
            if model_name == "ens":
                ens_f1s.append(val)
            else:
                b1_f1s.append(val)

    bars_ens = ax.bar(x - w/2, ens_f1s, w, color=RED,    label="Ensemble V2+V3",
                      edgecolor="white", linewidth=1.2, zorder=3)
    bars_b1  = ax.bar(x + w/2, b1_f1s,  w, color=BLUE,   label="B1 (seq-only)",
                      edgecolor="white", linewidth=1.2, zorder=3)

    for bars, vals in [(bars_ens, ens_f1s), (bars_b1, b1_f1s)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=DARK)

    short = ["Oxidored.\n(EC 1)", "Transfer.\n(EC 2)", "Hydrolase\n(EC 3)",
             "Lyase\n(EC 4)", "Isomerase\n(EC 5)", "Ligase\n(EC 6)",
             "Translocase\n(EC 7)"]
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("L4 Micro F1 (within EC class)", fontsize=11, fontweight="bold")
    ax.set_title("Per-EC-Class Performance: Ensemble vs Sequence-only (B1)",
                 fontsize=12, fontweight="bold", color=DARK, pad=10)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT / "fig4c_per_l1_f1.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("fig4c_per_l1_f1.png 저장 완료")


# ────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")

    print("\n[FusionV3 30ep] 추론 중...")
    preds_v3, labels_v3, masks_v3, encoders, n_classes = run_inference_v3(device)

    print("\n[B1] 추론 중...")
    preds_b1, labels_b1, masks_b1 = run_inference_b1(device, n_classes)

    print("\n=== Figure 생성 ===")
    fig4a_confusion(preds_v3, labels_v3, masks_v3)
    fig4b_longtail(preds_v3, labels_v3, masks_v3, encoders)
    fig4c_per_l1(preds_v3, labels_v3, masks_v3,
                 preds_b1,  labels_b1,  masks_b1)

    print(f"\n완료! → {OUT}")


if __name__ == "__main__":
    main()
