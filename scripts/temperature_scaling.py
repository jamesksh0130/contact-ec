"""
Temperature Scaling — 앙상블 로짓 보정 (post-hoc calibration)
Val set 로짓으로 최적 온도 T를 찾아 test set에 적용.
각 레벨별 독립 T (T1, T2, T3, T4) 사용.

사용법:
  python scripts/temperature_scaling.py
"""
import sys, pickle, yaml, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import ContactPairDataset, collate_fn_v3, ProteinDataset, collate_fn
import evaluate as ev

ENC_PATH = ROOT / "data/label_encoders.pkl"
META_CSV = ROOT / CFG["paths"]["meta_csv"]
SPLITS   = ROOT / "data/splits"
PAIR_DIR = ROOT / "data/processed/contact_pair_embs"
DEVICE   = torch.device("cuda:0")

with open(ENC_PATH, "rb") as f:
    encoders = pickle.load(f)
n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]


def collect_logits_v3(split="val", batch_size=128):
    """FusionV3 raw 로짓 수집 (softmax 전)."""
    ckpt  = torch.load(ROOT / "outputs/checkpoints/fusion_v3_phase1_best.pt",
                       map_location=DEVICE)
    model = ev.build_model("fusion_v3", n_classes).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = ContactPairDataset(
        ids_file=str(SPLITS / f"{split}_ids.txt"), meta_csv=str(META_CSV),
        embed_dir=str(ROOT / CFG["paths"]["embed_dir"]),
        cmap_dir=str(ROOT / CFG["paths"]["cmap_dir"]),
        label_enc_pkl=str(ENC_PATH), pair_emb_dir=str(PAIR_DIR),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn_v3, pin_memory=True)

    all_logits = [[], [], [], []]
    all_labels = [[], [], [], []]
    all_masks  = []

    with torch.no_grad():
        for batch in loader:
            esm, cmap, seqs, labels, masks, l4mh, accs, pair = batch
            logits = model(esm.to(DEVICE), pair.to(DEVICE))
            for i in range(4):
                all_logits[i].append(logits[i].cpu().float())
            for i in range(4):
                all_labels[i].extend(labels[:, i].tolist())
            all_masks.extend(masks.tolist())

    del model; torch.cuda.empty_cache()
    return (
        [torch.cat(all_logits[i]) for i in range(4)],
        [np.array(all_labels[i])  for i in range(4)],
        np.array(all_masks),
    )


def collect_logits_v2(split="val", batch_size=64):
    """FusionV2 raw 로짓 수집."""
    ckpt  = torch.load(ROOT / "outputs/checkpoints/fusion_v2_phase1_best.pt",
                       map_location=DEVICE)
    model = ev.build_model("fusion_v2", n_classes).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = ProteinDataset(
        ids_file=str(SPLITS / f"{split}_ids.txt"), meta_csv=str(META_CSV),
        embed_dir=str(ROOT / CFG["paths"]["embed_dir"]),
        cmap_dir=str(ROOT / CFG["paths"]["cmap_dir"]),
        label_enc_pkl=str(ENC_PATH),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    all_logits = [[], [], [], []]
    all_labels = [[], [], [], []]
    all_masks  = []

    with torch.no_grad():
        for batch in loader:
            esm, cmap, seqs, labels, masks, l4mh, accs = batch
            logits = model(esm.to(DEVICE), cmap.to(DEVICE))
            for i in range(4):
                all_logits[i].append(logits[i].cpu().float())
            for i in range(4):
                all_labels[i].extend(labels[:, i].tolist())
            all_masks.extend(masks.tolist())

    del model; torch.cuda.empty_cache()
    return (
        [torch.cat(all_logits[i]) for i in range(4)],
        [np.array(all_labels[i])  for i in range(4)],
        np.array(all_masks),
    )


def find_best_temperature(logits, labels, masks, level_idx, w2=0.6, w3=0.4,
                           logits_v2=None):
    """NLL 최소화로 최적 T 탐색 (앙상블 로짓에 적용)."""
    lv = level_idx
    valid = (masks[:, lv] == 1) & (labels[lv] >= 0)
    l_v3  = logits[lv][valid].float()
    l_v2  = logits_v2[lv][valid].float() if logits_v2 is not None else None
    lbl   = torch.tensor(labels[lv][valid], dtype=torch.long)

    def nll(T):
        T = max(T, 0.01)
        lg = w3 * l_v3 / T + (w2 * l_v2 / T if l_v2 is not None else 0)
        return F.cross_entropy(lg, lbl).item()

    res = minimize_scalar(nll, bounds=(0.1, 5.0), method="bounded")
    return res.x


def evaluate_with_temp(logits_v3, logits_v2, labels, masks, temps,
                        w2=0.6, w3=0.4, split_name=""):
    """주어진 T로 각 레벨 예측 후 Micro F1 계산."""
    results = {}
    for lv in range(4):
        valid = (masks[:, lv] == 1) & (labels[lv] >= 0)
        T = temps[lv]
        lg = w3 * logits_v3[lv][valid] / T + w2 * logits_v2[lv][valid] / T
        preds = lg.argmax(dim=1).numpy()
        lbls  = labels[lv][valid]
        micro = f1_score(lbls, preds, average="micro", zero_division=0)
        results[f"level{lv+1}"] = round(micro, 4)
    return results


def main():
    print("=== Temperature Scaling ===\n")

    # ── Val set 로짓 수집 ───────────────────────────────────
    print("[Val] FusionV3 로짓 수집 중...")
    val_logits_v3, val_labels, val_masks = collect_logits_v3("val")

    print("[Val] FusionV2 로짓 수집 중...")
    val_logits_v2, _, _ = collect_logits_v2("val")

    # ── 온도 탐색 ──────────────────────────────────────────
    print("\n최적 온도 탐색 중 (NLL 최소화)...")
    temps = []
    for lv in range(4):
        T = find_best_temperature(val_logits_v3, val_labels, val_masks, lv,
                                   logits_v2=val_logits_v2)
        temps.append(T)
        print(f"  Level {lv+1}: T* = {T:.4f}")

    # ── T=1.0 (보정 전) vs T* 비교 (Val) ──────────────────
    baseline_temps = [1.0, 1.0, 1.0, 1.0]
    val_before = evaluate_with_temp(val_logits_v3, val_logits_v2,
                                     val_labels, val_masks, baseline_temps)
    val_after  = evaluate_with_temp(val_logits_v3, val_logits_v2,
                                     val_labels, val_masks, temps)

    print("\n[Val] T=1.0 vs T* 비교:")
    for lv in range(4):
        k = f"level{lv+1}"
        print(f"  L{lv+1}: {val_before[k]:.4f} → {val_after[k]:.4f}"
              f"  (Δ={val_after[k]-val_before[k]:+.4f})")

    # ── Test set 로짓 수집 + 최종 평가 ───────────────────
    print("\n[Test] FusionV3 로짓 수집 중...")
    test_logits_v3, test_labels, test_masks = collect_logits_v3("test")

    print("[Test] FusionV2 로짓 수집 중...")
    test_logits_v2, _, _ = collect_logits_v2("test")

    test_before = evaluate_with_temp(test_logits_v3, test_logits_v2,
                                      test_labels, test_masks, baseline_temps)
    test_after  = evaluate_with_temp(test_logits_v3, test_logits_v2,
                                      test_labels, test_masks, temps)

    print("\n[Test] T=1.0 vs T* 비교:")
    for lv in range(4):
        k = f"level{lv+1}"
        print(f"  L{lv+1}: {test_before[k]:.4f} → {test_after[k]:.4f}"
              f"  (Δ={test_after[k]-test_before[k]:+.4f})")

    print(f"\n최종 [Test] L4 Micro F1:")
    print(f"  보정 전: {test_before['level4']:.4f}")
    print(f"  보정 후: {test_after['level4']:.4f}")
    print(f"  HIT-EC:  0.9300")

    # 저장
    out = {
        "optimal_temperatures": {f"level{i+1}": temps[i] for i in range(4)},
        "val":  {"before": val_before,  "after": val_after},
        "test": {"before": test_before, "after": test_after},
    }
    out_path = ROOT / "outputs/results/temperature_scaling.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
