"""
Contact Map Control Experiments (Bioinformatics 수준 검증)

B3 (Contact-only)와 Contact-EC 모델에 대해 세 가지 contact map variant 평가:
  1. original     — 실제 AlphaFold contact map (기준선)
  2. shuffled     — 행/열 대칭 순열 (위상 파괴, 밀도 보존)
  3. diagonal_only— 대각선 밴드만 ① 상수 (실제 접촉 무관, 단백질 길이 프록시)
  4. random_density— 동일 밀도의 랜덤 이진 행렬

핵심 질문:
  "B3 ≈ B1 성능이 실제 구조 위상에서 오는가, 아니면 단백질 길이나 접촉 밀도에서 오는가?"

사용법:
  CUDA_VISIBLE_DEVICES=1 python scripts/shuffled_control.py
"""
import sys, pickle, json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config_ecbench.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import _make_3ch_cmap, _SHORT_MASK

DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"
EMBED_DIR = ROOT / CFG["paths"]["embed_dir"]
CMAP_DIR  = ROOT / CFG["paths"]["cmap_dir"]
LABEL_ENC = ROOT / CFG["paths"]["label_enc"]
SPLITS    = ROOT / CFG["paths"]["splits_dir"]
PROC_DIR  = ROOT / "data" / "ecbench" / "processed"

# ── Contact map variants ────────────────────────────────────────────────────

def make_shuffled(raw: np.ndarray, uid: str) -> np.ndarray:
    """행/열 대칭 순열: 위상(topology) 파괴, 접촉 밀도 보존."""
    seed = abs(hash(uid)) % (2 ** 31)
    rng  = np.random.RandomState(seed)
    perm = rng.permutation(256)
    return raw[np.ix_(perm, perm)]


def make_diagonal_only(raw: np.ndarray) -> np.ndarray:
    """대각선 밴드만 상수 1 (|i-j|<12) — 단백질별 차이 없음.
    단백질 길이/2차 구조만으로 얼마나 예측 가능한지 테스트."""
    return _SHORT_MASK.copy()  # (256,256) float32, 단백질 무관 상수


def make_random_density(raw: np.ndarray, uid: str) -> np.ndarray:
    """동일 접촉 밀도를 가진 랜덤 이진 행렬 (대칭 보장).
    접촉 밀도(길이와 상관)만으로 얼마나 예측 가능한지 테스트."""
    seed    = abs(hash(uid + "_rand")) % (2 ** 31)
    rng     = np.random.RandomState(seed)
    density = float(raw.mean())
    r       = (rng.rand(256, 256) < density).astype(np.float32)
    return np.minimum(r + r.T, 1.0)


CMAP_MODES = {
    "original":      lambda raw, uid: raw,
    "shuffled":      make_shuffled,
    "diagonal_only": lambda raw, uid: make_diagonal_only(raw),
    "random_density": make_random_density,
}

CMAP_LABELS = {
    "original":       "Original (baseline)",
    "shuffled":       "Shuffled (symmetric perm.) ← topology destroyed",
    "diagonal_only":  "Diagonal-only (const. band) ← no protein-specific info",
    "random_density": "Random density-matched     ← density only",
}


# ── Dataset ─────────────────────────────────────────────────────────────────

class ControlDataset(Dataset):
    def __init__(self, ids_file, meta_csv, n_l4, cmap_mode="original"):
        import pandas as pd
        self.ids = Path(ids_file).read_text().strip().split("\n")
        meta     = pd.read_csv(meta_csv).set_index("accession")
        self.n_l4     = n_l4
        self.cmap_mode = cmap_mode

        self.rows, self.labels = [], []
        for uid in self.ids:
            if uid not in meta.index:
                continue
            if not (EMBED_DIR / f"{uid}.npy").exists():
                continue
            row = meta.loc[uid]
            mh  = np.zeros(n_l4, dtype=np.float32)
            idxs_raw = str(row.get("l4_all_idxs", row.get("l4_idx", ""))).strip()
            if idxs_raw and idxs_raw != "nan":
                sep = "|" if "|" in idxs_raw else ","
                for s in idxs_raw.split(sep):
                    s = s.strip()
                    if s.lstrip("-").isdigit():
                        i4 = int(s)
                        if 0 <= i4 < n_l4:
                            mh[i4] = 1.0
            if mh.sum() == 0:
                l4 = int(row.get("l4_idx", -1))
                if 0 <= l4 < n_l4:
                    mh[l4] = 1.0
            if mh.sum() == 0:
                continue
            self.rows.append(uid)
            self.labels.append(mh)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        uid = self.rows[idx]
        emb = np.load(EMBED_DIR / f"{uid}.npy").astype(np.float32)

        cp  = CMAP_DIR / f"{uid}.npy"
        raw = (np.load(cp).astype(np.float32)
               if cp.exists() else np.zeros((256, 256), dtype=np.float32))
        if raw.shape != (256, 256):
            raw = np.zeros((256, 256), dtype=np.float32)

        fn      = CMAP_MODES[self.cmap_mode]
        mod_raw = fn(raw, uid)
        cmap    = _make_3ch_cmap(mod_raw)

        return torch.tensor(emb), torch.tensor(cmap), torch.tensor(self.labels[idx]), uid


def collate_fn(batch):
    embs, cmaps, labels, uids = zip(*batch)
    return torch.stack(embs), torch.stack(cmaps), torch.stack(labels), list(uids)


# ── 모델 빌더 ────────────────────────────────────────────────────────────────

def build_model(name, n_classes):
    if name == "b1_esm2_fc":
        from models.esm2_fc import ESM2FC
        return ESM2FC(n_classes, esm_dim=CFG["model"]["esm2_dim"], dropout=0.0)
    elif name == "b2_esm2_hier":
        from models.esm2_hierarchical import ESM2Hierarchical
        return ESM2Hierarchical(n_classes, esm_dim=CFG["model"]["esm2_dim"], dropout=0.0)
    elif name == "b3_contact":
        from models.contact_resnet import ContactResNet
        return ContactResNet(n_classes, dropout=0.0)
    elif name == "fusion_v2":
        from models.fusion_v2 import FusionModelV2
        return FusionModelV2(n_classes,
                             esm_dim=CFG["model"]["esm2_dim"],
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=0.0)


MODELS = {
    "B3 (Contact only)": ("b3_contact", "outputs/checkpoints/ecbench_b3_phase1_best.pt"),
    "Contact-EC":        ("fusion_v2",  "outputs/checkpoints/ecbench_fv2_phase2_best.pt"),
}


@torch.no_grad()
def run_inference(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for emb, cmap, labels, _ in tqdm(loader, desc="    추론", leave=False):
        logits = model(emb.to(DEVICE), cmap.to(DEVICE))
        all_probs.append(torch.sigmoid(logits[3]).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    with open(LABEL_ENC, "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    n_l4      = n_classes[3]
    print(f"L4 클래스: {n_l4}")

    ids_file = SPLITS  / "test_ids_full.txt"
    meta_csv = PROC_DIR / "test_meta_full.csv"

    results = {}

    for model_label, (mname, ckpt_path) in MODELS.items():
        print(f"\n{'='*65}")
        print(f"  모델: {model_label}")
        print(f"{'='*65}")

        ckpt  = torch.load(ROOT / ckpt_path, map_location=DEVICE, weights_only=False)
        model = build_model(mname, n_classes).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()

        results[model_label] = {}

        for mode, mode_label in CMAP_LABELS.items():
            ds = ControlDataset(ids_file, meta_csv, n_l4, cmap_mode=mode)
            loader = DataLoader(ds, batch_size=256, shuffle=False,
                                num_workers=4, collate_fn=collate_fn,
                                pin_memory=True)
            probs, labels = run_inference(model, loader)
            preds   = (probs >= 0.5).astype(np.int32)
            micro_f1 = float(f1_score(labels, preds, average="micro", zero_division=0))
            results[model_label][mode] = {
                "micro_f1": round(micro_f1, 4),
                "n": len(probs),
            }
            orig_f1 = results[model_label].get("original", {}).get("micro_f1", micro_f1)
            delta   = micro_f1 - orig_f1
            sign    = "+" if delta >= 0 else ""
            print(f"  {mode:<16} Micro F1={micro_f1:.4f}  "
                  f"({'baseline' if mode=='original' else f'Δ={sign}{delta:.4f}'})")
            print(f"    ({mode_label})")

    # ── 요약 테이블 ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Contact Map Control 요약 테이블")
    print(f"{'='*65}")
    header = f"  {'Mode':<20} {'B3 F1':>10} {'Contact-EC F1':>14} {'해석'}"
    print(header)
    print(f"  {'-'*62}")
    for mode, lbl in CMAP_LABELS.items():
        b3_f1   = results["B3 (Contact only)"].get(mode, {}).get("micro_f1", "—")
        fec_f1  = results["Contact-EC"].get(mode, {}).get("micro_f1", "—")
        short   = lbl.split("←")[1].strip() if "←" in lbl else lbl
        print(f"  {mode:<20} {b3_f1:>10} {fec_f1:>14}    {short}")

    print(f"\n  해석 가이드:")
    print(f"  - shuffled ≈ original  → 모델이 위상(topology) 정보를 활용하지 않음 (confound)")
    print(f"  - shuffled < original  → 실제 구조 위상이 성능에 기여")
    print(f"  - diagonal_only ≈ original → 길이/2차 구조만으로 설명됨 (confound)")
    print(f"  - diagonal_only < original → 3차 구조(long-range contacts)가 필요")

    # ── 저장 ────────────────────────────────────────────────────────────────
    out = ROOT / "outputs" / "results" / "shuffled_control.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
