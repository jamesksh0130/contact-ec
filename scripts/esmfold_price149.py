"""
ESMFold으로 Price-149 단백질 구조 예측 → Contact Map 생성 → 재평가

Price-149는 NCBI RefSeq 세균 단백질로 AlphaFold DB에 없음.
ESMFold (facebook/esmfold_v1)으로 서열에서 직접 구조 예측.

Pipeline:
  Step 1: ESMFold 로드 및 149개 단백질 구조 예측
           → Cα 좌표 직접 추출 (atom37 표현의 index 1 = Cα)
  Step 2: Cα 거리 행렬 → 8Å 이진 contact map → 256×256 리사이즈
  Step 3: B3, Contact-EC 재평가 (Price-149)
  Step 4: 개선량 보고

사용법:
  CUDA_VISIBLE_DEVICES=1 python scripts/esmfold_price149.py
"""
import sys, pickle, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from scipy.ndimage import zoom
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config_ecbench.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import _make_3ch_cmap

DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"
EMBED_DIR    = ROOT / CFG["paths"]["embed_dir"]
CMAP_DIR     = ROOT / CFG["paths"]["cmap_dir"]
LABEL_ENC    = ROOT / CFG["paths"]["label_enc"]
PRICE149_CSV = ROOT / "data" / "new392" / "price149_labels.csv"

CONTACT_THR  = 8.0
CMAP_SIZE    = 256
MAX_SEQ_LEN  = 1024


# ── Step 1: ESMFold 구조 예측 ────────────────────────────────────────────────

def ca_coords_from_esmfold(seq: str, model, tokenizer) -> np.ndarray:
    """ESMFold 출력에서 Cα 좌표 추출.
    atom37 표현: index 0=N, 1=Cα, 2=C, 3=O, 4=Cβ, ...
    """
    if len(seq) > MAX_SEQ_LEN:
        seq = seq[:MAX_SEQ_LEN]

    tokenized = tokenizer(seq, return_tensors="pt",
                          add_special_tokens=False).to(DEVICE)
    with torch.no_grad():
        output = model(**tokenized)

    # positions: (n_layers, batch, seq_len, 37, 3) — 마지막 레이어 사용
    positions = output["positions"][-1]   # (1, L, 37, 3)
    ca        = positions[0, :, 1, :]    # Cα is atom index 1 → (L, 3)
    return ca.cpu().float().numpy()


def ca_to_contact_map(ca_coords: np.ndarray,
                       threshold: float = CONTACT_THR,
                       size: int = CMAP_SIZE) -> np.ndarray:
    """Cα 좌표 → 이진 contact map → 256×256 리사이즈."""
    n    = len(ca_coords)
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]   # (N, N, 3)
    dist = np.sqrt((diff ** 2).sum(-1))                     # (N, N)
    cmap = (dist < threshold).astype(np.float32)
    if n == size:
        return cmap
    s       = size / n
    resized = zoom(cmap, (s, s), order=1)
    return resized[:size, :size]


def predict_cmaps(df: pd.DataFrame):
    """Price-149 단백질 전체 ESMFold 구조 예측 → contact map 저장."""
    # 이미 있는 것 확인 (이전 zero-cmap은 제외)
    to_predict = []
    for _, row in df.iterrows():
        uid    = row["Entry"]
        cp     = CMAP_DIR / f"{uid}.npy"
        if cp.exists():
            existing = np.load(cp)
            if existing.sum() > 0:          # 실제 contact map
                continue
        to_predict.append((uid, row["Sequence"]))

    if not to_predict:
        print("[Step 1] 모든 contact map 이미 존재 — 건너뜀")
        return

    print(f"[Step 1] ESMFold 로드 중...")
    from transformers import EsmForProteinFolding, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    print("  ESMFold 가중치 로드 중... (~6분 소요)", flush=True)
    esm_fold  = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    esm_fold  = esm_fold.eval().to(DEVICE)
    print("  GPU 이동 완료", flush=True)
    print(f"  ESMFold 로드 완료 → {len(to_predict)}개 구조 예측 시작")

    ok, fail = 0, 0
    for uid, seq in tqdm(to_predict, desc="  ESMFold 예측"):
        try:
            ca   = ca_coords_from_esmfold(seq, esm_fold, tokenizer)
            cmap = ca_to_contact_map(ca)
            np.save(CMAP_DIR / f"{uid}.npy", cmap)
            ok  += 1
        except Exception as e:
            print(f"\n  ✗ {uid}: {e}")
            np.save(CMAP_DIR / f"{uid}.npy",
                    np.zeros((CMAP_SIZE, CMAP_SIZE), dtype=np.float32))
            fail += 1

    del esm_fold; torch.cuda.empty_cache()
    print(f"  → 성공: {ok}개  실패(zero cmap): {fail}개")


# ── Step 2: 평가 데이터셋 ────────────────────────────────────────────────────

class Price149Dataset(Dataset):
    def __init__(self, csv_path, l4_enc):
        df    = pd.read_csv(csv_path, sep="\t")
        l4_set = set(l4_enc.classes_)
        n_l4   = len(l4_enc.classes_)

        self.rows, self.labels = [], []
        for _, row in df.iterrows():
            uid = row["Entry"]
            if not (EMBED_DIR / f"{uid}.npy").exists():
                continue
            ecs   = [e.strip() for e in str(row["EC number"])
                     .replace(";", ",").split(",")]
            valid = [e for e in ecs if e in l4_set]
            if not valid:
                continue
            mh = np.zeros(n_l4, dtype=np.float32)
            for e in valid:
                mh[int(np.where(l4_enc.classes_ == e)[0][0])] = 1.0
            self.rows.append(uid)
            self.labels.append(mh)

        n_cmap = sum((CMAP_DIR/f"{u}.npy").exists() for u in self.rows)
        n_nonzero = 0
        for u in self.rows:
            cp = CMAP_DIR / f"{u}.npy"
            if cp.exists() and np.load(cp).sum() > 0:
                n_nonzero += 1
        print(f"  유효: {len(self.rows)}/{len(df)}  "
              f"cmap 있음: {n_cmap}  non-zero cmap: {n_nonzero}")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        uid  = self.rows[idx]
        emb  = np.load(EMBED_DIR / f"{uid}.npy").astype(np.float32)
        cp   = CMAP_DIR / f"{uid}.npy"
        if cp.exists():
            raw  = np.load(cp).astype(np.float32)
            cmap = _make_3ch_cmap(raw) if raw.shape == (256, 256) \
                   else np.zeros((3, 256, 256), dtype=np.float32)
        else:
            cmap = np.zeros((3, 256, 256), dtype=np.float32)
        return torch.tensor(emb), torch.tensor(cmap), \
               torch.tensor(self.labels[idx]), uid


def collate_fn(batch):
    embs, cmaps, labels, uids = zip(*batch)
    return torch.stack(embs), torch.stack(cmaps), torch.stack(labels), list(uids)


# ── Step 3: 모델 빌더 & 추론 ─────────────────────────────────────────────────

def build_model(name, n_classes):
    if name == "b1_esm2_fc":
        from models.esm2_fc import ESM2FC
        return ESM2FC(n_classes, esm_dim=CFG["model"]["esm2_dim"], dropout=0.0)
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
    "B1 (ESM-2)":  ("b1_esm2_fc",  "outputs/checkpoints/ecbench_b1_best.pt"),
    "B3 (Contact)":("b3_contact",   "outputs/checkpoints/ecbench_b3_phase1_best.pt"),
    "Contact-EC":  ("fusion_v2",    "outputs/checkpoints/ecbench_fv2_phase2_best.pt"),
}

# EC-Bench micro F1 기준 (이전 평가 결과)
PREV_RESULTS = {
    "B1 (ESM-2)":   0.0505,
    "B3 (Contact)": 0.0000,
    "Contact-EC":   0.1497,
    "CLEAN-Contact": 0.525,
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
    df = pd.read_csv(PRICE149_CSV, sep="\t")
    print(f"Price-149: {len(df)}개  평균 서열 길이: {df['Sequence'].str.len().mean():.0f}")

    # ── Step 1: ESMFold 구조 예측 ──────────────────────────────────────────
    predict_cmaps(df)

    # ── Step 2: 데이터셋 ───────────────────────────────────────────────────
    with open(LABEL_ENC, "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    l4_enc    = encoders["level4"]

    print("\n[Price-149 데이터셋]")
    ds     = Price149Dataset(PRICE149_CSV, l4_enc)
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # ── Step 3: 평가 ───────────────────────────────────────────────────────
    results = {}
    for label, (mname, ckpt_path) in MODELS.items():
        print(f"\n  [{label}]")
        ckpt  = torch.load(ROOT / ckpt_path, map_location=DEVICE, weights_only=False)
        model = build_model(mname, n_classes).to(DEVICE)
        model.load_state_dict(ckpt["model"])

        probs, labels_gt = run_inference(model, loader)
        preds    = (probs >= 0.5).astype(np.int32)
        micro_f1 = float(f1_score(labels_gt, preds, average="micro", zero_division=0))
        wtd_f1   = float(f1_score(labels_gt, preds, average="weighted", zero_division=0))

        prev = PREV_RESULTS.get(label, None)
        delta_str = f"  (이전: {prev:.4f}  Δ={micro_f1-prev:+.4f})" if prev is not None else ""
        print(f"    Micro F1={micro_f1:.4f}  Weighted F1={wtd_f1:.4f}{delta_str}")

        results[label] = {
            "micro_f1": round(micro_f1, 4),
            "weighted_f1": round(wtd_f1, 4),
            "prev_micro_f1": prev,
            "delta": round(micro_f1 - prev, 4) if prev is not None else None,
        }
        del model; torch.cuda.empty_cache()

    # ── 요약 ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Price-149 개선 결과 (ESMFold contact map 사용 후)")
    print(f"{'='*60}")
    print(f"  {'모델':<18}  {'이전 F1':>10}  {'새 F1':>10}  {'개선':>8}")
    print(f"  {'-'*52}")
    for lbl, r in results.items():
        prev = r['prev_micro_f1'] or 0
        delta = r['delta'] or 0
        print(f"  {lbl:<18}  {prev:>10.4f}  {r['micro_f1']:>10.4f}  {delta:>+8.4f}")
    print(f"  {'CLEAN-Contact':<18}  {'0.525':>10}  {'—':>10}  {'(참조)':>8}")

    # ── 저장 ───────────────────────────────────────────────────────────────
    out = ROOT / "outputs" / "results" / "price149_esmfold_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"n": len(ds), "results": results}, f, indent=2)
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
