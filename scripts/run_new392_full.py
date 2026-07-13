"""
New-392 전체 평가 파이프라인
1) 누락 ESM-2 임베딩 추출 (I1S2M5)
2) 누락 contact map 생성 (AlphaFold PDB 다운로드 → cmap)
3) 4개 모델 모두 평가 (B1, B2, B3, Contact-EC)
"""
import sys, pickle, json, requests, io
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.dataset import _make_3ch_cmap

import yaml
with open(ROOT / "configs" / "config_ecbench.yaml") as f:
    CFG = yaml.safe_load(f)

EMBED_DIR = ROOT / "data" / "processed" / "embeddings"
CMAP_DIR  = ROOT / "data" / "processed" / "contact_maps"
NEW392_CSV   = ROOT / "data" / "new392" / "new392_ec_labels.csv"
PRICE149_CSV = ROOT / "data" / "new392" / "price149_labels.csv"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CLEAN_REF = {
    "new392": {"CLEAN-Contact": 0.566, "CLEAN (P-value)": 0.504,
               "ProteInfer": 0.309, "DeepEC": 0.230},
    "price149": {"CLEAN-Contact": 0.525, "CLEAN (P-value)": 0.452},
}

MODELS = {
    "B1 (ESM-2)":    ("b1_esm2_fc",    "outputs/checkpoints/ecbench_b1_best.pt"),
    "B2 (Hier. FC)": ("b2_esm2_hier",  "outputs/checkpoints/ecbench_b2_hard_val_best.pt"),
    "B3 (Contact)":  ("b3_contact",    "outputs/checkpoints/ecbench_b3_phase1_best.pt"),
    "Contact-EC":    ("fusion_v2",     "outputs/checkpoints/ecbench_fv2_phase2_best.pt"),
}


# ── Step 1: 누락 임베딩 추출 ─────────────────────────────────────────────────
def extract_missing_embeddings(missing_ids, seq_map):
    from transformers import AutoTokenizer, EsmModel
    print(f"\n[Step 1] ESM-2 임베딩 추출: {missing_ids}")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model"]["esm2_model"])
    esm2      = EsmModel.from_pretrained(CFG["model"]["esm2_model"]).eval().to(DEVICE)
    for uid in missing_ids:
        seq = seq_map[uid]
        inputs = tokenizer(seq, return_tensors="pt",
                           truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            out = esm2(**inputs)
        emb = out.last_hidden_state[:, 0, :].squeeze().cpu().float().numpy()
        np.save(EMBED_DIR / f"{uid}.npy", emb)
        print(f"  ✓ {uid}  shape={emb.shape}")
    del esm2; torch.cuda.empty_cache()


# ── Step 2: 누락 contact map 생성 ────────────────────────────────────────────
def build_missing_cmaps(missing_ids):
    from Bio import PDB
    from scipy.ndimage import zoom

    print(f"\n[Step 2] contact map 생성: {len(missing_ids)}개")

    def pdb_to_cmap(pdb_str, threshold=8.0, size=256):
        parser = PDB.PDBParser(QUIET=True)
        struct = parser.get_structure("p", io.StringIO(pdb_str))
        ca = [r["CA"] for r in struct.get_residues() if r.has_id("CA")]
        if len(ca) < 3:
            return None
        coords = np.array([a.get_vector().get_array() for a in ca])
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff**2).sum(-1))
        cmap = (dist < threshold).astype(np.float32)
        s = size / len(ca)
        resized = zoom(cmap, (s, s), order=1)
        return resized[:size, :size]

    def download_pdb(uid):
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return r.text
        except Exception:
            pass
        # fallback v3
        url2 = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v3.pdb"
        try:
            r = requests.get(url2, timeout=30)
            if r.status_code == 200:
                return r.text
        except Exception:
            pass
        return None

    ok, fail = 0, []
    for uid in tqdm(missing_ids, desc="  PDB 다운로드"):
        pdb_text = download_pdb(uid)
        if pdb_text is None:
            print(f"  ✗ {uid}: AlphaFold PDB 없음 → zero cmap")
            fail.append(uid)
            np.save(CMAP_DIR / f"{uid}.npy",
                    np.zeros((256, 256), dtype=np.float32))
            continue
        cmap = pdb_to_cmap(pdb_text)
        if cmap is None:
            print(f"  ✗ {uid}: PDB 파싱 실패 → zero cmap")
            fail.append(uid)
            np.save(CMAP_DIR / f"{uid}.npy",
                    np.zeros((256, 256), dtype=np.float32))
        else:
            np.save(CMAP_DIR / f"{uid}.npy", cmap)
            ok += 1
    print(f"  완료: {ok}개 성공, {len(fail)}개 zero cmap 처리: {fail}")


# ── 데이터셋 ─────────────────────────────────────────────────────────────────
class New392Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, l4_enc):
        df = pd.read_csv(csv_path, sep="\t")
        l4_set = set(l4_enc.classes_)
        n_l4   = len(l4_enc.classes_)

        self.rows, self.labels = [], []
        self.missing_emb = []
        for _, row in df.iterrows():
            uid = row["Entry"]
            ecs = [e.strip() for e in str(row["EC number"]).replace(";",",").split(",")]
            if not (EMBED_DIR / f"{uid}.npy").exists():
                self.missing_emb.append(uid)
                continue
            valid = [e for e in ecs if e in l4_set]
            if not valid:
                continue
            mh = np.zeros(n_l4, dtype=np.float32)
            for e in valid:
                idx = int(np.where(l4_enc.classes_ == e)[0][0])
                mh[idx] = 1.0
            self.rows.append(uid)
            self.labels.append(mh)

        n_cmap = sum((CMAP_DIR/f"{uid}.npy").exists() for uid in self.rows)
        print(f"  유효: {len(self.rows)}/{len(df)}  cmap 있음: {n_cmap}")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        uid  = self.rows[idx]
        emb  = np.load(EMBED_DIR/f"{uid}.npy").astype(np.float32)
        cp   = CMAP_DIR/f"{uid}.npy"
        if cp.exists():
            raw = np.load(cp).astype(np.float32)
            cmap = _make_3ch_cmap(raw) if raw.shape == (256, 256) \
                   else np.zeros((3,256,256), dtype=np.float32)
        else:
            cmap = np.zeros((3,256,256), dtype=np.float32)
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
        return FusionModelV2(n_classes, esm_dim=CFG["model"]["esm2_dim"],
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"], dropout=0.0)


# ── 추론 ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for emb, cmap, labels, _ in tqdm(loader, desc="  추론", leave=False):
        logits = model(emb.to(DEVICE), cmap.to(DEVICE))
        all_probs.append(torch.sigmoid(logits[3]).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ── per-protein F1 (CLEAN 방식) ──────────────────────────────────────────────
def per_protein_f1(probs, labels, thr=0.5):
    scores = []
    top1_ok = 0
    for p, gt in zip(probs, labels):
        top1 = int(np.argmax(p))
        if gt[top1] == 1.0:
            top1_ok += 1
        pred = (p >= thr).astype(float)
        tp   = float((pred * gt).sum())
        np_  = float(pred.sum())
        ng   = float(gt.sum())
        if np_ == 0 and ng == 0:
            scores.append(1.0)
        elif np_ == 0 or ng == 0:
            scores.append(0.0)
        else:
            prec = tp / np_
            rec  = tp / ng
            scores.append(2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0)
    return float(np.mean(scores)), top1_ok / len(probs)


# ── 메트릭 출력 ─────────────────────────────────────────────────────────────
def compute_metrics(probs, labels, split, ref=None):
    preds = (probs >= 0.5).astype(np.int32)
    micro = f1_score(labels, preds, average="micro",    zero_division=0)
    wtd   = f1_score(labels, preds, average="weighted", zero_division=0)
    prec  = precision_score(labels, preds, average="micro", zero_division=0)
    rec   = recall_score(labels, preds, average="micro",    zero_division=0)
    pp05, top1 = per_protein_f1(probs, labels, 0.5)
    pp01, _    = per_protein_f1(probs, labels, 0.1)

    print(f"    Micro F1={micro:.4f}  Weighted={wtd:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    print(f"    Per-protein F1: thr=0.5 → {pp05:.4f}  thr=0.1 → {pp01:.4f}  Top-1={top1:.4f}")
    if ref:
        for name, val in ref.items():
            diff = pp05 - val
            print(f"      vs {name:<22}: {val:.3f}  ({'+'if diff>=0 else ''}{diff:.4f})")

    return {"split": split, "n": len(probs),
            "micro_f1": round(float(micro), 4),
            "weighted_f1": round(float(wtd), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "per_protein_f1_05": round(float(pp05), 4),
            "per_protein_f1_01": round(float(pp01), 4),
            "top1_accuracy": round(float(top1), 4)}


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    # 라벨 인코더
    with open(ROOT / CFG["paths"]["label_enc"], "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    l4_enc    = encoders["level4"]
    print(f"L4 클래스: {n_classes[3]}")

    # 누락 임베딩 확인
    df392 = pd.read_csv(NEW392_CSV, sep="\t")
    missing_emb = [r["Entry"] for _,r in df392.iterrows()
                   if not (EMBED_DIR/f"{r['Entry']}.npy").exists()]
    if missing_emb:
        seq_map = dict(zip(df392["Entry"], df392["Sequence"]))
        extract_missing_embeddings(missing_emb, seq_map)

    # 누락 cmap 확인
    missing_cmap = [r["Entry"] for _,r in df392.iterrows()
                    if not (CMAP_DIR/f"{r['Entry']}.npy").exists()]
    if missing_cmap:
        build_missing_cmaps(missing_cmap)

    # 데이터로더 준비
    print("\n[New-392 데이터셋]")
    ds392 = New392Dataset(NEW392_CSV, l4_enc)
    loader392 = DataLoader(ds392, batch_size=256, shuffle=False,
                           num_workers=4, collate_fn=collate_fn, pin_memory=True)

    print("\n[Price-149 데이터셋]")
    ds149 = New392Dataset(PRICE149_CSV, l4_enc)
    loader149 = DataLoader(ds149, batch_size=256, shuffle=False,
                           num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # 4개 모델 평가
    all_results = {}
    for label, (mname, ckpt_path) in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  모델: {label}  ({mname})")
        ckpt  = torch.load(ROOT/ckpt_path, map_location=DEVICE, weights_only=False)
        model = build_model(mname, n_classes).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        print(f"  (val micro_f1={ckpt.get('micro_f1', '?'):.4f})" if isinstance(ckpt.get('micro_f1'), float) else "")
        print(f"{'='*60}")

        print("  [New-392]")
        p392, l392 = run_inference(model, loader392)
        r392 = compute_metrics(p392, l392, "new392", CLEAN_REF["new392"])

        print("  [Price-149]")
        p149, l149 = run_inference(model, loader149)
        r149 = compute_metrics(p149, l149, "price149", CLEAN_REF["price149"])

        all_results[label] = {"new392": r392, "price149": r149}

    # 요약 테이블
    print(f"\n{'='*60}")
    print("  요약: Per-protein F1 (thr=0.5) vs CLEAN 비교")
    print(f"{'='*60}")
    print(f"  {'모델':<18}  {'New-392':>10}  {'Price-149':>10}")
    print(f"  {'CLEAN-Contact':<18}  {'0.566':>10}  {'0.525':>10}")
    print(f"  {'CLEAN (P-value)':<18}  {'0.504':>10}  {'0.452':>10}")
    print(f"  {'-'*44}")
    for label, res in all_results.items():
        n392 = res["new392"]["per_protein_f1_05"]
        n149 = res["price149"]["per_protein_f1_05"]
        print(f"  {label:<18}  {n392:>10.4f}  {n149:>10.4f}")

    # 저장
    out = ROOT / "outputs" / "results" / "new392_full_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"reference": CLEAN_REF, "results": all_results},
                  f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
