"""
전체 468개 EC-Bench 테스트셋 평가 스크립트

EC-Bench 프로토콜 적용:
  - 알려진 EC (124개): 정상 L4 multi-label 평가
  - 부분 EC (309개, L4='-'): L4 ground truth 없음 → 예측=모두 FP
  - 신규 EC (35개) : 훈련 어휘 밖 → ground truth 없음 → 예측=모두 FP

Step 1: 누락 ESM-2 임베딩 추출 (36개)
Step 2: 누락 contact map 생성 (AlphaFold PDB → 없으면 zeros) (45개)
Step 3: 4개 모델 전체 추론
Step 4: micro F1 / weighted F1 / macro F1 계산 (전체 468)
Step 5: 124 vs 468 비교 테이블 출력

사용법:
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_ecbench_full468.py
"""
import sys, pickle, json, requests, io
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.ndimage import zoom
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
with open(ROOT / "configs" / "config_ecbench.yaml") as f:
    CFG = yaml.safe_load(f)

from models.dataset import _make_3ch_cmap

DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"
EMBED_DIR = ROOT / CFG["paths"]["embed_dir"]
CMAP_DIR  = ROOT / CFG["paths"]["cmap_dir"]
LABEL_ENC = ROOT / CFG["paths"]["label_enc"]
RAW_TEST  = ROOT / "data" / "ecbench" / "raw" / "test_ec.csv"

MODELS = {
    "B1 (ESM-2)":   ("b1_esm2_fc",       "outputs/checkpoints/ecbench_b1_best.pt"),
    "B2 (Hier.FC)": ("b2_esm2_hier",     "outputs/checkpoints/ecbench_b2_hard_val_best.pt"),
    "B3 (Contact)": ("b3_contact",       "outputs/checkpoints/ecbench_b3_phase1_best.pt"),
    "B4 (GCA)":     ("fusion_v2_flatfc", "outputs/checkpoints/ecbench_b4_flatfc_best.pt"),
    "Contact-EC":   ("fusion_v2",        "outputs/checkpoints/ecbench_fv2_phase2_best.pt"),
}


# ── EC 파싱 ─────────────────────────────────────────────────────────────────

def parse_ecs(ec_raw: str):
    return [e.strip() for e in str(ec_raw).replace(";", ",").split(",") if e.strip()]


# ── Step 1: 누락 ESM-2 임베딩 추출 ─────────────────────────────────────────

def extract_missing_embeddings(df: pd.DataFrame):
    missing = [(r.id, r.seq) for _, r in df.iterrows()
               if not (EMBED_DIR / f"{r.id}.npy").exists()]
    if not missing:
        print("[Step 1] 모든 임베딩 존재 — 건너뜀")
        return

    print(f"[Step 1] 누락 임베딩 {len(missing)}개 추출 중...")
    from transformers import AutoTokenizer, EsmModel
    tokenizer = AutoTokenizer.from_pretrained(CFG["model"]["esm2_model"])
    esm2      = EsmModel.from_pretrained(CFG["model"]["esm2_model"]).eval().to(DEVICE)

    for uid, seq in tqdm(missing, desc="  ESM-2"):
        inputs = tokenizer(seq, return_tensors="pt",
                           truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            out = esm2(**inputs)
        emb = out.last_hidden_state[:, 0, :].squeeze().cpu().float().numpy()
        np.save(EMBED_DIR / f"{uid}.npy", emb)

    del esm2; torch.cuda.empty_cache()
    print(f"  → {len(missing)}개 완료")


# ── Step 2: 누락 contact map 생성 ───────────────────────────────────────────

def build_missing_cmaps(df: pd.DataFrame):
    missing_ids = [r.id for _, r in df.iterrows()
                   if not (CMAP_DIR / f"{r.id}.npy").exists()]
    if not missing_ids:
        print("[Step 2] 모든 contact map 존재 — 건너뜀")
        return

    print(f"[Step 2] 누락 contact map {len(missing_ids)}개 생성 중...")
    from Bio import PDB

    def download_pdb(uid):
        for ver in ("v4", "v3"):
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_{ver}.pdb"
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    return r.text
            except Exception:
                pass
        return None

    def pdb_to_cmap(pdb_str, threshold=8.0, size=256):
        parser = PDB.PDBParser(QUIET=True)
        struct  = parser.get_structure("p", io.StringIO(pdb_str))
        ca      = [r["CA"] for r in struct.get_residues() if r.has_id("CA")]
        if len(ca) < 3:
            return None
        coords  = np.array([a.get_vector().get_array() for a in ca])
        diff    = coords[:, None, :] - coords[None, :, :]
        dist    = np.sqrt((diff ** 2).sum(-1))
        cmap    = (dist < threshold).astype(np.float32)
        s       = size / len(ca)
        resized = zoom(cmap, (s, s), order=1)
        return resized[:size, :size]

    ok, fail = 0, 0
    for uid in tqdm(missing_ids, desc="  PDB 다운로드"):
        pdb_text = download_pdb(uid)
        if pdb_text:
            cmap = pdb_to_cmap(pdb_text)
            if cmap is not None:
                np.save(CMAP_DIR / f"{uid}.npy", cmap)
                ok += 1
                continue
        np.save(CMAP_DIR / f"{uid}.npy", np.zeros((256, 256), dtype=np.float32))
        fail += 1

    print(f"  → 성공: {ok}개  zero cmap: {fail}개")


# ── Step 3: Ground truth 라벨 생성 (전체 468) ───────────────────────────────

def build_ground_truth(df: pd.DataFrame, l4_enc) -> tuple:
    """468개 단백질의 L4 multi-hot 라벨 생성.
    - 알려진 EC: 해당 인덱스를 1로
    - 부분/신규 EC: 모두 0 (예측시 FP)
    반환: (n_l4, uid_list, gt_matrix, ec_type_list)
    """
    n_l4  = len(l4_enc.classes_)
    l4_set = set(l4_enc.classes_)

    uids, gt_rows, ec_types = [], [], []
    for _, row in df.iterrows():
        uid  = row["id"]
        ecs  = parse_ecs(row["ec_number"])

        mh       = np.zeros(n_l4, dtype=np.float32)
        ec_type  = "unknown"

        known_l4 = [e for e in ecs if "-" not in e.split(".")[-1] and e in l4_set]
        partial  = any("-" in e.split(".")[-1] for e in ecs)
        novel    = any("-" not in e.split(".")[-1] and e not in l4_set for e in ecs)

        if known_l4:
            for e in known_l4:
                idx = int(np.where(l4_enc.classes_ == e)[0][0])
                mh[idx] = 1.0
            ec_type = "known"
        elif partial:
            ec_type = "partial"
        else:
            ec_type = "novel"

        uids.append(uid)
        gt_rows.append(mh)
        ec_types.append(ec_type)

    return uids, np.array(gt_rows, dtype=np.float32), ec_types


# ── 데이터셋 ─────────────────────────────────────────────────────────────────

class FullTestDataset(Dataset):
    def __init__(self, uids):
        self.uids = uids

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        ep  = EMBED_DIR / f"{uid}.npy"
        emb = np.load(ep).astype(np.float32) if ep.exists() \
              else np.zeros(1280, dtype=np.float32)

        cp  = CMAP_DIR / f"{uid}.npy"
        if cp.exists():
            raw  = np.load(cp).astype(np.float32)
            cmap = _make_3ch_cmap(raw) if raw.shape == (256, 256) \
                   else np.zeros((3, 256, 256), dtype=np.float32)
        else:
            cmap = np.zeros((3, 256, 256), dtype=np.float32)

        return torch.tensor(emb), torch.tensor(cmap), uid


def collate_fn(batch):
    embs, cmaps, uids = zip(*batch)
    return torch.stack(embs), torch.stack(cmaps), list(uids)


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
    elif name == "fusion_v2_flatfc":
        from models.fusion_v2_flatfc import FusionV2FlatFC
        return FusionV2FlatFC(n_classes,
                              esm_dim=CFG["model"]["esm2_dim"],
                              contact_dim=CFG["model"]["resnet_out_dim"],
                              fusion_dim=CFG["model"]["fusion_dim"],
                              dropout=0.0)
    elif name == "fusion_v2":
        from models.fusion_v2 import FusionModelV2
        return FusionModelV2(n_classes,
                             esm_dim=CFG["model"]["esm2_dim"],
                             contact_dim=CFG["model"]["resnet_out_dim"],
                             fusion_dim=CFG["model"]["fusion_dim"],
                             dropout=0.0)


# ── 추론 ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, uid_order):
    """uid_order: 원하는 출력 순서(df 순서)"""
    model.eval()
    probs_dict = {}
    for emb, cmap, uids in tqdm(loader, desc="    추론", leave=False):
        logits = model(emb.to(DEVICE), cmap.to(DEVICE))
        probs  = torch.sigmoid(logits[3]).cpu().numpy()
        for uid, p in zip(uids, probs):
            probs_dict[uid] = p

    return np.array([probs_dict[uid] for uid in uid_order])


# ── 메트릭 ───────────────────────────────────────────────────────────────────

def compute_metrics(probs, gt, ec_types, thr=0.5, split_name="all"):
    preds   = (probs >= thr).astype(np.int32)
    gt_int  = gt.astype(np.int32)

    micro   = float(f1_score(gt_int, preds, average="micro",    zero_division=0))
    wtd     = float(f1_score(gt_int, preds, average="weighted", zero_division=0))
    macro   = float(f1_score(gt_int, preds, average="macro",    zero_division=0))
    prec    = float(precision_score(gt_int, preds, average="micro", zero_division=0))
    rec     = float(recall_score(gt_int, preds, average="micro",    zero_division=0))

    # 타입별 micro F1
    type_f1 = {}
    for t in ("known", "partial", "novel"):
        mask = np.array([e == t for e in ec_types])
        if mask.sum() == 0:
            type_f1[t] = float("nan")
            continue
        type_f1[t] = float(f1_score(gt_int[mask], preds[mask],
                                    average="micro", zero_division=0))

    print(f"\n  [{split_name}]  N={len(probs)}")
    print(f"    Micro F1  : {micro:.4f}")
    print(f"    Weighted F1: {wtd:.4f}")
    print(f"    Macro F1  : {macro:.4f}")
    print(f"    Precision : {prec:.4f}  Recall: {rec:.4f}")
    print(f"    타입별 — 알려진({type_f1['known']:.4f})  "
          f"부분({type_f1['partial']:.4f})  신규({type_f1['novel']:.4f})")

    return {
        "split": split_name, "n": len(probs),
        "micro_f1": round(micro, 4), "weighted_f1": round(wtd, 4),
        "macro_f1": round(macro, 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "type_known_micro_f1": round(type_f1["known"], 4)
                               if not np.isnan(type_f1["known"]) else None,
        "type_partial_micro_f1": round(type_f1["partial"], 4)
                                 if not np.isnan(type_f1["partial"]) else None,
        "type_novel_micro_f1": round(type_f1["novel"], 4)
                               if not np.isnan(type_f1["novel"]) else None,
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(RAW_TEST)
    print(f"EC-Bench 테스트셋: {len(df)}개")

    with open(LABEL_ENC, "rb") as f:
        encoders = pickle.load(f)
    n_classes = [len(encoders[f"level{i}"].classes_) for i in range(1, 5)]
    l4_enc    = encoders["level4"]
    print(f"L4 클래스: {n_classes[3]}")

    # ── Step 1: 누락 임베딩 추출 ────────────────────────────────────────────
    extract_missing_embeddings(df)

    # ── Step 2: 누락 contact map 생성 ───────────────────────────────────────
    build_missing_cmaps(df)

    # ── Step 3: Ground truth 라벨 생성 ──────────────────────────────────────
    uids, gt_all, ec_types = build_ground_truth(df, l4_enc)
    ec_counts = {t: ec_types.count(t) for t in ("known", "partial", "novel")}
    print(f"\nEC 타입: 알려진={ec_counts['known']}  "
          f"부분={ec_counts['partial']}  신규={ec_counts['novel']}")

    # 임베딩 있는 것만 최종 평가
    valid_mask   = np.array([(EMBED_DIR/f"{uid}.npy").exists() for uid in uids])
    uids_eval    = [uid for uid, v in zip(uids, valid_mask) if v]
    gt_eval      = gt_all[valid_mask]
    ec_types_eval= [t for t, v in zip(ec_types, valid_mask) if v]
    print(f"임베딩 있음: {len(uids_eval)}개 (없음: {(~valid_mask).sum()}개)")

    # ── Step 4: 데이터로더 ───────────────────────────────────────────────────
    ds     = FullTestDataset(uids_eval)
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # ── Step 5: 4개 모델 평가 ───────────────────────────────────────────────
    all_results = {}

    for model_label, (mname, ckpt_path) in MODELS.items():
        print(f"\n{'='*65}")
        print(f"  모델: {model_label}")
        ckpt  = torch.load(ROOT / ckpt_path, map_location=DEVICE, weights_only=False)
        model = build_model(mname, n_classes).to(DEVICE)
        model.load_state_dict(ckpt["model"])

        probs = run_inference(model, loader, uids_eval)

        # 전체 468 (임베딩 있는 것)
        r_all = compute_metrics(probs, gt_eval, ec_types_eval,
                                split_name=f"전체 ({len(uids_eval)}개)")

        # 알려진 EC만 (124개, 비교용)
        known_mask = np.array([t == "known" for t in ec_types_eval])
        r_known    = compute_metrics(probs[known_mask], gt_eval[known_mask],
                                     [t for t in ec_types_eval if t == "known"],
                                     split_name=f"알려진 EC ({known_mask.sum()}개)")

        all_results[model_label] = {
            "full_eval": r_all,
            "known_only": r_known,
        }
        del model; torch.cuda.empty_cache()

    # ── 요약 테이블 ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  요약: 전체 468 vs 알려진 EC 124 비교 (Micro F1)")
    print(f"{'='*65}")
    print(f"  {'모델':<18}  {'전체(468)':>10}  {'알려진(124)':>12}  {'리더보드':>10}")
    lb = {"B1 (ESM-2)":"—", "B2 (Hier.FC)":"—",
          "B3 (Contact)":"—", "B4 (GCA)":"—", "Contact-EC":"0.463~0.524"}
    for lbl, res in all_results.items():
        f_all   = res["full_eval"]["micro_f1"]
        f_known = res["known_only"]["micro_f1"]
        print(f"  {lbl:<18}  {f_all:>10.4f}  {f_known:>12.4f}  {lb[lbl]:>10}")
    print(f"\n  * 리더보드: EnzBert-learnt 0.463, Stacked 0.524 (가중 F1, 전체 468)")
    print(f"  * 우리 모델은 micro F1 기준, 리더보드는 weighted F1 기준 — 프로토콜 상이")

    # ── 저장 ────────────────────────────────────────────────────────────────
    out = ROOT / "outputs" / "results" / "eval_ecbench_full468.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"ec_type_counts": ec_counts,
                   "n_eval": len(uids_eval),
                   "results": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
