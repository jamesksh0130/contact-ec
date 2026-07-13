"""
HIT-EC inference on EC-Bench temporal test set (N=124)
Compares directly with Contact-EC using the same test proteins and micro F1 metric.

HIT-EC model config (from Demo.ipynb):
  vocab_size=23, dimension=1024, attn_heads=2,
  output_dims=[7,72,268,4255], dropout=0.1, beta=0.59
  threshold=0.4 (sigmoid)
"""
import sys, pickle, warnings, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

ROOT    = Path("/home/user/Desktop/unlv")
HIT_EC  = ROOT / "HIT-EC"
sys.path.insert(0, str(HIT_EC))

from model.model import Model   # HIT-EC model class

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.4

# ── 1. HIT-EC 구성요소 로드 ──────────────────────────────────────────────────
print("Loading HIT-EC tokenizer and label encoder...")
with open(HIT_EC / "utils/tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

with open(HIT_EC / "utils/label_encoder.pkl", "rb") as f:
    hitec_enc = pickle.load(f)

hitec_classes = hitec_enc.classes_.tolist()   # 4255 EC strings
print(f"  HIT-EC label space: {len(hitec_classes)} L4 classes")

# ── 2. HIT-EC 모델 로드 ────────────────────────────────────────────────────
config = {
    "ah":          2,
    "dr":          0.1,
    "beta":        0.59,
    "output_dims": [7, 72, 268, 4255],
}
print("Loading HIT-EC model checkpoint...")
model = Model(config, dimension=1024, vocab_size=23)
ckpt = torch.load(HIT_EC / "utils/model.ckpt",
                  map_location=DEVICE, weights_only=False)

# PL checkpoint → state_dict 추출
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    sd = ckpt["state_dict"]
else:
    sd = ckpt

model.load_state_dict(sd, strict=False)
model.to(DEVICE)
model.eval()
print("  Model loaded successfully.")

# ── 3. 테스트 데이터 로드 ────────────────────────────────────────────────────
meta = pd.read_csv(ROOT / "data/ecbench/processed/test_meta_full.csv")
print(f"  Test proteins: {len(meta)}")

# ── 4. 토크나이징 함수 ────────────────────────────────────────────────────────
MAX_LEN = 1024
BOS_ID  = 22   # prepend BOS

def tokenize_sequence(seq):
    seq_lower = seq.lower()
    token_ids = tokenizer.texts_to_sequences([seq_lower])[0]
    token_ids = [BOS_ID] + token_ids
    token_ids = token_ids[:MAX_LEN]
    token_ids += [0] * (MAX_LEN - len(token_ids))
    return torch.tensor([token_ids], dtype=torch.long)

# ── 5. 추론 ──────────────────────────────────────────────────────────────────
# HIT-EC output layout: concat of [7, 72, 268, 4255] → total 4602
l4_start = 7 + 72 + 268   # 347
l4_end   = l4_start + 4255 # 4602

records = []
print("\nRunning HIT-EC inference on 124 test proteins...")

with torch.no_grad():
    for i, row in meta.iterrows():
        uid = row["accession"]
        seq = row["sequence"]
        ec_raw = str(row["ec_raw"])

        x = tokenize_sequence(seq).to(DEVICE)
        logits = model(x, mode="infer")          # (1, 4602)
        l4_logits = logits[0, l4_start:l4_end]   # (4255,)
        probs = torch.sigmoid(l4_logits).cpu().numpy()

        preds_idx = np.where(probs >= THRESHOLD)[0].tolist()
        if not preds_idx:
            preds_idx = [int(np.argmax(probs))]  # fallback: top-1

        pred_ecs = [hitec_classes[j] for j in preds_idx]
        true_ecs = [e.strip() for e in ec_raw.split(",")]

        records.append({
            "uid":       uid,
            "true_ecs":  true_ecs,
            "pred_ecs":  pred_ecs,
            "top1_prob": float(probs.max()),
        })

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:3d}/124]  {uid}  true={true_ecs[0]}  "
                  f"pred={pred_ecs[0]}  p={probs.max():.3f}")

# ── 6. Micro F1 계산 ─────────────────────────────────────────────────────────
print("\n=== Computing metrics ===")

all_true_ecs = set()
for r in records:
    all_true_ecs.update(r["true_ecs"])

hitec_set   = set(hitec_classes)
missing_ecs = sorted(all_true_ecs - hitec_set)

print(f"  True EC classes : {len(all_true_ecs)}")
print(f"  HIT-EC covers   : {len(all_true_ecs & hitec_set)} / {len(all_true_ecs)}")
print(f"  Novel (outside) : {len(missing_ecs)}  {missing_ecs[:8]}")

# 전체 유니언 상의 binary matrix
all_classes = sorted(hitec_set | all_true_ecs)
cls2idx     = {c: i for i, c in enumerate(all_classes)}
N = len(records)
C = len(all_classes)

y_true = np.zeros((N, C), dtype=int)
y_pred = np.zeros((N, C), dtype=int)

for i, r in enumerate(records):
    for ec in r["true_ecs"]:
        if ec in cls2idx:
            y_true[i, cls2idx[ec]] = 1
    for ec in r["pred_ecs"]:
        if ec in cls2idx:
            y_pred[i, cls2idx[ec]] = 1

micro_f1    = f1_score(y_true, y_pred, average="micro",    zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)

# Rare-EC F1 (≤25 training examples)
rare_f1 = None
our_enc_pkl = ROOT / "data/ecbench/label_encoders.pkl"
if our_enc_pkl.exists():
    import pickle as pkl2
    from collections import Counter
    our_enc    = pkl2.load(open(our_enc_pkl, "rb"))
    train_meta = pd.read_csv(ROOT / "data/ecbench/processed/train_meta.csv")
    l4_counts  = Counter()
    for row in train_meta.itertuples():
        if row.m4 and str(row.l4_idx) not in ("nan", ""):
            try:
                l4_counts[our_enc["level4"].classes_[int(row.l4_idx)]] += 1
            except Exception:
                pass
    rare_ecs  = {ec for ec, cnt in l4_counts.items() if cnt <= 25}
    rare_cols = [cls2idx[ec] for ec in rare_ecs if ec in cls2idx]
    if rare_cols:
        rare_f1 = f1_score(y_true[:, rare_cols], y_pred[:, rare_cols],
                           average="micro", zero_division=0)

# ── 7. 결과 출력 ──────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
print(f"{'Model':<28} {'Micro F1':>9} {'Weighted':>9} {'Rare-EC':>9}")
print(f"{'-'*58}")
rare_str = f"{rare_f1:.4f}" if rare_f1 is not None else "  N/A  "
print(f"{'HIT-EC (same test set)':<28} {micro_f1:>9.4f} {weighted_f1:>9.4f} {rare_str:>9}")
print(f"{'Contact-EC (ours)':<28} {'0.6032':>9} {'0.5026':>9} {'0.4219':>9}")
print(f"{'Contact-EC-Hier':<28} {'0.5690':>9} {'0.4528':>9} {'0.3621':>9}")
print(f"{'B1 (ESM-2 only)':<28} {'0.4216':>9} {'0.3178':>9} {'0.2308':>9}")
print(f"{'B3 (Contact map only)':<28} {'0.3646':>9} {'0.2734':>9} {'0.2424':>9}")
print(f"{'='*58}")

# ── 8. 저장 ───────────────────────────────────────────────────────────────────
out = {
    "model": "HIT-EC v2.0.0",
    "test_set": "EC-Bench temporal (Swiss-Prot 2023-01, N=124)",
    "threshold": THRESHOLD,
    "n_proteins": N,
    "hitec_label_space": len(hitec_classes),
    "true_ec_classes": len(all_true_ecs),
    "common_ec_classes": len(all_true_ecs & hitec_set),
    "novel_ec_missing": missing_ecs,
    "results": {
        "micro_f1":    round(micro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_f1":    round(macro_f1, 4),
        "rare_ec_f1":  round(rare_f1, 4) if rare_f1 is not None else None,
    },
    "per_protein": records,
}

out_path = ROOT / "outputs/results/hitec_eval.json"
json.dump(out, open(out_path, "w"), indent=2)
print(f"\nResults saved → {out_path}")
