"""
Contact-EC inference time benchmark.
Measures three modes:
  A) Full pipeline: raw sequence → ESM-2 → ResNet → GCA → prediction
  B) Embed-cached:  precomputed embedding + contact map → ResNet → GCA → prediction
  C) Seq-only:      precomputed embedding → FC head (no contact map)

Comparison target: HIT-EC 38ms, CLEAN 3915ms (from paper)
"""
import sys, time, json
import numpy as np
import torch
from pathlib import Path

ROOT = Path("/home/user/Desktop/unlv")
sys.path.insert(0, str(ROOT))

CKPT      = ROOT / "outputs/checkpoints/expa_flatfc_phase2_best.pt"
EMB_DIR   = ROOT / "data/processed/embeddings"
CMAP_DIR  = ROOT / "data/processed/contact_maps"
META_CSV  = ROOT / "data/expa/dataset_meta_reenc.csv"
TEST_IDS  = ROOT / "data/ecbench/splits/test_ids_full.txt"
N_CLASSES = [7, 71, 273, 5306]
N_WARMUP  = 20
N_MEASURE = 100
DEVICE    = "cuda:0"

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading Contact-EC model...")
from models.fusion_v2_flatfc import FusionV2FlatFC
model = FusionV2FlatFC(n_classes=N_CLASSES).to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE)
state = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state, strict=False)
model.eval()
print("Model loaded.")

# ── Load ESM-2 tokenizer + model for full-pipeline mode ──────────────────────
print("Loading ESM-2 650M for full-pipeline benchmark...")
from transformers import AutoTokenizer, EsmModel
esm_tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
esm_mdl = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(DEVICE)
esm_mdl.eval()
print("ESM-2 loaded.")

# ── Load test protein IDs + metadata ─────────────────────────────────────────
import pandas as pd
test_ids = [l.strip() for l in open(TEST_IDS) if l.strip()]
meta     = pd.read_csv(META_CSV, index_col=0)

# filter to proteins with both embedding and cmap
valid = []
for uid in test_ids:
    if (EMB_DIR / f"{uid}.npy").exists() and (CMAP_DIR / f"{uid}.npy").exists():
        if uid in meta.index:
            valid.append(uid)
    if len(valid) >= N_WARMUP + N_MEASURE:
        break

print(f"Valid proteins found: {len(valid)}")
if len(valid) < N_WARMUP + N_MEASURE:
    # supplement from train set
    for uid in meta.index:
        if uid not in valid:
            if (EMB_DIR / f"{uid}.npy").exists() and (CMAP_DIR / f"{uid}.npy").exists():
                valid.append(uid)
        if len(valid) >= N_WARMUP + N_MEASURE:
            break

print(f"Using {N_MEASURE} proteins for benchmark (+ {N_WARMUP} warmup)")

# ── Pre-load tensors ──────────────────────────────────────────────────────────
embs  = []
cmaps = []
seqs  = []
for uid in valid[:N_WARMUP + N_MEASURE]:
    emb  = torch.tensor(np.load(EMB_DIR  / f"{uid}.npy"), dtype=torch.float32)
    cmap = torch.tensor(np.load(CMAP_DIR / f"{uid}.npy"), dtype=torch.float32)
    if cmap.ndim == 2:
        cmap = cmap.unsqueeze(0)  # (1, H, W)
    # replicate to 3 channels if needed
    if cmap.shape[0] == 1:
        cmap = cmap.expand(3, -1, -1)
    embs.append(emb)
    cmaps.append(cmap)
    if uid in meta.index and "sequence" in meta.columns:
        seqs.append(str(meta.loc[uid, "sequence"])[:1024])
    else:
        seqs.append("MKTAYIAKQRQISFVKSHFSRQ")  # fallback sequence for timing only

# ── Benchmark helper ──────────────────────────────────────────────────────────
def measure(fn, n_warmup, n_measure):
    for i in range(n_warmup):
        fn(i)
    torch.cuda.synchronize()
    times = []
    for i in range(n_warmup, n_warmup + n_measure):
        t0 = time.perf_counter()
        fn(i)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms
    return np.array(times)

# ── Mode B: embed-cached (main deployment mode) ───────────────────────────────
print("\n=== Mode B: Embed-cached (precomputed emb + cmap) ===")
def run_cached(i):
    e = embs[i].unsqueeze(0).to(DEVICE)
    c = cmaps[i].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        model(e, c)

t_cached = measure(run_cached, N_WARMUP, N_MEASURE)
print(f"  Mean ± Std : {t_cached.mean():.2f} ± {t_cached.std():.2f} ms")
print(f"  Median     : {np.median(t_cached):.2f} ms")
print(f"  P95        : {np.percentile(t_cached, 95):.2f} ms")

# ── Mode C: seq-only (no contact map, zeros) ─────────────────────────────────
print("\n=== Mode C: Seq-only (zeros contact map) ===")
zero_cmap = torch.zeros(1, 3, 256, 256, device=DEVICE)
def run_seqonly(i):
    e = embs[i].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        model(e, zero_cmap)

t_seqonly = measure(run_seqonly, N_WARMUP, N_MEASURE)
print(f"  Mean ± Std : {t_seqonly.mean():.2f} ± {t_seqonly.std():.2f} ms")
print(f"  Median     : {np.median(t_seqonly):.2f} ms")

# ── Mode A: full pipeline (ESM-2 on-the-fly) ─────────────────────────────────
print("\n=== Mode A: Full pipeline (ESM-2 + model) ===")
def run_full(i):
    seq = seqs[i]
    inputs = esm_tok(seq, return_tensors="pt",
                     truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        out = esm_mdl(**inputs)
        emb = out.last_hidden_state[:, 0, :]     # (1, 1280)
        c   = cmaps[i].unsqueeze(0).to(DEVICE)
        model(emb, c)

t_full = measure(run_full, N_WARMUP, N_MEASURE)
print(f"  Mean ± Std : {t_full.mean():.2f} ± {t_full.std():.2f} ms")
print(f"  Median     : {np.median(t_full):.2f} ms")
print(f"  P95        : {np.percentile(t_full, 95):.2f} ms")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("INFERENCE TIME SUMMARY")
print("="*55)
print(f"  Contact-EC (embed-cached) : {t_cached.mean():.1f} ± {t_cached.std():.1f} ms")
print(f"  Contact-EC (seq-only)     : {t_seqonly.mean():.1f} ± {t_seqonly.std():.1f} ms")
print(f"  Contact-EC (full pipeline): {t_full.mean():.1f} ± {t_full.std():.1f} ms")
print(f"  HIT-EC (reported)         : 38 ms")
print(f"  CLEAN (reported)          : 3915 ms")
print("="*55)

result = {
    "cached_mean_ms":   round(float(t_cached.mean()),  2),
    "cached_std_ms":    round(float(t_cached.std()),   2),
    "cached_p95_ms":    round(float(np.percentile(t_cached, 95)), 2),
    "seqonly_mean_ms":  round(float(t_seqonly.mean()), 2),
    "full_mean_ms":     round(float(t_full.mean()),    2),
    "full_std_ms":      round(float(t_full.std()),     2),
    "full_p95_ms":      round(float(np.percentile(t_full, 95)), 2),
    "n_proteins":       N_MEASURE,
    "device":           DEVICE,
    "hitec_ms":         38,
    "clean_ms":         3915,
}
out = ROOT / "outputs/results/inference_time.json"
out.write_text(json.dumps(result, indent=2))
print(f"\nSaved: {out}")
