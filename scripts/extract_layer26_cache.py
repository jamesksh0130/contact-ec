"""
ESM-2 Layer 26 출력 캐시 추출 (Partial Cache for Phase 2 fine-tuning)

- 대상: 243K train + 27K val 단백질 (data/expa/splits/{train,val}_ids.txt)
- 저장: data/processed/esm2_layer26/{uid}.npy  (fp16, shape: (seq_len, 1280))
          seq_len = 실제 토큰 수 (BOS/EOS 포함, 패딩 없음)
- 완료 플래그: data/processed/esm2_layer26/DONE

수학적 동등성:
  Phase 2에서 layer 0-26은 frozen → 동일 입력에 항상 동일 출력
  → 캐싱과 실시간 계산이 수학적으로 완전히 동일

사용법:
  CUDA_VISIBLE_DEVICES=1 python scripts/extract_layer26_cache.py
  CUDA_VISIBLE_DEVICES=1 python scripts/extract_layer26_cache.py --splits train val --batch_size 32
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

ROOT     = Path("/home/user/Desktop/unlv")
OUT_DIR  = ROOT / "data/processed/esm2_layer26"
DONE_FLAG = OUT_DIR / "DONE"
ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
CACHE_LAYER = 27   # number of layers to run (0-indexed layers 0..26 → 27 layers total)


def load_ids(splits):
    ids = []
    for split in splits:
        f = ROOT / f"data/expa/splits/{split}_ids.txt"
        if f.exists():
            batch = [l.strip() for l in f.read_text().strip().splitlines() if l.strip()]
            ids.extend(batch)
            print(f"  {split}: {len(batch):,} proteins")
        else:
            print(f"  WARNING: {f} not found, skipping")
    return ids


def extract_batch(esm, tokenizer, seqs, device):
    """
    Run ESM-2 layers 0-26 and return hidden states (without padding).
    Returns: list of np.ndarray, shape (actual_seq_len, 1280), fp16
    """
    enc = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    input_ids      = enc["input_ids"]        # (B, max_len)
    attention_mask = enc["attention_mask"]   # (B, max_len), 1=valid 0=padding

    with torch.no_grad():
        # 1. Token embeddings (no position embeddings in ESM-2, RoPE is in attention)
        h = esm.embeddings(input_ids=input_ids, attention_mask=attention_mask)

        # 2. Extended attention mask: (B, 1, 1, max_len), 0 valid / -10000 padding
        extended_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0

        # 3. Run layers 0..26 only (CACHE_LAYER = 27 layers)
        for layer_module in esm.encoder.layer[:CACHE_LAYER]:
            h = layer_module(h, extended_mask)[0]

        # 4. Strip padding — save only valid tokens per sample
        results = []
        for i in range(len(seqs)):
            seq_len = int(attention_mask[i].sum().item())   # includes BOS + EOS
            cache_fp16 = h[i, :seq_len, :].cpu().half().numpy()  # (seq_len, 1280) fp16
            results.append(cache_fp16)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits",     nargs="+", default=["train", "val"],
                        help="분할 이름 목록 (예: train val)")
    parser.add_argument("--batch_size", type=int,  default=32)
    parser.add_argument("--device",     default="cuda:0",
                        help="GPU 장치 (기본: cuda:0; CUDA_VISIBLE_DEVICES 설정 시 cuda:0)")
    parser.add_argument("--overwrite",  action="store_true",
                        help="이미 추출된 캐시도 덮어쓰기")
    args = parser.parse_args()

    device = args.device
    print(f"장치: {device}")
    print(f"분할: {args.splits}")
    print(f"배치 크기: {args.batch_size}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load sequences from metadata ──────────────────────────────────────────
    import pandas as pd
    meta = pd.read_csv(ROOT / "data/expa/dataset_meta_reenc.csv").set_index("accession")
    print(f"메타데이터 로드: {len(meta):,} proteins")

    ids = load_ids(args.splits)
    print(f"총 대상: {len(ids):,} proteins")

    # Filter: skip already done (unless --overwrite)
    if not args.overwrite:
        remaining = [uid for uid in ids if not (OUT_DIR / f"{uid}.npy").exists()]
        print(f"이미 완료: {len(ids)-len(remaining):,}, 남은 것: {len(remaining):,}")
        ids = remaining

    if not ids:
        print("모든 캐시 이미 추출 완료.")
        DONE_FLAG.write_text("DONE")
        return

    # Sort by sequence length for efficient batching (minimize padding)
    def get_len(uid):
        if uid in meta.index:
            return len(str(meta.loc[uid, "sequence"]))
        return 0

    ids.sort(key=get_len)

    # ── Load ESM-2 model ──────────────────────────────────────────────────────
    print(f"\nESM-2 로드 중 ({ESM_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    esm = EsmModel.from_pretrained(ESM_MODEL, torch_dtype=torch.float32)
    esm = esm.to(device)
    esm.eval()

    # Freeze all (inference only, no gradients needed)
    for p in esm.parameters():
        p.requires_grad = False

    print(f"ESM-2 로드 완료. Layer 0-{CACHE_LAYER-1} ({CACHE_LAYER}개) 실행 예정.")
    n_param_frozen = sum(p.numel() for p in esm.parameters()) / 1e6
    print(f"모델 파라미터: {n_param_frozen:.0f}M (추론 전용)")

    # ── Extraction loop ───────────────────────────────────────────────────────
    B = args.batch_size
    n_done = 0
    n_missing_seq = 0

    for start in tqdm(range(0, len(ids), B), desc="Layer26 캐시 추출", unit="batch"):
        batch_ids = ids[start:start + B]

        # Get sequences
        batch_seqs = []
        valid_ids  = []
        for uid in batch_ids:
            if uid in meta.index:
                seq = str(meta.loc[uid, "sequence"])[:1024]
                batch_seqs.append(seq)
                valid_ids.append(uid)
            else:
                n_missing_seq += 1

        if not batch_seqs:
            continue

        try:
            caches = extract_batch(esm, tokenizer, batch_seqs, device)
        except torch.cuda.OutOfMemoryError:
            # Fallback: process one by one
            print(f"\nOOM at batch {start//B}, falling back to single-sample mode")
            torch.cuda.empty_cache()
            caches = []
            for seq in batch_seqs:
                c = extract_batch(esm, tokenizer, [seq], device)
                caches.extend(c)

        for uid, cache in zip(valid_ids, caches):
            np.save(OUT_DIR / f"{uid}.npy", cache)
        n_done += len(valid_ids)

        # Progress log every 5000 proteins
        if n_done % 5000 < B:
            total = len(ids)
            pct = n_done / total * 100
            # Estimate storage used
            gb_done = n_done * 350 * 1280 * 2 / 1e9  # rough estimate
            print(f"\n  {n_done:,}/{total:,} ({pct:.1f}%) — 예상 사용 디스크: {gb_done:.1f} GB")

    print(f"\n추출 완료: {n_done:,} proteins")
    if n_missing_seq:
        print(f"  서열 누락으로 스킵: {n_missing_seq}")

    DONE_FLAG.write_text(f"DONE\nextracted={n_done}\nmissing={n_missing_seq}\n")
    print(f"완료 플래그 저장: {DONE_FLAG}")


if __name__ == "__main__":
    main()
