"""
Step 6: Contact-Pair ESM-2 임베딩 추출
- 각 단백질에서 top-K long-range contact (i,j) 쌍 선택
- 해당 잔기 위치의 ESM-2 residue token (1280-dim) 추출
- 출력: data/processed/contact_pair_embs/{acc}.npy  (K, 2, 1280) float16

핵심 아이디어 (FusionV3):
  contact(i,j) → token_i + token_j 직접 참조
  = 위치 대응이 살아있는 상태에서 서열+구조 통합
"""
import os, sys, time, yaml, argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

META_CSV    = ROOT / CFG["paths"]["meta_csv"]
CMAP_DIR    = ROOT / CFG["paths"]["cmap_dir"]
OUT_DIR     = ROOT / "data" / "processed" / "contact_pair_embs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN     = CFG["data"]["max_seq_len"]   # 1024
ESM_MODEL   = CFG["model"]["esm2_model"]
CMAP_SIZE   = 256
K_PAIRS     = 32    # 저장할 contact pair 수
LR_THRESH   = 12    # long-range: |i-j| >= 12 (256-grid 기준)


# ── Contact pair 선택 ─────────────────────────────────────────
def select_top_k_pairs(cmap: np.ndarray, k: int, lr_thresh: int) -> np.ndarray:
    """
    256×256 binary contact map → top-k long-range contact 위치 반환
    Returns: (k, 2) int array  [[r0,c0], [r1,c1], ...]  (256-grid 좌표)
    """
    H, W = cmap.shape
    # long-range mask: |i-j| >= lr_thresh
    idx  = np.arange(H)
    diff = np.abs(idx[:, None] - idx[None, :])
    lr_mask = (diff >= lr_thresh)

    # contact이면서 long-range인 위치 (상삼각형만 — 중복 제거)
    upper = np.triu(np.ones((H, W), dtype=bool), k=1)
    valid = cmap.astype(bool) & lr_mask & upper
    positions = np.argwhere(valid)   # (N_valid, 2)

    if len(positions) == 0:
        # long-range contact 없으면 all contact fallback
        positions = np.argwhere(np.triu(cmap.astype(bool), k=1))

    if len(positions) == 0:
        # contact 자체가 없으면 zero-padded coordinates 반환
        return np.zeros((k, 2), dtype=np.int32)

    # 균일 샘플링 (재현성: seed 고정)
    rng = np.random.default_rng(seed=42)
    if len(positions) >= k:
        chosen = rng.choice(len(positions), k, replace=False)
    else:
        # 부족하면 반복 채우기
        chosen = rng.choice(len(positions), k, replace=True)

    return positions[chosen].astype(np.int32)   # (k, 2)


def grid_to_residue(grid_pos: int, seq_len: int, grid_size: int = 256) -> int:
    """256-grid 위치 → 실제 잔기 인덱스 (0-based)"""
    r = int(grid_pos * seq_len / grid_size)
    return min(r, seq_len - 1)


# ── Dataset ───────────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, records):
        # records: [(acc, seq, seq_len), ...]
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def collate_fn(batch):
    accs     = [b[0] for b in batch]
    seqs     = [b[1] for b in batch]
    seq_lens = [b[2] for b in batch]
    return accs, seqs, seq_lens


# ── 메인 ──────────────────────────────────────────────────────
def main(batch_size: int, gpu: int, workers: int, shard: int = 0, n_shards: int = 1):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device}")

    meta = pd.read_csv(META_CSV)
    meta = meta[meta["sequence"].notna() & (meta["sequence"].str.len() >= 4)]

    # 이미 처리된 것 제외
    existing = {p.stem for p in OUT_DIR.glob("*.npy")}
    todo = [
        (row["accession"], row["sequence"], min(len(row["sequence"]), MAX_LEN))
        for _, row in meta.iterrows()
        if row["accession"] not in existing
        and (CMAP_DIR / f"{row['accession']}.npy").exists()
    ]

    # 샤딩: 작업을 n_shards로 나눠 병렬 GPU에서 분담
    if n_shards > 1:
        todo = [t for i, t in enumerate(todo) if i % n_shards == shard]
        print(f"샤드 {shard}/{n_shards}: {len(todo):,}개 담당")

    print(f"총 {len(meta):,}개  기존={len(existing):,}  남은={len(todo):,}")
    print(f"출력 디렉토리: {OUT_DIR}")
    print(f"K={K_PAIRS} pairs/protein  |  저장 예상: {len(todo)*K_PAIRS*2*1280*2/1e9:.1f}GB (float16)")

    if not todo:
        print("모든 pair 임베딩이 이미 존재합니다.")
        return

    print(f"\nESM-2 로드: {ESM_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)
    esm_model = EsmModel.from_pretrained(ESM_MODEL)
    esm_model.eval().to(device)

    ds     = PairDataset(todo)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, collate_fn=collate_fn)

    t0   = time.time()
    done = 0
    total = len(todo)

    for batch_accs, batch_seqs, batch_seq_lens in loader:
        # ESM-2 forward → residue tokens
        inputs = tokenizer(
            list(batch_seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(device)

        with torch.no_grad():
            out = esm_model(**inputs)

        # last_hidden_state: (B, L_pad+2, 1280) — [BOS, res1, ..., resL, EOS, PAD...]
        hidden = out.last_hidden_state.cpu().float().numpy()  # (B, seq+2, 1280)

        for b_idx, (acc, seq_len) in enumerate(zip(batch_accs, batch_seq_lens)):
            # 잔기 토큰: index 1 ~ seq_len (BOS 제외, EOS/PAD 제외)
            res_tokens = hidden[b_idx, 1:seq_len + 1, :]   # (seq_len, 1280)

            # contact map 로드
            cmap_path = CMAP_DIR / f"{acc}.npy"
            cmap = np.load(str(cmap_path)).astype(np.float32)  # (256,256)

            # top-K long-range contact 위치 (256-grid)
            pairs_grid = select_top_k_pairs(cmap, K_PAIRS, LR_THRESH)  # (K,2)

            # 256-grid → 잔기 인덱스
            pair_tokens = np.zeros((K_PAIRS, 2, 1280), dtype=np.float16)
            for k, (gi, gj) in enumerate(pairs_grid):
                ri = grid_to_residue(gi, seq_len)
                rj = grid_to_residue(gj, seq_len)
                if ri < len(res_tokens) and rj < len(res_tokens):
                    pair_tokens[k, 0] = res_tokens[ri].astype(np.float16)
                    pair_tokens[k, 1] = res_tokens[rj].astype(np.float16)

            np.save(str(OUT_DIR / f"{acc}.npy"), pair_tokens)
            done += 1

        elapsed = time.time() - t0
        speed   = done / elapsed
        eta     = (total - done) / max(speed, 1e-6)
        print(f"  [{done:>6}/{total:>6}]  {speed:.1f} seq/s  ETA {eta/3600:.1f}h", end="\r")

    print(f"\n완료! 총 {done:,}개  ({time.time()-t0:.0f}s)")
    print(f"저장 경로: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--shard",      type=int, default=0)
    parser.add_argument("--n_shards",   type=int, default=1)
    args = parser.parse_args()
    main(args.batch_size, args.gpu, args.workers, args.shard, args.n_shards)
