"""
Step 4: PDB 파일 → Contact Map 생성
- 입력: data/raw/pdb/{accession}.pdb
- 출력: data/processed/contact_maps/{accession}.npy  (256×256 float32)
- PDB 없는 단백질은 건너뜀 (학습 시 zeros로 대체)
"""
import os, sys, yaml, time, warnings, argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

PDB_DIR  = ROOT / CFG["paths"]["pdb_dir"]
CMAP_DIR = ROOT / CFG["paths"]["cmap_dir"]
CMAP_DIR.mkdir(parents=True, exist_ok=True)

THRESH = CFG["data"]["contact_thresh"]
SIZE   = CFG["data"]["cmap_size"]


def pdb_to_contact_map(pdb_path: Path, threshold: float, size: int) -> np.ndarray:
    """Cα 원자 거리 기반 Contact Map → size×size로 리사이즈."""
    from Bio import PDB
    from scipy.ndimage import zoom

    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("p", str(pdb_path))
    except Exception:
        return None

    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id("CA"):
                    ca_coords.append(residue["CA"].get_vector().get_array())
        break  # 첫 번째 모델만

    if len(ca_coords) < 4:
        return None

    coords = np.array(ca_coords, dtype=np.float32)   # (N, 3)
    n = len(coords)

    # 거리 행렬 (N×N)
    diff  = coords[:, None, :] - coords[None, :, :]  # (N,N,3)
    dists = np.sqrt((diff ** 2).sum(-1))             # (N,N)
    cmap  = (dists < threshold).astype(np.float32)

    if n == size:
        return cmap
    if n < 4:
        return None

    scale = size / n
    cmap_r = zoom(cmap, (scale, scale), order=1)
    # 정확히 size×size
    result = np.zeros((size, size), dtype=np.float32)
    h, w = cmap_r.shape
    result[:min(h, size), :min(w, size)] = cmap_r[:size, :size]
    return result


def process_one(args):
    uid, pdb_path, threshold, size, out_path = args
    if Path(out_path).exists():
        return uid, "skip"
    cmap = pdb_to_contact_map(Path(pdb_path), threshold, size)
    if cmap is None:
        return uid, "fail"
    np.save(out_path, cmap)
    return uid, "ok"


def main(workers: int):
    pdb_files = list(PDB_DIR.glob("*.pdb"))
    print(f"PDB 파일 수: {len(pdb_files):,}")
    print(f"저장 경로: {CMAP_DIR}")
    print(f"설정: threshold={THRESH}Å, output_size={SIZE}×{SIZE}\n")

    tasks = []
    for pdb_path in pdb_files:
        uid = pdb_path.stem
        out_path = str(CMAP_DIR / f"{uid}.npy")
        tasks.append((uid, str(pdb_path), THRESH, SIZE, out_path))

    counter = {"ok": 0, "skip": 0, "fail": 0}
    done = 0
    total = len(tasks)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, t) for t in tasks]
        for fut in as_completed(futures):
            uid, status = fut.result()
            counter[status] += 1
            done += 1
            if done % 1000 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta  = (total - done) / rate if rate > 0 else 0
                print(f"  [{done:>6}/{total}]  "
                      f"ok={counter['ok']:,}  skip={counter['skip']:,}  "
                      f"fail={counter['fail']:,}  "
                      f"| {rate:.0f}/s  ETA {eta/60:.1f}min")

    elapsed = time.time() - t0
    print(f"\n완료: {elapsed/60:.1f}분")
    print(f"  성공: {counter['ok']:,}  |  이미존재: {counter['skip']:,}  "
          f"|  실패: {counter['fail']:,}")

    # 저장된 .npy 수 확인
    n_saved = len(list(CMAP_DIR.glob("*.npy")))
    print(f"  저장된 Contact Map: {n_saved:,}개")

    # 검증: 랜덤 샘플 shape 확인
    samples = list(CMAP_DIR.glob("*.npy"))[:3]
    for s in samples:
        arr = np.load(s)
        print(f"  {s.stem}: shape={arr.shape}, min={arr.min():.2f}, max={arr.max():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    main(args.workers)
