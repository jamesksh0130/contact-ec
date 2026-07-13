"""
Step 3: AlphaFold DB에서 PDB 파일 병렬 다운로드
- 입력: data/processed/dataset_meta.csv
- 출력: data/raw/pdb/{accession}.pdb
- AlphaFold DB에 없는 단백질은 건너뜀 (Contact Map은 zeros 처리)
"""
import os, sys, time, yaml, requests, argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "configs" / "config.yaml") as f:
    CFG = yaml.safe_load(f)

META_CSV = ROOT / CFG["paths"]["meta_csv"]
PDB_DIR  = ROOT / CFG["paths"]["pdb_dir"]
PDB_DIR.mkdir(parents=True, exist_ok=True)

TIMEOUT  = CFG["download"]["pdb_timeout"]
# v6 (2025 최신) → v4 (2022 batch) 순으로 fallback
AF_VERSIONS = [6, 4, 5, 3]
AF_URL   = "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v{ver}.pdb"

_lock    = Lock()
_counter = {"ok": 0, "miss": 0, "err": 0, "skip": 0}


def fetch_pdb(uid: str) -> str:
    out_path = PDB_DIR / f"{uid}.pdb"
    if out_path.exists() and out_path.stat().st_size > 0:
        with _lock:
            _counter["skip"] += 1
        return "skip"

    # v6 → v4 → v5 → v3 순으로 fallback
    for ver in AF_VERSIONS:
        url = AF_URL.format(uid=uid, ver=ver)
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                out_path.write_bytes(resp.content)
                with _lock:
                    _counter["ok"] += 1
                return "ok"
            elif resp.status_code == 404:
                continue
            else:
                with _lock:
                    _counter["err"] += 1
                return f"err:{resp.status_code}"
        except Exception as e:
            with _lock:
                _counter["err"] += 1
            return f"err:{e}"

    with _lock:
        _counter["miss"] += 1
    return "miss"


def main(workers: int):
    meta = pd.read_csv(META_CSV)
    uids = meta["accession"].tolist()
    print(f"총 {len(uids):,}개 단백질 PDB 다운로드 시작 (workers={workers})")
    print(f"저장 경로: {PDB_DIR}\n")

    t0 = time.time()
    done = 0
    total = len(uids)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_pdb, uid): uid for uid in uids}
        for fut in as_completed(futures):
            done += 1
            if done % 500 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta  = (total - done) / rate if rate > 0 else 0
                c = _counter
                print(f"  [{done:>6}/{total}]  "
                      f"ok={c['ok']:,}  miss={c['miss']:,}  "
                      f"err={c['err']:,}  skip={c['skip']:,}  "
                      f"| {rate:.0f}/s  ETA {eta/60:.1f}min")

    elapsed = time.time() - t0
    c = _counter
    print(f"\n완료: {elapsed/60:.1f}분")
    print(f"  성공: {c['ok']:,}  |  AlphaFold 미존재: {c['miss']:,}  "
          f"|  오류: {c['err']:,}  |  이미존재(skip): {c['skip']:,}")

    coverage = (c['ok'] + c['skip']) / total * 100
    print(f"  PDB 커버리지: {coverage:.1f}%")

    # 누락 목록 저장
    miss_log = ROOT / "data" / "raw" / "pdb_missing.txt"
    downloaded = {p.stem for p in PDB_DIR.glob("*.pdb")}
    missing = [uid for uid in uids if uid not in downloaded]
    miss_log.write_text("\n".join(missing))
    print(f"  미존재 목록: {miss_log} ({len(missing):,}개)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int,
                        default=CFG["download"]["pdb_workers"])
    args = parser.parse_args()
    main(args.workers)
