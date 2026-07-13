"""
EC-Bench Step 4: 새 단백질 AlphaFold PDB 다운로드

data/ecbench/new_proteins.txt 중 기존 contact map이 없는 것만 다운로드.
대부분은 이미 data/raw/pdb/ 에 존재.
"""
import requests, time, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

ROOT     = Path("/home/user/Desktop/unlv")
PDB_DIR  = ROOT / "data/raw/pdb"
CMAP_DIR = ROOT / "data/processed/contact_maps"
ECBENCH  = ROOT / "data/ecbench"

WORKERS  = 20
TIMEOUT  = 30
AF_URL   = "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"


def download_one(uid):
    out_path = PDB_DIR / f"{uid}.pdb"
    if out_path.exists():
        return uid, "exists"
    url = AF_URL.format(uid=uid)
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            out_path.write_bytes(r.content)
            return uid, "ok"
        return uid, f"http_{r.status_code}"
    except Exception as e:
        return uid, f"err_{str(e)[:30]}"


def main():
    # new_proteins.txt에서 contact map 없는 것만
    new_ids = set((ECBENCH / "new_proteins.txt").read_text().strip().split("\n"))
    existing_cmap = set(p.stem for p in CMAP_DIR.glob("*.npy"))
    existing_pdb  = set(p.stem for p in PDB_DIR.glob("*.pdb"))
    to_download   = [uid for uid in sorted(new_ids)
                     if uid not in existing_cmap and uid not in existing_pdb]
    print(f"새 PDB 다운로드 필요: {len(to_download):,}개")
    if not to_download:
        print("모두 이미 존재 — 종료")
        return

    ok, miss, err = 0, 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(download_one, uid): uid for uid in to_download}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="PDB 다운로드"):
            uid, status = fut.result()
            if status == "ok":
                ok += 1
            elif status.startswith("http_404"):
                miss += 1
            else:
                err += 1

    print(f"\n완료: 성공={ok}, AlphaFold없음={miss}, 오류={err}")


if __name__ == "__main__":
    main()
