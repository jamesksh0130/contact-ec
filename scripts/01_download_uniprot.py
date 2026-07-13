"""
Step 1: UniProt Swiss-Prot 다운로드
- 조건: reviewed=true, EC number 존재
- 출력: data/raw/swissprot_ec.tsv
- 예상 크기: ~230k 행
"""
import os, sys, time, requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "data" / "raw" / "swissprot_ec.tsv"
OUT.parent.mkdir(parents=True, exist_ok=True)

UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=reviewed:true+AND+ec:*"
    "&format=tsv"
    "&fields=accession,id,sequence,ec,length,organism_id,organism_name"
)

def download():
    print("UniProt Swiss-Prot (EC 있는 항목) 다운로드 시작...")
    print(f"저장 경로: {OUT}")

    resp = requests.get(UNIPROT_URL, stream=True, timeout=120)
    if resp.status_code != 200:
        print(f"오류: HTTP {resp.status_code}")
        sys.exit(1)

    total_bytes = 0
    n_lines = 0
    t0 = time.time()

    with open(OUT, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            total_bytes += len(chunk)
            n_lines += chunk.count(b"\n")
            elapsed = time.time() - t0
            mb = total_bytes / 1e6
            print(f"\r  {mb:.1f} MB  |  ~{n_lines:,} rows  |  {elapsed:.0f}s", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n완료: {total_bytes/1e6:.1f} MB, {elapsed:.1f}s")
    print(f"저장: {OUT}")

    # 빠른 검증
    import pandas as pd
    df = pd.read_csv(OUT, sep="\t", nrows=5)
    print("\n컬럼:", list(df.columns))
    print(df[["Entry", "EC number", "Length"]].head())

if __name__ == "__main__":
    download()
