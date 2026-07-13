"""
Step 1: UniProt REST API로 43,848개 신규 accession의 EC 라벨 취득
배치 크기 500, 최대 재시도 3회
출력: data/raw/hitec_new_ec.tsv  (accession, ec, sequence)
"""
import requests, time, sys, os
from pathlib import Path

ROOT     = Path("/home/user/Desktop/unlv")
IN_FILE  = ROOT / "data/raw/hitec_new_accessions.txt"
OUT_FILE = ROOT / "data/raw/hitec_new_ec.tsv"
BATCH    = 100   # 500 → URL too long (400 Bad Request), 100 works (~2200 chars)
RETRY    = 3
SLEEP    = 0.5

accessions = IN_FILE.read_text().strip().splitlines()
print(f"총 accession: {len(accessions):,}")

# 이미 처리된 것은 건너뜀 (재시작 지원)
done = set()
if OUT_FILE.exists() and OUT_FILE.stat().st_size > 100:
    import pandas as pd
    prev = pd.read_csv(OUT_FILE, sep="\t")
    done = set(prev["accession"].tolist())
    print(f"이미 처리됨: {len(done):,} → 나머지 {len(accessions)-len(done):,}개만 처리")

remaining = [a for a in accessions if a not in done]
print(f"처리할 accession: {len(remaining):,}")

URL = "https://rest.uniprot.org/uniprotkb/search"

with open(OUT_FILE, "a") as fout:
    if not done:   # 새 파일이면 헤더 작성
        fout.write("accession\tec\tsequence\n")

    total_saved = len(done)
    for i in range(0, len(remaining), BATCH):
        batch = remaining[i : i + BATCH]
        query = " OR ".join(f"accession:{a}" for a in batch)

        for attempt in range(RETRY):
            try:
                resp = requests.get(URL, params={
                    "query":  query,
                    "format": "tsv",
                    "fields": "accession,ec,sequence",
                    "size":   BATCH,
                }, timeout=60)
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt == RETRY - 1:
                    print(f"  [skip batch {i//BATCH}] error: {e}")
                    resp = None
                else:
                    time.sleep(3)

        if resp is None:
            continue

        lines = resp.text.strip().splitlines()
        # 첫 줄은 헤더 (Accession\tEC number\tSequence)
        saved = 0
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            acc, ec, seq = parts[0], parts[1], parts[2]
            if not ec.strip():   # EC 없는 단백질 제외
                continue
            fout.write(f"{acc}\t{ec}\t{seq}\n")
            saved += 1

        total_saved += saved
        progress = (i + len(batch)) / len(remaining) * 100
        print(f"  [{i+len(batch):>6}/{len(remaining):>6}] {progress:5.1f}%  "
              f"saved_this_batch={saved}  total_with_ec={total_saved}", flush=True)
        time.sleep(SLEEP)

print(f"\n완료! EC 있는 단백질: {total_saved:,}개")
print(f"출력: {OUT_FILE}")
