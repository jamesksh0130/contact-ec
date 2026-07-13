"""
EC-Bench Step 1: Swiss-Prot 2018-02 파싱 → 훈련 데이터 구성

입력: data/ecbench/raw/uniprot_sprot_2018_02.tar.gz
출력:
  data/ecbench/raw/swissprot_2018_02.tsv   (accession, sequence, ec_number)
  data/ecbench/raw/train_ids_raw.txt        (EC 있는 accession 목록)
"""
import tarfile, gzip, re, csv, os
from pathlib import Path

RAW_DIR  = Path("/home/user/Desktop/unlv/data/ecbench/raw")
TAR_PATH = RAW_DIR / "uniprot_sprot_2018_02.tar.gz"
OUT_TSV  = RAW_DIR / "swissprot_2018_02.tsv"


def parse_sprot_dat(gz_stream):
    """uniprot_sprot.dat.gz → (accession, sequence, ec_list) 제너레이터."""
    acc, seq_lines, ec_set = None, [], set()
    in_seq = False

    for raw in gz_stream:
        line = raw.decode("utf-8", errors="ignore")
        tag = line[:2]

        if tag == "ID":
            acc, seq_lines, ec_set, in_seq = None, [], set(), False

        elif tag == "AC":
            if acc is None:
                # 첫 번째 AC 줄의 첫 번째 accession
                acc = line[5:].strip().rstrip(";").split(";")[0].strip()

        elif tag == "DE":
            # EC=1.1.1.1 패턴
            for m in re.finditer(r"EC=(\d+\.\d+\.[\d-]+\.[\d-]+)", line):
                ec_set.add(m.group(1))

        elif tag == "  ":   # sequence 줄 (공백 2개)
            seq_lines.append(line[5:].replace(" ", "").strip())

        elif line.startswith("//"):
            if acc and ec_set:
                seq = "".join(seq_lines)
                yield acc, seq, sorted(ec_set)
            acc, seq_lines, ec_set, in_seq = None, [], set(), False


def main():
    print(f"파싱 중: {TAR_PATH}")
    if not TAR_PATH.exists():
        raise FileNotFoundError(f"다운로드 완료 후 실행하세요: {TAR_PATH}")

    count = 0
    with open(OUT_TSV, "w", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(["accession", "sequence", "ec_number", "seq_len"])

        with tarfile.open(TAR_PATH, "r:gz") as tar:
            # uniprot_sprot.dat.gz 멤버 찾기
            dat_member = None
            for m in tar.getmembers():
                if "uniprot_sprot.dat" in m.name:
                    dat_member = m
                    break
            if dat_member is None:
                raise RuntimeError("uniprot_sprot.dat.gz 파일을 tar에서 찾을 수 없음")
            print(f"발견: {dat_member.name}")

            f = tar.extractfile(dat_member)
            # .dat.gz면 gzip으로 한 번 더 압축 해제
            if dat_member.name.endswith(".gz"):
                f = gzip.open(f, "rb")

            for acc, seq, ec_list in parse_sprot_dat(f):
                ec_str = "; ".join(ec_list)
                writer.writerow([acc, seq, ec_str, len(seq)])
                count += 1
                if count % 10000 == 0:
                    print(f"  {count:,}개 처리 중...")

    print(f"\n완료: {count:,}개 → {OUT_TSV}")


if __name__ == "__main__":
    main()
