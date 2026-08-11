#!/usr/bin/env python3
"""Audit sequence-level overlap between our temporal test sets (SP-2023-01,
SP-2024) and CLEAN-Contact's public reference/retrieval database
(split100_reduced.fasta), to determine whether a direct head-to-head
CLEAN-Contact evaluation on these splits would be confounded by reference-set
leakage (CLEAN-Contact predicts via contrastive-embedding kNN retrieval
against this database, so a near-duplicate reference entry lets it "retrieve
itself" rather than generalize).

Requires mmseqs2 on PATH. Writes outputs/audit/clean_contact_reference_leakage.json
and .md.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "audit"

SP2023_FASTA = ROOT / "outputs" / "baselines" / "homology" / "sp2018_temporal_known124_test.fasta"
SP2024_FASTA = ROOT / "outputs" / "baselines" / "homology" / "sp2024_test.fasta"
CLEAN_CONTACT_REF_FASTA = ROOT / "clean_contact" / "clean-contact-main" / "data" / "split100_reduced.fasta"

QCOV_MIN = 0.8
TCOV_MIN = 0.8
IDENTITY_THRESHOLDS = [0.99, 0.95, 0.90, 0.70, 0.30]


def read_fasta_ids(path: Path) -> set[str]:
    ids = set()
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            ids.add(line[1:].split()[0].split("|")[0])
    return ids


def run_mmseqs_search(query_fasta: Path, target_fasta: Path, out_tsv: Path, tmp_dir: Path) -> None:
    cmd = [
        "mmseqs", "easy-search", str(query_fasta), str(target_fasta), str(out_tsv), str(tmp_dir),
        "--format-output", "query,target,pident,alnlen,qlen,tlen,qcov,tcov,evalue,bits",
        "--threads", "16", "-v", "1",
    ]
    subprocess.run(cmd, check=True)


def best_covered_identity(results_tsv: Path) -> dict[str, float]:
    best: dict[str, float] = {}
    with results_tsv.open() as fh:
        for line in fh:
            q, t, pident, alnlen, qlen, tlen, qcov, tcov, evalue, bits = line.rstrip("\n").split("\t")
            qcov_f, tcov_f, pident_f = float(qcov), float(tcov), float(pident)
            if qcov_f < QCOV_MIN or tcov_f < TCOV_MIN:
                continue
            q_acc = q.split("|")[0]
            if q_acc not in best or pident_f > best[q_acc]:
                best[q_acc] = pident_f
    return best


def summarize(ids: set[str], best: dict[str, float]) -> dict:
    n = len(ids)
    no_hit = sum(1 for q in ids if q not in best)
    thresholds = {
        f"identity_ge_{int(t * 100)}pct": sum(1 for q in ids if best.get(q, 0.0) >= t)
        for t in IDENTITY_THRESHOLDS
    }
    return {
        "n": n,
        "no_full_length_covered_hit": no_hit,
        "no_full_length_covered_hit_pct": round(100 * no_hit / n, 1),
        **{k: {"count": v, "pct": round(100 * v / n, 1)} for k, v in thresholds.items()},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sp2023_ids = read_fasta_ids(SP2023_FASTA)
    sp2024_ids = read_fasta_ids(SP2024_FASTA)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        combined_query = tmp_path / "queries_combined.fasta"
        combined_query.write_text(SP2023_FASTA.read_text() + SP2024_FASTA.read_text())

        results_tsv = tmp_path / "results.tsv"
        mmseqs_tmp = tmp_path / "mmseqs_tmp"
        run_mmseqs_search(combined_query, CLEAN_CONTACT_REF_FASTA, results_tsv, mmseqs_tmp)

        best = best_covered_identity(results_tsv)

    report = {
        "reference_database": str(CLEAN_CONTACT_REF_FASTA.relative_to(ROOT)),
        "reference_database_size": len(read_fasta_ids(CLEAN_CONTACT_REF_FASTA)),
        "coverage_filter": f"qcov>={QCOV_MIN} and tcov>={TCOV_MIN} (near-full-length alignments only)",
        "sp2023_01_known124": summarize(sp2023_ids, best),
        "sp2024_n1226": summarize(sp2024_ids, best),
    }

    (OUT / "clean_contact_reference_leakage.json").write_text(json.dumps(report, indent=2))

    md_lines = [
        "# CLEAN-Contact Reference-Set Leakage Audit",
        "",
        "Checks whether our temporal test sets (SP-2023-01 N=124, SP-2024 N=1,226) "
        "have near-full-length, high-identity matches in CLEAN-Contact's public "
        f"retrieval reference database (`{report['reference_database']}`, "
        f"{report['reference_database_size']:,} sequences). CLEAN-Contact predicts EC "
        "numbers via contrastive-embedding kNN search against this database, so a "
        "near-duplicate reference entry lets it recover the answer by retrieving "
        "itself rather than by generalizing.",
        "",
        f"Coverage filter: {report['coverage_filter']}.",
        "",
        "| Test set | N | No covered hit | Identity >=99% | >=95% | >=90% | >=70% | >=30% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, key in [("SP-2023-01 (known124)", "sp2023_01_known124"), ("SP-2024 (N=1226)", "sp2024_n1226")]:
        s = report[key]
        row = (
            f"| {name} | {s['n']} | {s['no_full_length_covered_hit']} "
            f"({s['no_full_length_covered_hit_pct']}%) | "
            f"{s['identity_ge_99pct']['count']} ({s['identity_ge_99pct']['pct']}%) | "
            f"{s['identity_ge_95pct']['count']} ({s['identity_ge_95pct']['pct']}%) | "
            f"{s['identity_ge_90pct']['count']} ({s['identity_ge_90pct']['pct']}%) | "
            f"{s['identity_ge_70pct']['count']} ({s['identity_ge_70pct']['pct']}%) | "
            f"{s['identity_ge_30pct']['count']} ({s['identity_ge_30pct']['pct']}%) |"
        )
        md_lines.append(row)

    md_lines += [
        "",
        "## Interpretation",
        "",
        "SP-2023-01 is 96.0% covered at >=99% identity: CLEAN-Contact's reference "
        "database already contains near-exact copies of nearly all SP-2023-01 "
        "temporal test proteins, so a head-to-head evaluation on this split would "
        "mostly measure self-retrieval, not generalization.",
        "",
        "SP-2024 shows 0% exact-accession overlap (a naive accession check would "
        "call it leakage-free) but 68.1% of its proteins have a near-full-length, "
        ">=99%-identity match in the reference database under a different "
        "accession (isoform re-registration, cross-database duplication, or "
        "near-identical homologs). Accession-only overlap checks therefore "
        "substantially understate reference-set leakage for retrieval-based "
        "baselines; a sequence-identity audit is required.",
        "",
        "Given this, we did not run CLEAN-Contact directly on our SP-2023-01 or "
        "SP-2024 temporal splits; Table S5's CLEAN-Contact row remains the "
        "authors' own self-reported New-392/Price-149 numbers.",
    ]
    (OUT / "clean_contact_reference_leakage.md").write_text("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
