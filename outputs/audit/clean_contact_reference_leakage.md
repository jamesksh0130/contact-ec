# CLEAN-Contact Reference-Set Leakage Audit

Checks whether our temporal test sets (SP-2023-01 N=124, SP-2024 N=1,226) have near-full-length, high-identity matches in CLEAN-Contact's public retrieval reference database (`clean_contact/clean-contact-main/data/split100_reduced.fasta`, 224,741 sequences). CLEAN-Contact predicts EC numbers via contrastive-embedding kNN search against this database, so a near-duplicate reference entry lets it recover the answer by retrieving itself rather than by generalizing.

Coverage filter: qcov>=0.8 and tcov>=0.8 (near-full-length alignments only).

| Test set | N | No covered hit | Identity >=99% | >=95% | >=90% | >=70% | >=30% |
|---|---|---|---|---|---|---|---|
| SP-2023-01 (known124) | 124 | 5 (4.0%) | 119 (96.0%) | 119 (96.0%) | 119 (96.0%) | 119 (96.0%) | 119 (96.0%) |
| SP-2024 (N=1226) | 1226 | 391 (31.9%) | 835 (68.1%) | 835 (68.1%) | 835 (68.1%) | 835 (68.1%) | 835 (68.1%) |

## Interpretation

SP-2023-01 is 96.0% covered at >=99% identity: CLEAN-Contact's reference database already contains near-exact copies of nearly all SP-2023-01 temporal test proteins, so a head-to-head evaluation on this split would mostly measure self-retrieval, not generalization.

SP-2024 shows 0% exact-accession overlap (a naive accession check would call it leakage-free) but 68.1% of its proteins have a near-full-length, >=99%-identity match in the reference database under a different accession (isoform re-registration, cross-database duplication, or near-identical homologs). Accession-only overlap checks therefore substantially understate reference-set leakage for retrieval-based baselines; a sequence-identity audit is required.

Given this, we did not run CLEAN-Contact directly on our SP-2023-01 or SP-2024 temporal splits; Table S5's CLEAN-Contact row remains the authors' own self-reported New-392/Price-149 numbers.
