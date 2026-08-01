# Recency Homology and Label-Coverage Audit

This audit searches complete-known temporal EC-Bench proteins against each training split with MMseqs2 and reports nearest-neighbor identity, training label coverage, and top-hit EC agreement.

## Summary

| subset                  | cutoff       |   n |   median_top_identity |   n_identity_ge_30 |   n_identity_ge_60 |   n_identity_ge_90 |   n_identity_ge_99 |   proteins_all_l4_in_train |   top_hit_same_l4 |
|:------------------------|:-------------|----:|----------------------:|-------------------:|-------------------:|-------------------:|-------------------:|---------------------------:|------------------:|
| complete-known-124      | SP-2018      | 124 |                 0.555 |                103 |                 46 |                  9 |                  1 |                        113 |                85 |
| complete-known-124      | SP-2022      | 124 |                 0.644 |                102 |                 63 |                 18 |                  4 |                        115 |                95 |
| complete-known-124      | SP-2026-ExpA | 124 |                 0.644 |                102 |                 63 |                 18 |                  4 |                        114 |                95 |
| recency-intersection-99 | SP-2018      |  99 |                 0.556 |                 83 |                 38 |                  8 |                  1 |                         99 |                69 |
| recency-intersection-99 | SP-2022      |  99 |                 0.619 |                 83 |                 49 |                 14 |                  3 |                         99 |                78 |
| recency-intersection-99 | SP-2026-ExpA |  99 |                 0.619 |                 83 |                 49 |                 14 |                  3 |                         99 |                78 |

## Interpretation

- Increased ExpA performance should be interpreted together with any change in nearest-neighbor identity and training label coverage.

- If newer cutoffs show higher identity or more labels observed in training, the gain is not a pure architectural or pure calendar-time effect; it also reflects corpus expansion, vocabulary coverage, and homolog availability.

- This audit does not replace fold-disjoint evaluation. It is a sequence-level coverage control that should be reported alongside Foldseek/TM-align split results.


## Files

- `recency_homology_coverage_per_protein.csv`

- `recency_homology_coverage_summary.csv`

- cached MMseqs2 result TSVs: `recency_homology_mmseqs_*.tsv`


Generated rows: 372 per-protein cutoff records.
