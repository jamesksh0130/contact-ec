# Contact-EC Reliability Audit

Generated from `scripts/audit_reliability.py`.

## Executive Summary

Current evidence is promising, but several claims in the paper need tighter wording or additional validation.

Highest-risk issues:

1. **New-392 is not clean external under the current 2026 Swiss-Prot dataset.**
   - New-392 size: 389.
   - Accession overlap with current random train: 307 / 389.
   - Exact sequence overlap with current random train: 303 / 389.
   - Accession overlap with full current metadata: 389 / 389.
   - Therefore, the paper should not call New-392 a clean time-stratified external benchmark unless the model used for that result was trained with an older cutoff excluding these proteins.

2. **Random split has exact sequence duplicates across train/test.**
   - Random train/test accession overlap: 0.
   - Random train/test exact sequence overlap: 4,694.
   - This weakens random-test claims and makes cluster/EC-Bench-style evaluation more important.

3. **Cluster split is internally disjoint, but must be matched with cluster training.**
   - `cluster_train` vs `cluster_test` accession overlap: 0.
   - `cluster_train` vs `cluster_test` exact sequence overlap: 0.
   - However, `cluster_test` overlaps heavily with the random `train` split:
     - accession overlap: 24,408.
     - exact sequence overlap: 20,964.
   - Therefore, cluster-test results are only valid OOD evidence if the evaluated checkpoint was trained on `cluster_train`, not random `train`.

4. **Price-149 lacks contact maps in the current pipeline.**
   - Price-149 embedding coverage: 145 / 145.
   - Price-149 contact map coverage: 0 / 145.
   - This supports the paper's limitation claim, but Price-149 should be presented as a failure-mode benchmark, not as evidence for structure-aware generalization.

5. **EC-Bench hard validation is a better current reliability target.**
   - EC-Bench train/val/test accession overlaps are clean, except `val_easy` and `val_hard` are subsets of `val`.
   - `val_hard` size: 6,247.
   - `val_hard` unseen L4 classes vs train: 117.
   - `val_hard` contact-map coverage: 5,683 / 6,247 = 90.97%.
   - Current running experiments on `config_ecbench.yaml` are therefore important for paper credibility.

## Main Swiss-Prot Split Checks

### Split Sizes

| Split | N |
|---|---:|
| train | 216,268 |
| val | 27,034 |
| test | 27,034 |
| cluster_train | 210,149 |
| cluster_val | 29,789 |
| cluster_test | 30,398 |
| new392 | 389 |
| price149 | 145 |

### Accession Overlap

Important clean checks:

| Pair | Overlap |
|---|---:|
| train vs val | 0 |
| train vs test | 0 |
| val vs test | 0 |
| cluster_train vs cluster_val | 0 |
| cluster_train vs cluster_test | 0 |
| cluster_val vs cluster_test | 0 |
| price149 vs train | 0 |

Important warning checks:

| Pair | Overlap |
|---|---:|
| new392 vs train | 307 |
| new392 vs val | 44 |
| new392 vs test | 38 |
| cluster_test vs random train | 24,408 |
| cluster_test vs random val | 3,019 |
| cluster_test vs random test | 2,971 |

Interpretation:

- New-392 is already mostly included in the current 2026 Swiss-Prot split.
- Cluster split should be treated as a separate training protocol, not just an extra test set for random-trained checkpoints.

### Exact Sequence Overlap

Important warnings:

| Pair | Exact sequence overlap |
|---|---:|
| random train vs random test | 4,694 |
| random train vs random val | 4,734 |
| random val vs random test | 1,318 |
| new392 vs random train | 303 |
| cluster_test vs random train | 20,964 |

Clean cluster checks:

| Pair | Exact sequence overlap |
|---|---:|
| cluster_train vs cluster_val | 0 |
| cluster_train vs cluster_test | 0 |
| cluster_val vs cluster_test | 0 |

Interpretation:

- Random split is not sequence-disjoint.
- Cluster split is much stronger, but only when training and testing are both from the cluster protocol.

## Label and Multi-Label Statistics

### Main Split Label Stats

| Split | valid L4 rows | unique L4 | avg L4 labels/protein | multi-L4 rate | unseen L4 vs train |
|---|---:|---:|---:|---:|---:|
| train | 183,675 | 4,981 | 0.8929 | 3.88% | 0 |
| val | 22,922 | 2,468 | 0.8913 | 3.94% | 165 |
| test | 23,046 | 2,525 | 0.8945 | 3.80% | 177 |
| cluster_train | 177,484 | 4,703 | 0.8880 | 3.94% | 268 |
| cluster_val | 25,884 | 1,020 | 0.8924 | 1.88% | 25 |
| cluster_test | 26,275 | 1,053 | 0.9273 | 5.37% | 35 |
| new392 | 388 | 173 | 1.2468 | 14.14% | 1 |

Interpretation:

- The task is genuinely multi-label, but the multi-L4 rate in the main Swiss-Prot random split is modest, around 3.8-3.9%.
- New-392 has a much higher multi-label rate, but it overlaps heavily with train under the current dataset.
- The paper should report label-cardinality statistics to justify multi-label BCE.

### Asset Coverage

| Split | embedding coverage | contact map coverage |
|---|---:|---:|
| train | 100.00% | 98.72% |
| val | 100.00% | 98.74% |
| test | 100.00% | 98.65% |
| cluster_train | 100.00% | 98.68% |
| cluster_val | 100.00% | 98.49% |
| cluster_test | 100.00% | 99.19% |
| new392 | 100.00% | 98.71% |
| price149 | 100.00% | 0.00% |

Interpretation:

- Structure-aware claims are well supported on Swiss-Prot splits because contact maps are nearly complete.
- Price-149 cannot test contact-map contribution unless structures are generated or mapped.

## EC-Bench Audit

### Split Sizes

| Split | N |
|---|---:|
| train | 201,865 |
| val | 22,429 |
| val_easy | 16,658 |
| val_hard | 6,247 |
| test | 101 |
| price149 | 136 |

### Accession Overlap

Clean checks:

| Pair | Overlap |
|---|---:|
| train vs val | 0 |
| train vs val_easy | 0 |
| train vs val_hard | 0 |
| train vs test | 0 |
| train vs price149 | 0 |
| val_easy vs val_hard | 0 |

Expected subset checks:

| Pair | Overlap |
|---|---:|
| val vs val_easy | 16,658 |
| val vs val_hard | 5,771 |

Note:

- The audit summary reports `val_hard_ids.txt` has 6,247 IDs, but accession overlap table reports 5,771 overlap with `val`. This should be investigated. It may indicate IDs present in `val_hard_ids.txt` but missing from the combined EC-Bench metadata used by the audit script.

### EC-Bench Label Stats

| Split | valid L4 rows | unique L4 | avg L4 labels/protein | multi-L4 rate | unseen L4 vs train |
|---|---:|---:|---:|---:|---:|
| train | 201,865 | 4,521 | 1.0629 | 5.05% | 0 |
| val | 22,429 | 2,229 | 1.0655 | 5.11% | 126 |
| val_easy | 16,658 | 1,722 | 1.0660 | 5.05% | 10 |
| val_hard | 6,247 | 1,467 | 1.0647 | 5.33% | 117 |
| test | 101 | 53 | 1.0000 | 0.00% | 1 |
| price149 | 136 | 48 | 1.0000 | 0.00% | 0 |

### EC-Bench Asset Coverage

| Split | embedding coverage | contact map coverage |
|---|---:|---:|
| train | 100.00% | 94.90% |
| val | 100.00% | 94.98% |
| val_easy | 100.00% | 95.77% |
| val_hard | 100.00% | 90.97% |
| test | 100.00% | 92.08% |
| price149 | 100.00% | 0.00% |

Interpretation:

- EC-Bench is currently the best route for a defensible OOD validation claim.
- Contact map missingness is higher on EC-Bench hard validation than on random Swiss-Prot, so report missingness explicitly.

## Paper Changes Recommended

### Claims to Weaken or Reframe

1. Replace:
   - "New-392 is a time-stratified external test beyond our training cutoff."

   With:
   - "New-392 is evaluated as a published benchmark. Because our current Swiss-Prot 2026 training corpus overlaps with New-392, we report this result only as a comparability check unless a cutoff-controlled model is trained."

2. Replace:
   - "Cluster test proves OOD generalization."

   With:
   - "Cluster-split results are OOD only for models trained on the corresponding cluster-train split. We therefore report cluster-trained checkpoints separately from random-trained checkpoints."

3. Replace:
   - "Random test demonstrates generalization."

   With:
   - "Random test measures in-distribution performance and may include exact sequence duplicates across splits; sequence-disjoint evaluations are therefore emphasized."

### Experiments to Prioritize

1. Finish the currently running EC-Bench `fusion_v2` and `b3_contact` runs.
2. Evaluate all EC-Bench checkpoints on:
   - `val_hard`
   - `test`
   - `price149`
3. Run a leakage-resistant cluster protocol:
   - train on `cluster_train`
   - validate on `cluster_val`
   - test on `cluster_test`
4. For New-392, either:
   - train a cutoff-controlled model that excludes New-392, or
   - remove temporal-external language from the paper.
5. Add a length-only or shuffled-contact-map baseline to verify B3 contact-only performance.

## Files Generated

- `outputs/audit/audit_summary.json`
- `outputs/audit/main_accession_overlap.csv`
- `outputs/audit/main_exact_sequence_overlap.csv`
- `outputs/audit/main_external_overlap.csv`
- `outputs/audit/main_split_label_stats.csv`
- `outputs/audit/main_asset_coverage.csv`
- `outputs/audit/main_top_l4.csv`
- `outputs/audit/ecbench_accession_overlap.csv`
- `outputs/audit/ecbench_exact_sequence_overlap.csv`
- `outputs/audit/ecbench_split_label_stats.csv`
- `outputs/audit/ecbench_asset_coverage.csv`
- `outputs/audit/ecbench_top_l4.csv`
