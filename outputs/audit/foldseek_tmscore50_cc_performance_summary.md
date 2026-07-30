# Foldseek/TM-score Structure-Disjoint Performance Summary

This audit evaluates Contact-EC variants on the `foldseek_tmscore50_cc_test` split.

## Split Construction

- Metadata proteins: 224,294
- Structures available: 212,863
- Structures missing: 11,431
- Foldseek clusters: 3,813
- Foldseek TM-score threshold: 0.50
- Coverage threshold: 0.80
- Minimum sequence identity: 0.0
- Alignment type: 1
- Cluster mode: 1
- Assignment mode: cluster-count balanced
- Train proteins/clusters: 170,573 / 3,051
- Validation proteins/clusters: 25,565 / 381
- Test proteins/clusters: 16,725 / 381
- Train/validation/test protein ID overlaps: 0 / 0 / 0

## Test Performance

| Model | L1 micro F1 | L2 micro F1 | L3 micro F1 | L4 micro F1 | L4 macro F1 | L4 precision | L4 recall | Rare L4 micro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 ESM-2 only | 0.4542 | 0.0385 | 0.0130 | 0.0452 | 0.0015 | 0.3402 | 0.0242 | 0.0000 |
| B3 contact only | 0.3679 | 0.0393 | 0.0014 | 0.0436 | 0.0017 | 0.2711 | 0.0237 | 0.0000 |
| Fusion ESM-2 + contact | 0.5151 | 0.0177 | 0.0078 | 0.0864 | 0.0033 | 0.5175 | 0.0471 | 0.0014 |

## Threshold and Top-k Diagnostics

The fixed 0.5 threshold is conservative on the fold-disjoint test split. A post-hoc threshold sweep gives the following best Level-4 micro F1 values:

| Model | Best threshold | L4 micro F1 | Precision | Recall | Avg predicted labels |
|---|---:|---:|---:|---:|---:|
| B1 ESM-2 only | 0.050 | 0.0713 | 0.0793 | 0.0648 | 0.83 |
| B3 contact only | 0.120 | 0.0692 | 0.0905 | 0.0560 | 0.63 |
| Fusion ESM-2 + contact | 0.050 | 0.1778 | 0.2018 | 0.1589 | 0.80 |

Top-k diagnostics show the same ranking pattern:

| Model | Top-1 hit rate | Top-3 hit rate | Top-5 hit rate | Top-10 hit rate |
|---|---:|---:|---:|---:|
| B1 ESM-2 only | 0.0511 | 0.0984 | 0.1193 | 0.1438 |
| B3 contact only | 0.0673 | 0.1084 | 0.1271 | 0.1522 |
| Fusion ESM-2 + contact | 0.1369 | 0.2386 | 0.2710 | 0.3167 |

The best-threshold values are diagnostic rather than final headline metrics unless thresholds are selected on a held-out validation split. They nevertheless show that Fusion retains stronger ranking and calibration behavior than either single-modality baseline under fold-level shift.

Validation-selected thresholds give similar conclusions while avoiding post-hoc threshold selection on test:

| Model | Val threshold | Val micro F1 | Test micro F1 | Test precision | Test recall | Post-hoc test best F1 |
|---|---:|---:|---:|---:|---:|---:|
| B1 ESM-2 only | 0.130 | 0.0601 | 0.0613 | 0.1407 | 0.0392 | 0.0713 |
| B3 contact only | 0.070 | 0.0146 | 0.0652 | 0.0619 | 0.0690 | 0.0692 |
| Fusion ESM-2 + contact | 0.080 | 0.0757 | 0.1747 | 0.2406 | 0.1371 | 0.1778 |

## Label Difficulty Decomposition

Fold-disjoint evaluation also introduces a substantial label-coverage shift:

| Label-frequency bin in Foldseek train | L4 classes | Test positive labels | Fraction |
|---|---:|---:|---:|
| unseen | 1,335 | 5,352 | 0.3153 |
| 1-5 | 2,281 | 4,049 | 0.2385 |
| 6-25 | 907 | 2,483 | 0.1463 |
| 26-100 | 403 | 3,295 | 0.1941 |
| >100 | 381 | 1,795 | 0.1057 |

At the sample level, 5,287 of 16,725 test proteins (0.3161) contain at least one Level-4 label unseen in the Foldseek training split.

Performance by train-frequency bin at model-specific best thresholds:

| Model | unseen | 1-5 | 6-25 | 26-100 | >100 |
|---|---:|---:|---:|---:|---:|
| B1 ESM-2 only | 0.0000 | 0.0000 | 0.0000 | 0.0847 | 0.1242 |
| B3 contact only | 0.0000 | 0.0055 | 0.0448 | 0.1081 | 0.1825 |
| Fusion ESM-2 + contact | 0.0000 | 0.0038 | 0.2484 | 0.3278 | 0.2167 |

This indicates that the low fold-disjoint score is not only a fold-shift effect. It is compounded by Level-4 vocabulary and frequency shift. Fusion is strongest for moderately represented labels (6-100 training examples), while no closed-set model can recover labels absent from the training vocabulary.

EC Level-1 family decomposition at model-specific best thresholds shows that Fusion gains are concentrated in oxidoreductases, hydrolases, and transferases:

| Family | B1 micro F1 | B3 micro F1 | Fusion micro F1 |
|---|---:|---:|---:|
| Oxidoreductases | 0.1157 | 0.0507 | 0.2612 |
| Transferases | 0.0635 | 0.0606 | 0.1497 |
| Hydrolases | 0.0777 | 0.0967 | 0.2576 |
| Lyases | 0.1288 | 0.1135 | 0.0468 |
| Isomerases | 0.2468 | 0.2440 | 0.2318 |
| Ligases | 0.0011 | 0.0005 | 0.0036 |

## Interpretation

The Foldseek/TM-score-disjoint split is substantially harder than the original sequence-disjoint benchmark. Both single-modality baselines show low Level-4 performance at the default 0.5 sigmoid threshold. The fusion model remains low in absolute terms but improves Level-4 micro F1 by 0.0412 over ESM-2 only and by 0.0428 over contact-only, corresponding to approximately 1.9x and 2.0x relative gains, respectively.

These results support a revised framing: sequence-disjoint EC benchmarks can overestimate generalization under fold-level distribution shift, while sequence-structure fusion preserves a measurable but still limited advantage in novel-fold-like settings. Threshold and top-k diagnostics further suggest that some of the default-threshold drop reflects calibration, but not enough to erase the large fold-disjoint performance gap. The label-difficulty audit shows that fold-disjoint evaluation also induces substantial label-coverage and rare-class shift, which must be reported separately from structural generalization.

Further required analyses:

- Multiple random seeds with mean and standard deviation.
- Additional baseline architectures under the same Foldseek split.
