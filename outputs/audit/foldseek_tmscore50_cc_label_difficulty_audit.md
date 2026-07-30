# Foldseek Label Difficulty Audit

- Split: `foldseek_tmscore50_cc_test`
- Train prefix: `foldseek_tmscore50_cc_`
- L4 classes seen in foldseek train: 3,972 / 5,307

## Label Coverage

| Bin | Classes | Positive labels | Fraction |
|---|---:|---:|---:|
| unseen | 1335.0 | 5352 | 0.3153 |
| 1-5 | 2281.0 | 4049 | 0.2385 |
| 6-25 | 907.0 | 2483 | 0.1463 |
| 26-100 | 403.0 | 3295 | 0.1941 |
| >100 | 381.0 | 1795 | 0.1057 |
| sample_all_labels_seen |  | 11438 | 0.6839 |
| sample_any_unseen_label |  | 5287 | 0.3161 |

## Frequency-bin Performance at Model-specific Best Thresholds

| Model | Bin | Classes | Micro F1 | Precision | Recall | Support |
|---|---|---:|---:|---:|---:|---:|
| B1 ESM-2 only | unseen | 1335 | 0.0000 | 0.0000 | 0.0000 | 5352 |
| B1 ESM-2 only | 1-5 | 2281 | 0.0000 | 0.0000 | 0.0000 | 4049 |
| B1 ESM-2 only | 6-25 | 907 | 0.0000 | 0.0000 | 0.0000 | 2483 |
| B1 ESM-2 only | 26-100 | 403 | 0.0847 | 0.2665 | 0.0504 | 3295 |
| B1 ESM-2 only | >100 | 381 | 0.1242 | 0.0705 | 0.5203 | 1795 |
| B3 contact only | unseen | 1335 | 0.0000 | 0.0000 | 0.0000 | 5352 |
| B3 contact only | 1-5 | 2281 | 0.0055 | 0.0093 | 0.0040 | 4049 |
| B3 contact only | 6-25 | 907 | 0.0448 | 0.0519 | 0.0395 | 2483 |
| B3 contact only | 26-100 | 403 | 0.1081 | 0.1034 | 0.1132 | 3295 |
| B3 contact only | >100 | 381 | 0.1825 | 0.1410 | 0.2585 | 1795 |
| Fusion ESM-2 + contact | unseen | 1335 | 0.0000 | 0.0000 | 0.0000 | 5352 |
| Fusion ESM-2 + contact | 1-5 | 2281 | 0.0038 | 0.0503 | 0.0020 | 4049 |
| Fusion ESM-2 + contact | 6-25 | 907 | 0.2484 | 0.2418 | 0.2553 | 2483 |
| Fusion ESM-2 + contact | 26-100 | 403 | 0.3278 | 0.3361 | 0.3199 | 3295 |
| Fusion ESM-2 + contact | >100 | 381 | 0.2167 | 0.1344 | 0.5582 | 1795 |

## EC Level-1 Family Performance

| Model | L1 | Family | n | Micro F1 | Precision | Recall | Mean sample F1 |
|---|---:|---|---:|---:|---:|---:|---:|
| B1 ESM-2 only | 1 | Oxidoreductases | 2082 | 0.1157 | 0.1630 | 0.0897 | 0.0596 |
| B1 ESM-2 only | 2 | Transferases | 5315 | 0.0635 | 0.0765 | 0.0543 | 0.0376 |
| B1 ESM-2 only | 3 | Hydrolases | 5248 | 0.0777 | 0.0818 | 0.0740 | 0.0482 |
| B1 ESM-2 only | 4 | Lyases | 1123 | 0.1288 | 0.1467 | 0.1148 | 0.1016 |
| B1 ESM-2 only | 5 | Isomerases | 397 | 0.2468 | 0.2794 | 0.2211 | 0.2133 |
| B1 ESM-2 only | 6 | Ligases | 2560 | 0.0011 | 0.0011 | 0.0012 | 0.0010 |
| B3 contact only | 1 | Oxidoreductases | 2082 | 0.0507 | 0.0681 | 0.0404 | 0.0367 |
| B3 contact only | 2 | Transferases | 5315 | 0.0606 | 0.0662 | 0.0559 | 0.0542 |
| B3 contact only | 3 | Hydrolases | 5248 | 0.0967 | 0.1406 | 0.0737 | 0.0601 |
| B3 contact only | 4 | Lyases | 1123 | 0.1135 | 0.2041 | 0.0786 | 0.0703 |
| B3 contact only | 5 | Isomerases | 397 | 0.2440 | 0.3045 | 0.2035 | 0.1966 |
| B3 contact only | 6 | Ligases | 2560 | 0.0005 | 0.0008 | 0.0004 | 0.0004 |
| Fusion ESM-2 + contact | 1 | Oxidoreductases | 2082 | 0.2612 | 0.3132 | 0.2240 | 0.1997 |
| Fusion ESM-2 + contact | 2 | Transferases | 5315 | 0.1497 | 0.1937 | 0.1220 | 0.0995 |
| Fusion ESM-2 + contact | 3 | Hydrolases | 5248 | 0.2576 | 0.2461 | 0.2701 | 0.1943 |
| Fusion ESM-2 + contact | 4 | Lyases | 1123 | 0.0468 | 0.0889 | 0.0318 | 0.0288 |
| Fusion ESM-2 + contact | 5 | Isomerases | 397 | 0.2318 | 0.2691 | 0.2035 | 0.2024 |
| Fusion ESM-2 + contact | 6 | Ligases | 2560 | 0.0036 | 0.0042 | 0.0031 | 0.0016 |

## Interpretation

This audit separates fold-disjoint performance by label coverage, label frequency, and broad EC family. It should be used to explain whether the fold-disjoint drop is dominated by unseen/rare labels or is broadly present across enzyme families.
