# Foldseek Threshold and Top-k Audit

- Split: `foldseek_tmscore50_cc_test`
- Foldseek train prefix for rare-class counts: `foldseek_tmscore50_cc_`
- Rare-class cutoff: <= 25 train proteins
- Rare Level-4 classes: 3028

## Best Threshold by Level-4 Micro F1

| Model | Best threshold | Micro F1 | Macro F1 | Precision | Recall | Avg predicted labels |
|---|---:|---:|---:|---:|---:|---:|
| B1_ESM2_only | 0.050 | 0.0713 | 0.0034 | 0.0793 | 0.0648 | 0.83 |
| B3_contact_only | 0.120 | 0.0692 | 0.0025 | 0.0905 | 0.0560 | 0.63 |
| Fusion_ESM2_contact | 0.050 | 0.1778 | 0.0072 | 0.2018 | 0.1589 | 0.80 |

## Top-k Evaluation

| Model | k | Micro F1 | Macro F1 | Precision | Recall | Hit rate |
|---|---:|---:|---:|---:|---:|---:|
| B1_ESM2_only | 1 | 0.0507 | 0.0024 | 0.0511 | 0.0504 | 0.0511 |
| B1_ESM2_only | 3 | 0.0491 | 0.0022 | 0.0328 | 0.0971 | 0.0984 |
| B1_ESM2_only | 5 | 0.0398 | 0.0020 | 0.0240 | 0.1180 | 0.1193 |
| B1_ESM2_only | 10 | 0.0262 | 0.0020 | 0.0144 | 0.1422 | 0.1438 |
| B3_contact_only | 1 | 0.0668 | 0.0025 | 0.0673 | 0.0663 | 0.0673 |
| B3_contact_only | 3 | 0.0542 | 0.0029 | 0.0363 | 0.1072 | 0.1084 |
| B3_contact_only | 5 | 0.0424 | 0.0028 | 0.0255 | 0.1257 | 0.1271 |
| B3_contact_only | 10 | 0.0277 | 0.0025 | 0.0153 | 0.1505 | 0.1522 |
| Fusion_ESM2_contact | 1 | 0.1358 | 0.0062 | 0.1369 | 0.1349 | 0.1369 |
| Fusion_ESM2_contact | 3 | 0.1192 | 0.0054 | 0.0797 | 0.2357 | 0.2386 |
| Fusion_ESM2_contact | 5 | 0.0904 | 0.0051 | 0.0544 | 0.2679 | 0.2710 |
| Fusion_ESM2_contact | 10 | 0.0578 | 0.0047 | 0.0318 | 0.3134 | 0.3167 |

## Interpretation

The default threshold of 0.5 is conservative on the Foldseek-disjoint split. The best-threshold and top-k views distinguish model ranking ability from fixed-threshold calibration. These values should be reported as diagnostic analyses unless the threshold is selected on a held-out validation split.
