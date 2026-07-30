# Foldseek Threshold and Top-k Audit

- Split: `foldseek_tmscore50_cc_val`
- Foldseek train prefix for rare-class counts: `foldseek_tmscore50_cc_`
- Rare-class cutoff: <= 25 train proteins
- Rare Level-4 classes: 3028

## Best Threshold by Level-4 Micro F1

| Model | Best threshold | Micro F1 | Macro F1 | Precision | Recall | Avg predicted labels |
|---|---:|---:|---:|---:|---:|---:|
| B1_ESM2_only | 0.130 | 0.0601 | 0.0022 | 0.1266 | 0.0394 | 0.35 |
| B3_contact_only | 0.070 | 0.0146 | 0.0014 | 0.0148 | 0.0145 | 1.10 |
| Fusion_ESM2_contact | 0.080 | 0.0757 | 0.0056 | 0.1346 | 0.0526 | 0.44 |

## Top-k Evaluation

| Model | k | Micro F1 | Macro F1 | Precision | Recall | Hit rate |
|---|---:|---:|---:|---:|---:|---:|
| B1_ESM2_only | 1 | 0.0496 | 0.0016 | 0.0526 | 0.0469 | 0.0526 |
| B1_ESM2_only | 3 | 0.0393 | 0.0020 | 0.0270 | 0.0722 | 0.0798 |
| B1_ESM2_only | 5 | 0.0322 | 0.0021 | 0.0197 | 0.0878 | 0.0960 |
| B1_ESM2_only | 10 | 0.0212 | 0.0015 | 0.0118 | 0.1053 | 0.1148 |
| B3_contact_only | 1 | 0.0129 | 0.0015 | 0.0137 | 0.0122 | 0.0137 |
| B3_contact_only | 3 | 0.0172 | 0.0019 | 0.0118 | 0.0316 | 0.0354 |
| B3_contact_only | 5 | 0.0154 | 0.0018 | 0.0094 | 0.0419 | 0.0470 |
| B3_contact_only | 10 | 0.0115 | 0.0019 | 0.0064 | 0.0568 | 0.0633 |
| Fusion_ESM2_contact | 1 | 0.0959 | 0.0054 | 0.1018 | 0.0907 | 0.1018 |
| Fusion_ESM2_contact | 3 | 0.0859 | 0.0058 | 0.0590 | 0.1578 | 0.1741 |
| Fusion_ESM2_contact | 5 | 0.0713 | 0.0055 | 0.0437 | 0.1947 | 0.2152 |
| Fusion_ESM2_contact | 10 | 0.0506 | 0.0045 | 0.0281 | 0.2507 | 0.2777 |

## Interpretation

The default threshold of 0.5 is conservative on the Foldseek-disjoint split. The best-threshold and top-k views distinguish model ranking ability from fixed-threshold calibration. These values should be reported as diagnostic analyses unless the threshold is selected on a held-out validation split.
