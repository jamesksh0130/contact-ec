# Validation-selected Threshold Test Performance

- Test split: `foldseek_tmscore50_cc_test`
- Thresholds are selected on the Foldseek validation split and applied once to the Foldseek test split.

| Model | Val-selected threshold | Val micro F1 | Test micro F1 | Test precision | Test recall | Post-hoc test best F1 |
|---|---:|---:|---:|---:|---:|---:|
| B1_ESM2_only | 0.130 | 0.0601 | 0.0613 | 0.1407 | 0.0392 | 0.0713 |
| B3_contact_only | 0.070 | 0.0146 | 0.0652 | 0.0619 | 0.0690 | 0.0692 |
| Fusion_ESM2_contact | 0.080 | 0.0757 | 0.1747 | 0.2406 | 0.1371 | 0.1778 |

## Interpretation

These values are safer headline diagnostic metrics than post-hoc test-selected thresholds. The post-hoc values remain useful as calibration diagnostics, but the validation-selected thresholds better approximate a deployable threshold selection protocol.
