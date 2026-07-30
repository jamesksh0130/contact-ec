# Foldseek Seed Repeat Audit

- Model: `fusion`
- Model label: `Fusion_ESM2_contact`
- Split: `foldseek_tmscore50_cc_test`
- Runs: 3
- Validation-selected threshold used for test audit: 0.080

## Summary

| Mode | n runs | Micro F1 mean | Micro F1 sd | Precision mean | Recall mean |
|---|---:|---:|---:|---:|---:|
| Fixed threshold 0.5 | 3 | 0.0872 | 0.0122 | 0.5392 | 0.0475 |
| Validation threshold 0.08 | 3 | 0.1685 | 0.0334 | 0.2520 | 0.1270 |
| Post-hoc best threshold | 3 | 0.1769 | 0.0378 | 0.2036 | 0.1574 |

## Individual Runs

### Validation Threshold

| run         |   threshold |     n |   micro_f1 |   macro_f1 |   precision |   recall |   avg_pred_labels |   median_pred_labels |
|:------------|------------:|------:|-----------:|-----------:|------------:|---------:|------------------:|---------------------:|
| seed42_base |        0.08 | 16725 |   0.17466  | 0.00669906 |    0.240591 | 0.137092 |          0.578296 |                    0 |
| seed43      |        0.08 | 16725 |   0.132381 | 0.00499635 |    0.215673 | 0.095499 |          0.449387 |                    0 |
| seed44      |        0.08 | 16725 |   0.198416 | 0.00945012 |    0.29975  | 0.148286 |          0.502063 |                    0 |

### Post-hoc Best Threshold

| run         |     n |   threshold |   micro_f1 |   macro_f1 |   precision |   recall |   avg_pred_labels |   median_pred_labels |
|:------------|------:|------------:|-----------:|-----------:|------------:|---------:|------------------:|---------------------:|
| seed42_base | 16725 |        0.05 |   0.177822 | 0.00715148 |    0.20178  | 0.158949 |          0.799462 |                    1 |
| seed43      | 16725 |        0.05 |   0.138605 | 0.00565088 |    0.178781 | 0.113173 |          0.642451 |                    0 |
| seed44      | 16725 |        0.03 |   0.214169 | 0.00895045 |    0.230248 | 0.200189 |          0.882392 |                    1 |

