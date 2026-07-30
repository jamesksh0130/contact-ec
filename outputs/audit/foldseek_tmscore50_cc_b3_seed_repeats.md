# Foldseek Seed Repeat Audit

- Model: `b3_contact`
- Model label: `B3_contact_only`
- Split: `foldseek_tmscore50_cc_test`
- Runs: 3
- Validation-selected threshold used for test audit: 0.070

## Summary

| Mode | n runs | Micro F1 mean | Micro F1 sd | Precision mean | Recall mean |
|---|---:|---:|---:|---:|---:|
| Fixed threshold 0.5 | 3 | 0.0400 | 0.0054 | 0.2440 | 0.0218 |
| Validation threshold 0.07 | 3 | 0.0654 | 0.0016 | 0.0612 | 0.0702 |
| Post-hoc best threshold | 3 | 0.0700 | 0.0036 | 0.0950 | 0.0558 |

## Individual Runs

### Validation Threshold

| run         |   threshold |     n |   micro_f1 |   macro_f1 |   precision |    recall |   avg_pred_labels |   median_pred_labels |
|:------------|------------:|------:|-----------:|-----------:|------------:|----------:|------------------:|---------------------:|
| seed42_base |        0.07 | 16725 |  0.0652422 | 0.00302633 |   0.0618824 | 0.0689879 |           1.13142 |                    1 |
| seed43      |        0.07 | 16725 |  0.0670215 | 0.00247136 |   0.0613704 | 0.0738188 |           1.22075 |                    1 |
| seed44      |        0.07 | 16725 |  0.0638357 | 0.00217599 |   0.0603484 | 0.0677507 |           1.13937 |                    1 |

### Post-hoc Best Threshold

| run         |     n |   threshold |   micro_f1 |   macro_f1 |   precision |    recall |   avg_pred_labels |   median_pred_labels |
|:------------|------:|------------:|-----------:|-----------:|------------:|----------:|------------------:|---------------------:|
| seed42_base | 16725 |        0.12 |  0.0691261 | 0.00253049 |   0.0903729 | 0.055968  |          0.62852  |                    0 |
| seed43      | 16725 |        0.15 |  0.074019  | 0.00213075 |   0.110397  | 0.0556734 |          0.511809 |                    0 |
| seed44      | 16725 |        0.11 |  0.066948  | 0.00213693 |   0.084083  | 0.0556145 |          0.671271 |                    0 |

