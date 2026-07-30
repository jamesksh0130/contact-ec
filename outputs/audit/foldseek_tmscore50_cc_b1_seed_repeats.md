# Foldseek Seed Repeat Audit

- Model: `b1_esm2_fc`
- Model label: `B1_ESM2_only`
- Split: `foldseek_tmscore50_cc_test`
- Runs: 3
- Validation-selected threshold used for test audit: 0.130

## Summary

| Mode | n runs | Micro F1 mean | Micro F1 sd | Precision mean | Recall mean |
|---|---:|---:|---:|---:|---:|
| Fixed threshold 0.5 | 3 | 0.0433 | 0.0042 | 0.3897 | 0.0230 |
| Validation threshold 0.13 | 3 | 0.0607 | 0.0009 | 0.1359 | 0.0391 |
| Post-hoc best threshold | 3 | 0.0666 | 0.0042 | 0.0692 | 0.0657 |

## Individual Runs

### Validation Threshold

| run         |   threshold |     n |   micro_f1 |   macro_f1 |   precision |    recall |   avg_pred_labels |   median_pred_labels |
|:------------|------------:|------:|-----------:|-----------:|------------:|----------:|------------------:|---------------------:|
| seed42_base |        0.13 | 16725 |  0.0612875 | 0.00266514 |    0.140681 | 0.0391776 |          0.282631 |                    0 |
| seed43      |        0.13 | 16725 |  0.0596961 | 0.0028989  |    0.128455 | 0.038883  |          0.307205 |                    0 |
| seed44      |        0.13 | 16725 |  0.0610876 | 0.00281661 |    0.138599 | 0.0391776 |          0.286876 |                    0 |

### Post-hoc Best Threshold

| run         |     n |   threshold |   micro_f1 |   macro_f1 |   precision |    recall |   avg_pred_labels |   median_pred_labels |
|:------------|------:|------------:|-----------:|-----------:|------------:|----------:|------------------:|---------------------:|
| seed42_base | 16725 |        0.05 |  0.0713359 | 0.00335962 |   0.0793307 | 0.064805  |          0.829058 |                    0 |
| seed43      | 16725 |        0.04 |  0.0631289 | 0.00358839 |   0.0557158 | 0.0728172 |          1.3264   |                    1 |
| seed44      | 16725 |        0.05 |  0.0653591 | 0.00332643 |   0.0726696 | 0.0593849 |          0.829357 |                    1 |

