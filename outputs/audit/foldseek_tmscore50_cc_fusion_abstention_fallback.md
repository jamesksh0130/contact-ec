# Foldseek Abstention and Parent-Level Fallback Audit

- Model: `Fusion_ESM2_contact`
- Level-4 threshold: 0.08
- Samples: 16,725

| model               |   abstain_low_conf_fraction |   coverage |   retained_l4_micro_f1 |   retained_l4_sample_hit |   abstained_unseen_prevalence |   abstained_l3_top1_hit |   overall_l4_or_fallback_l3_hit |   mean_top5_sum_retained |   mean_top5_sum_abstained |
|:--------------------|----------------------------:|-----------:|-----------------------:|-------------------------:|------------------------------:|------------------------:|--------------------------------:|-------------------------:|--------------------------:|
| Fusion_ESM2_contact |                        0    |   1        |               0.208228 |                 0.164843 |                    nan        |             nan         |                        0.164843 |                 0.185508 |              nan          |
| Fusion_ESM2_contact |                        0.05 |   0.950015 |               0.215006 |                 0.173516 |                      0.843301 |               0.034689  |                        0.166577 |                 0.195113 |                0.00296062 |
| Fusion_ESM2_contact |                        0.1  |   0.90003  |               0.222303 |                 0.183153 |                      0.712321 |               0.0592105 |                        0.170762 |                 0.205446 |                0.00600698 |
| Fusion_ESM2_contact |                        0.15 |   0.849985 |               0.230179 |                 0.193936 |                      0.64727  |               0.082503  |                        0.17722  |                 0.216571 |                0.00950353 |
| Fusion_ESM2_contact |                        0.2  |   0.8      |               0.238624 |                 0.206054 |                      0.591928 |               0.112108  |                        0.187265 |                 0.228549 |                0.0133433  |
| Fusion_ESM2_contact |                        0.3  |   0.69997  |               0.257472 |                 0.2355   |                      0.532882 |               0.157433  |                        0.212078 |                 0.255532 |                0.0221417  |
| Fusion_ESM2_contact |                        0.4  |   0.6      |               0.279496 |                 0.274738 |                      0.495665 |               0.184753  |                        0.238744 |                 0.287572 |                0.0324127  |
| Fusion_ESM2_contact |                        0.5  |   0.50003  |               0.30468  |                 0.324166 |                      0.465439 |               0.204497  |                        0.264335 |                 0.326711 |                0.0442885  |

## Interpretation

- Abstention removes the lowest-confidence proteins according to the top-5 Level-4 probability sum.
- Retained Level-4 micro F1 estimates performance on accepted closed-set predictions.
- Abstained Level-3 top-1 hit estimates whether a conservative parent-level fallback remains useful.
