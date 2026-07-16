# EC Level-1 Family Analysis

N = 124 fully evaluable Swiss-Prot 2023-01 temporal proteins.

## Summary

|   true_l1 | family          |   n |   mean_seq_len |   ESM2_micro_f1 |   ESM2_hit_rate |   ContactOnly_micro_f1 |   ContactOnly_hit_rate |   ContactEC_micro_f1 |   ContactEC_hit_rate |   HITEC_micro_f1 |   HITEC_hit_rate |   fusion_minus_esm2 |   fusion_minus_contact |
|----------:|:----------------|----:|---------------:|----------------:|----------------:|-----------------------:|-----------------------:|---------------------:|---------------------:|-----------------:|-----------------:|--------------------:|-----------------------:|
|         1 | Oxidoreductases |  18 |          436.8 |          0.0909 |          0.0556 |                 0.2222 |                 0.1667 |               0.7179 |               0.7222 |           0.9231 |           0.9444 |              0.627  |                 0.4957 |
|         2 | Transferases    |  25 |          529.8 |          0.6667 |          0.52   |                 0.6154 |                 0.28   |               0.7059 |               0.56   |           0.8649 |           0.84   |              0.0392 |                 0.0905 |
|         3 | Hydrolases      |  42 |          256.8 |          0.5085 |          0.3571 |                 0.4561 |                 0.3095 |               0.7838 |               0.6905 |           0.9545 |           0.9524 |              0.2753 |                 0.3277 |
|         4 | Lyases          |  32 |          456.9 |          0.1923 |          0.1562 |                 0.0833 |                 0.0625 |               0.1695 |               0.1562 |           0.6667 |           0.7812 |             -0.0228 |                 0.0862 |
|         5 | Isomerases      |   7 |          539.4 |          0.3636 |          0.2857 |                 0.25   |                 0.1429 |               0.6667 |               0.5714 |           0.7143 |           0.7143 |              0.3031 |                 0.4167 |

## Interpretation

Contact-EC is strongest in EC families where ESM-2 and contact-only signals
are both non-trivial, while families with few examples remain unstable.
This analysis adds biological granularity but should not be overinterpreted
as a mechanistic active-site explanation.

Figure: outputs/figures/l1_family_micro_f1.png
