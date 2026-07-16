# Contact-EC calibration and prediction-cardinality audit

| dataset                 |   n |   avg_pred_labels |   median_pred_labels |   empty_prediction_rate |   top1_prob_mean |   top1_prob_median |   mean_probability |   micro_f1 |   precision |   recall |   top1_hit_rate |    brier |      ece |
|:------------------------|----:|------------------:|---------------------:|------------------------:|-----------------:|-------------------:|-------------------:|-----------:|------------:|---------:|----------------:|---------:|---------:|
| temporal_known_complete | 124 |            0.7984 |               1.0000 |                  0.3629 |           0.6657 |             0.8997 |             0.0002 |     0.6032 |      0.7677 |   0.4967 |          0.5726 |   0.0002 |   0.0001 |
| temporal_partial        | 309 |            0.4595 |               0.0000 |                  0.6440 |           0.3862 |             0.2699 |             0.0002 |   nan      |    nan      | nan      |        nan      | nan      | nan      |
| temporal_novel_complete |  35 |            0.4286 |               0.0000 |                  0.7429 |           0.3404 |             0.2199 |             0.0002 |   nan      |    nan      | nan      |        nan      | nan      | nan      |
| Price-149               | 136 |            0.6471 |               1.0000 |                  0.4853 |           0.5187 |             0.5527 |             0.0002 |     0.2500 |      0.3182 |   0.2059 |          0.2794 |   0.0002 |   0.0002 |

## Interpretation

- Partial and novel temporal proteins are not scored at Level 4, but Contact-EC still emits closed-set predictions for them.
- Price-149 shows non-empty predictions but low exact Level-4 correctness, supporting an OOD specificity/calibration interpretation.
- This diagnostic complements the threshold-sensitivity and Price-149 failure analyses.
