# Foldseek Open-Vocabulary Confidence Audit

- Samples: 16,725
- Proteins with any unseen true Level-4 label: 0.316

## Best confidence score by model

| model               | score        | target                |   prevalence |    auroc |   average_precision |
|:--------------------|:-------------|:----------------------|-------------:|---------:|--------------------:|
| Fusion_ESM2_contact | low_top5_sum | any_unseen_true_label |     0.316114 | 0.73825  |            0.589875 |
| B1_ESM2_only        | low_top5_sum | any_unseen_true_label |     0.316114 | 0.675618 |            0.436001 |
| B3_contact_only     | low_top1     | any_unseen_true_label |     0.316114 | 0.573879 |            0.365212 |

## All confidence scores

| model               | score              | target                |   prevalence |    auroc |   average_precision |
|:--------------------|:-------------------|:----------------------|-------------:|---------:|--------------------:|
| B1_ESM2_only        | low_top5_sum       | any_unseen_true_label |     0.316114 | 0.675618 |            0.436001 |
| B1_ESM2_only        | low_top1           | any_unseen_true_label |     0.316114 | 0.673545 |            0.446128 |
| B1_ESM2_only        | low_pred_count_005 | any_unseen_true_label |     0.316114 | 0.644543 |            0.40386  |
| B1_ESM2_only        | low_prob_sum       | any_unseen_true_label |     0.316114 | 0.626929 |            0.41756  |
| B1_ESM2_only        | low_pred_count_010 | any_unseen_true_label |     0.316114 | 0.625673 |            0.309095 |
| B1_ESM2_only        | high_entropy       | any_unseen_true_label |     0.316114 | 0.453238 |            0.283867 |
| B3_contact_only     | low_top1           | any_unseen_true_label |     0.316114 | 0.573879 |            0.365212 |
| B3_contact_only     | low_top5_sum       | any_unseen_true_label |     0.316114 | 0.558238 |            0.363872 |
| B3_contact_only     | low_pred_count_010 | any_unseen_true_label |     0.316114 | 0.545622 |            0.394864 |
| B3_contact_only     | high_entropy       | any_unseen_true_label |     0.316114 | 0.54183  |            0.368484 |
| B3_contact_only     | low_pred_count_005 | any_unseen_true_label |     0.316114 | 0.525933 |            0.363641 |
| B3_contact_only     | low_prob_sum       | any_unseen_true_label |     0.316114 | 0.491267 |            0.364625 |
| Fusion_ESM2_contact | low_top5_sum       | any_unseen_true_label |     0.316114 | 0.73825  |            0.589875 |
| Fusion_ESM2_contact | low_prob_sum       | any_unseen_true_label |     0.316114 | 0.731726 |            0.588086 |
| Fusion_ESM2_contact | low_top1           | any_unseen_true_label |     0.316114 | 0.728694 |            0.575201 |
| Fusion_ESM2_contact | low_pred_count_005 | any_unseen_true_label |     0.316114 | 0.677493 |            0.494619 |
| Fusion_ESM2_contact | low_pred_count_010 | any_unseen_true_label |     0.316114 | 0.650494 |            0.487544 |
| Fusion_ESM2_contact | high_entropy       | any_unseen_true_label |     0.316114 | 0.30798  |            0.231113 |

## Interpretation

- AUROC close to 0.5 means model confidence is weak for detecting unseen-label cases.
- Strong AUROC/AP would support an abstention or hierarchical fallback mechanism.
- This is a diagnostic audit, not a supervised open-set classifier.
