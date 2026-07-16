# Contact-pair nearest-neighbor audit

This is not a Foldseek/TM-align fold-disjoint benchmark. It uses the existing contact-pair ESM representation as a model-proximal relatedness audit.

## Overall

|   temporal_known_with_pair_embeddings |   train_processed |   train_with_pair_embeddings |   median_nearest_cosine |   mean_nearest_cosine |   n_cosine_gt_0_80 |   n_cosine_gt_0_90 |   same_l1_rate |   same_l2_rate |   same_l3_rate |   same_l4_rate |
|--------------------------------------:|------------------:|-----------------------------:|------------------------:|----------------------:|-------------------:|-------------------:|---------------:|---------------:|---------------:|---------------:|
|                                   113 |            224294 |                       212863 |                0.985153 |              0.984486 |                113 |                113 |       0.876106 |       0.823009 |       0.823009 |        0.59292 |

## Bins

| cosine_bin   |   n |   mean_cosine |   same_l1_rate |   same_l2_rate |   same_l3_rate |   same_l4_rate |
|:-------------|----:|--------------:|---------------:|---------------:|---------------:|---------------:|
| <=0.40       |   0 |    nan        |     nan        |     nan        |     nan        |      nan       |
| 0.40-0.60    |   0 |    nan        |     nan        |     nan        |     nan        |      nan       |
| 0.60-0.80    |   0 |    nan        |     nan        |     nan        |     nan        |      nan       |
| 0.80-0.90    |   0 |    nan        |     nan        |     nan        |     nan        |      nan       |
| >0.90        | 113 |      0.984486 |       0.876106 |       0.823009 |       0.823009 |        0.59292 |

## Interpretation

- High nearest-neighbor similarity indicates that temporal performance should not be interpreted as fold-disjoint structural generalization.
- Same-EC rates among nearest neighbours quantify how often contact-conditioned representation proximity also recovers EC hierarchy.
- A true fold-disjoint claim still requires Foldseek/TM-align or CATH/SCOP clustering and retraining/evaluation on the resulting split.
