# Raw contact-map topology nearest-neighbor audit

This is not a Foldseek/TM-align fold-disjoint benchmark. It uses hand-crafted fingerprints derived from the raw 256x256 contact maps: a 32x32 coarse contact grid, sequence-separation contact densities, and node-degree statistics.

## Overall

|   temporal_known_with_contact_maps |   train_processed |   train_with_contact_maps |   feature_dimension |   median_nearest_cosine |   mean_nearest_cosine |   n_cosine_gt_0_90 |   n_cosine_gt_0_95 |   n_cosine_gt_0_98 |   same_l1_rate |   same_l2_rate |   same_l3_rate |   same_l4_rate |
|-----------------------------------:|------------------:|--------------------------:|--------------------:|------------------------:|----------------------:|-------------------:|-------------------:|-------------------:|---------------:|---------------:|---------------:|---------------:|
|                                115 |            224294 |                    212863 |                1041 |                0.982188 |              0.973251 |                114 |                 93 |                 63 |       0.756522 |       0.686957 |       0.686957 |       0.408696 |

## Bins

| cosine_bin   |   n |   mean_cosine |   same_l1_rate |   same_l2_rate |   same_l3_rate |   same_l4_rate |
|:-------------|----:|--------------:|---------------:|---------------:|---------------:|---------------:|
| <=0.80       |   0 |    nan        |     nan        |     nan        |     nan        |     nan        |
| 0.80-0.90    |   1 |      0.888666 |       1        |       0        |       0        |       0        |
| 0.90-0.95    |  21 |      0.933588 |       0.428571 |       0.238095 |       0.238095 |       0.142857 |
| 0.95-0.98    |  30 |      0.968831 |       0.566667 |       0.466667 |       0.466667 |       0.266667 |
| >0.98        |  63 |      0.98992  |       0.952381 |       0.952381 |       0.952381 |       0.571429 |

## Interpretation

- This audit is independent of the learned contact-pair ESM representation.
- High nearest-neighbor topology similarity indicates that temporal performance should still be interpreted conservatively.
- EC overlap rates among nearest topology neighbours quantify whether contact-map proximity also recovers enzyme hierarchy.
- A publication-grade fold-disjoint claim still requires Foldseek/TM-align or CATH/SCOP clustering followed by retraining/evaluation.
