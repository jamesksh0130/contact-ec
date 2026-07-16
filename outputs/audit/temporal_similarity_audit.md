# Temporal Similarity-Stratified Audit

This audit joins the final 124-protein temporal case-wise comparison with the
available EC-Bench `test_vs_train_sim.json` sequence-similarity file.

Important limitation: the similarity file covers 101 temporal proteins, not
all 124 fully evaluable proteins. These results therefore support a
sequence-similarity sensitivity analysis, not a fold-disjoint claim.

## Overall

- Case-wise temporal proteins: 124
- Proteins with recorded train similarity: 101
- Proteins without recorded train similarity: 23
- Max recorded train identity: 1.0000
- Median recorded train identity: 0.4870
- N with identity = 0: 32
- N with identity <= 0.30: 32
- N with identity > 0.90: 6
- Contact-EC micro F1 on similarity subset: 0.6322
- HIT-EC micro F1 on similarity subset: 0.8544
- Figure: outputs/figures/temporal_similarity_strata.png

## Bin Summary

| similarity_bin   |   n |   mean_max_train_seq_identity |   median_max_train_seq_identity |   mean_seq_len |   contactec_micro_f1 |   hitec_micro_f1 |   contactec_hit_rate |   hitec_hit_rate |   contactec_mean_per_protein_f1 |   hitec_mean_per_protein_f1 |
|:-----------------|----:|------------------------------:|--------------------------------:|---------------:|---------------------:|-----------------:|---------------------:|-----------------:|--------------------------------:|----------------------------:|
| 0.00             |  32 |                        0      |                           0     |          364.1 |               0.3673 |           0.8125 |               0.2812 |           0.8125 |                          0.2708 |                      0.8125 |
| (0.30,0.60]      |  36 |                        0.4634 |                           0.479 |          482.3 |               0.5902 |           0.8108 |               0.5    |           0.8333 |                          0.463  |                      0.8241 |
| (0.60,0.90]      |  27 |                        0.7639 |                           0.795 |          307.4 |               0.8889 |           0.9643 |               0.8889 |           1      |                          0.8765 |                      0.9815 |
| >0.90            |   6 |                        0.9562 |                           0.963 |          400.3 |               0.8    |           0.8333 |               0.6667 |           0.8333 |                          0.6667 |                      0.8333 |

## Use in Manuscript

Recommended wording: performance was stratified by the available EC-Bench
train-similarity audit for 101 temporal proteins. The analysis shows how
Contact-EC and HIT-EC vary across low- and high-similarity temporal cases,
but does not replace a Foldseek/TM-align or CATH/SCOP fold-disjoint test.
