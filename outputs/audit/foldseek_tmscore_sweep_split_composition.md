# Foldseek TM-score Sweep Split Composition

| TM-score | Assignment | Clusters | Max cluster | Test proteins | Valid test | Seen L4 classes | Unseen positive labels | Any-unseen proteins | Rare seen positives |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | protein_balanced | 10,893 | 1,026 | 21,617 | 21,617 | 4,500 | 0.159 | 0.170 | 0.508 |
| 0.50 | cluster_count | 3,813 | 18,692 | 16,725 | 16,725 | 3,972 | 0.315 | 0.316 | 0.385 |
| 0.60 | protein_balanced | 11,869 | 1,026 | 21,555 | 21,555 | 4,500 | 0.103 | 0.110 | 0.467 |

Interpretation: the threshold sweep changes cluster granularity and, in the current generated splits, also changes the assignment mode for the TM-score 0.50 split. Therefore, the sweep should be interpreted as a robustness/composition audit rather than a strictly monotonic structural-stringency curve.
