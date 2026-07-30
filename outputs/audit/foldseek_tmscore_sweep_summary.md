# Foldseek TM-score Strictness Sweep

## Available results

|   tm_score_threshold | split                      | model             |   test_proteins |   clusters |   cluster_size_median |   cluster_size_max |   l1_accuracy |   l2_accuracy |   l3_accuracy |   l4_micro_f1 |   l4_precision |   l4_recall |
|---------------------:|:---------------------------|:------------------|----------------:|-----------:|----------------------:|-------------------:|--------------:|--------------:|--------------:|--------------:|---------------:|------------:|
|                  0.4 | foldseek_tmscore40_cc_test | B1 ESM-2 only     |           21617 |      10893 |                     2 |               1026 |        0.4108 |        0.0701 |        0.0548 |        0.0683 |         0.4115 |      0.0372 |
|                  0.4 | foldseek_tmscore40_cc_test | B3 contact only   |           21617 |      10893 |                     2 |               1026 |        0.4198 |        0.0798 |        0.0584 |        0.2622 |         0.7039 |      0.1611 |
|                  0.4 | foldseek_tmscore40_cc_test | Contact-EC fusion |           21617 |      10893 |                     2 |               1026 |        0.4115 |        0.0879 |        0.0564 |        0.2981 |         0.6311 |      0.1951 |
|                  0.5 | foldseek_tmscore50_cc_test | B1 ESM-2 only     |           16725 |       3813 |                     2 |              18692 |        0.4692 |        0.0356 |        0.0132 |        0.0462 |         0.3759 |      0.0246 |
|                  0.5 | foldseek_tmscore50_cc_test | B3 contact only   |           16725 |       3813 |                     2 |              18692 |        0.3651 |        0.0097 |        0.0029 |        0.0337 |         0.1984 |      0.0184 |
|                  0.5 | foldseek_tmscore50_cc_test | Contact-EC fusion |           16725 |       3813 |                     2 |              18692 |        0.5366 |        0.031  |        0.005  |        0.0998 |         0.5794 |      0.0546 |
|                  0.6 | foldseek_tmscore60_cc_test | B1 ESM-2 only     |           21555 |      11869 |                     2 |               1026 |        0.3686 |        0.0001 |        0.0298 |        0.0538 |         0.3347 |      0.0292 |
|                  0.6 | foldseek_tmscore60_cc_test | B3 contact only   |           21555 |      11869 |                     2 |               1026 |        0.3947 |        0      |        0.0332 |        0.3985 |         0.7364 |      0.2731 |
|                  0.6 | foldseek_tmscore60_cc_test | Contact-EC fusion |           21555 |      11869 |                     2 |               1026 |        0.4103 |        0      |        0.0295 |        0.4403 |         0.7036 |      0.3203 |

## Expected rows, including pending runs

|   tm_score_threshold | split                      | model             |   test_proteins |   clusters |   cluster_size_median |   cluster_size_max |   l1_accuracy |   l2_accuracy |   l3_accuracy |   l4_micro_f1 |   l4_precision |   l4_recall |
|---------------------:|:---------------------------|:------------------|----------------:|-----------:|----------------------:|-------------------:|--------------:|--------------:|--------------:|--------------:|---------------:|------------:|
|                  0.4 | foldseek_tmscore40_cc_test | B1 ESM-2 only     |           21617 |      10893 |                     2 |               1026 |        0.4108 |        0.0701 |        0.0548 |        0.0683 |         0.4115 |      0.0372 |
|                  0.4 | foldseek_tmscore40_cc_test | B3 contact only   |           21617 |      10893 |                     2 |               1026 |        0.4198 |        0.0798 |        0.0584 |        0.2622 |         0.7039 |      0.1611 |
|                  0.4 | foldseek_tmscore40_cc_test | Contact-EC fusion |           21617 |      10893 |                     2 |               1026 |        0.4115 |        0.0879 |        0.0564 |        0.2981 |         0.6311 |      0.1951 |
|                  0.5 | foldseek_tmscore50_cc_test | B1 ESM-2 only     |           16725 |       3813 |                     2 |              18692 |        0.4692 |        0.0356 |        0.0132 |        0.0462 |         0.3759 |      0.0246 |
|                  0.5 | foldseek_tmscore50_cc_test | B3 contact only   |           16725 |       3813 |                     2 |              18692 |        0.3651 |        0.0097 |        0.0029 |        0.0337 |         0.1984 |      0.0184 |
|                  0.5 | foldseek_tmscore50_cc_test | Contact-EC fusion |           16725 |       3813 |                     2 |              18692 |        0.5366 |        0.031  |        0.005  |        0.0998 |         0.5794 |      0.0546 |
|                  0.6 | foldseek_tmscore60_cc_test | B1 ESM-2 only     |           21555 |      11869 |                     2 |               1026 |        0.3686 |        0.0001 |        0.0298 |        0.0538 |         0.3347 |      0.0292 |
|                  0.6 | foldseek_tmscore60_cc_test | B3 contact only   |           21555 |      11869 |                     2 |               1026 |        0.3947 |        0      |        0.0332 |        0.3985 |         0.7364 |      0.2731 |
|                  0.6 | foldseek_tmscore60_cc_test | Contact-EC fusion |           21555 |      11869 |                     2 |               1026 |        0.4103 |        0      |        0.0295 |        0.4403 |         0.7036 |      0.3203 |

## Interpretation

- TM-score 0.40 is the looser structural clustering threshold; 0.60 is the stricter threshold.
- The key question is whether Contact-EC fusion retains a relative advantage as structural disjointness becomes stricter.
- Missing metric cells mean the corresponding long training/evaluation run has not finished yet.
