# Partial EC hierarchical prefix audit

Partial EC labels are scored only up to the deepest available known prefix.

## Counts

|   known_complete |   novel_complete |   partial |
|-----------------:|-----------------:|----------:|
|              124 |               35 |       309 |

## Contact-EC summary

| subset              |   L1_micro_f1 |   L2_micro_f1 |   L3_micro_f1 |   L4_micro_f1 |
|:--------------------|--------------:|--------------:|--------------:|--------------:|
| all_prefix_scorable |        0.8388 |        0.7694 |        0.7239 |        0.6032 |
| novel_complete      |        0.8438 |        0.7500 |        0.5484 |      nan      |
| partial             |        0.8259 |        0.7322 |        0.6629 |      nan      |

## All metrics

| model           | subset              |   level |   n |   micro_f1 |   precision |   recall |   any_hit_rate |   avg_pred_labels |
|:----------------|:--------------------|--------:|----:|-----------:|------------:|---------:|---------------:|------------------:|
| B1 (ESM-2)      | all_prefix_scorable |       1 | 466 |     0.8328 |      0.8600 |   0.8072 |         0.8176 |            0.9506 |
| B1 (ESM-2)      | all_prefix_scorable |       2 | 387 |     0.6545 |      0.8120 |   0.5482 |         0.5581 |            0.6873 |
| B1 (ESM-2)      | all_prefix_scorable |       3 | 359 |     0.5749 |      0.7804 |   0.4550 |         0.4652 |            0.5961 |
| B1 (ESM-2)      | all_prefix_scorable |       4 | 124 |     0.4216 |      0.8431 |   0.2810 |         0.2903 |            0.4113 |
| B1 (ESM-2)      | known_complete      |       1 | 124 |     0.8642 |      0.9052 |   0.8268 |         0.8468 |            0.9355 |
| B1 (ESM-2)      | known_complete      |       2 | 124 |     0.7431 |      0.9000 |   0.6328 |         0.6532 |            0.7258 |
| B1 (ESM-2)      | known_complete      |       3 | 124 |     0.7315 |      0.9080 |   0.6124 |         0.6371 |            0.7016 |
| B1 (ESM-2)      | known_complete      |       4 | 124 |     0.4216 |      0.8431 |   0.2810 |         0.2903 |            0.4113 |
| B1 (ESM-2)      | partial             |       1 | 309 |     0.8311 |      0.8567 |   0.8071 |         0.8123 |            0.9482 |
| B1 (ESM-2)      | partial             |       2 | 230 |     0.5805 |      0.7483 |   0.4741 |         0.4783 |            0.6391 |
| B1 (ESM-2)      | partial             |       3 | 202 |     0.4656 |      0.7030 |   0.3480 |         0.3515 |            0.5000 |
| B1 (ESM-2)      | partial             |       4 |   0 |   nan      |    nan      | nan      |       nan      |          nan      |
| B1 (ESM-2)      | novel_complete      |       1 |  33 |     0.7353 |      0.7353 |   0.7353 |         0.7576 |            1.0303 |
| B1 (ESM-2)      | novel_complete      |       2 |  33 |     0.7937 |      0.8621 |   0.7353 |         0.7576 |            0.8788 |
| B1 (ESM-2)      | novel_complete      |       3 |  33 |     0.5667 |      0.6538 |   0.5000 |         0.5152 |            0.7879 |
| B1 (ESM-2)      | novel_complete      |       4 |   0 |   nan      |    nan      | nan      |       nan      |          nan      |
| B3 (Contact)    | all_prefix_scorable |       1 | 466 |     0.8158 |      0.8741 |   0.7648 |         0.7747 |            0.8863 |
| B3 (Contact)    | all_prefix_scorable |       2 | 387 |     0.7225 |      0.8571 |   0.6244 |         0.6357 |            0.7416 |
| B3 (Contact)    | all_prefix_scorable |       3 | 359 |     0.6568 |      0.8264 |   0.5450 |         0.5571 |            0.6741 |
| B3 (Contact)    | all_prefix_scorable |       4 | 124 |     0.3646 |      0.8974 |   0.2288 |         0.2097 |            0.3145 |
| B3 (Contact)    | known_complete      |       1 | 124 |     0.8560 |      0.8966 |   0.8189 |         0.8387 |            0.9355 |
| B3 (Contact)    | known_complete      |       2 | 124 |     0.8052 |      0.9029 |   0.7266 |         0.7500 |            0.8306 |
| B3 (Contact)    | known_complete      |       3 | 124 |     0.7671 |      0.9333 |   0.6512 |         0.6774 |            0.7258 |
| B3 (Contact)    | known_complete      |       4 | 124 |     0.3646 |      0.8974 |   0.2288 |         0.2097 |            0.3145 |
| B3 (Contact)    | partial             |       1 | 309 |     0.7930 |      0.8636 |   0.7331 |         0.7379 |            0.8544 |
| B3 (Contact)    | partial             |       2 | 230 |     0.6615 |      0.8258 |   0.5517 |         0.5565 |            0.6739 |
| B3 (Contact)    | partial             |       3 | 202 |     0.5933 |      0.7886 |   0.4755 |         0.4802 |            0.6089 |
| B3 (Contact)    | partial             |       4 |   0 |   nan      |    nan      | nan      |       nan      |          nan      |
| B3 (Contact)    | novel_complete      |       1 |  33 |     0.8657 |      0.8788 |   0.8529 |         0.8788 |            1.0000 |
| B3 (Contact)    | novel_complete      |       2 |  33 |     0.7937 |      0.8621 |   0.7353 |         0.7576 |            0.8788 |
| B3 (Contact)    | novel_complete      |       3 |  33 |     0.6032 |      0.6552 |   0.5588 |         0.5758 |            0.8788 |
| B3 (Contact)    | novel_complete      |       4 |   0 |   nan      |    nan      | nan      |       nan      |          nan      |
| Contact-EC      | all_prefix_scorable |       1 | 466 |     0.8388 |      0.8632 |   0.8157 |         0.8219 |            0.9571 |
| Contact-EC      | all_prefix_scorable |       2 | 387 |     0.7694 |      0.8690 |   0.6904 |         0.6977 |            0.8088 |
| Contact-EC      | all_prefix_scorable |       3 | 359 |     0.7239 |      0.8281 |   0.6431 |         0.6546 |            0.7939 |
| Contact-EC      | all_prefix_scorable |       4 | 124 |     0.6032 |      0.7677 |   0.4967 |         0.5242 |            0.7984 |
| Contact-EC      | known_complete      |       1 | 124 |     0.8685 |      0.8790 |   0.8583 |         0.8710 |            1.0000 |
| Contact-EC      | known_complete      |       2 | 124 |     0.8390 |      0.9167 |   0.7734 |         0.7903 |            0.8710 |
| Contact-EC      | known_complete      |       3 | 124 |     0.8608 |      0.9444 |   0.7907 |         0.8145 |            0.8710 |
| Contact-EC      | known_complete      |       4 | 124 |     0.6032 |      0.7677 |   0.4967 |         0.5242 |            0.7984 |
| Contact-EC      | partial             |       1 | 309 |     0.8259 |      0.8527 |   0.8006 |         0.8026 |            0.9450 |
| Contact-EC      | partial             |       2 | 230 |     0.7322 |      0.8514 |   0.6422 |         0.6435 |            0.7609 |
| Contact-EC      | partial             |       3 | 202 |     0.6629 |      0.7852 |   0.5735 |         0.5792 |            0.7376 |
| Contact-EC      | partial             |       4 |   0 |   nan      |    nan      | nan      |       nan      |          nan      |
| Contact-EC      | novel_complete      |       1 |  33 |     0.8438 |      0.9000 |   0.7941 |         0.8182 |            0.9091 |
| Contact-EC      | novel_complete      |       2 |  33 |     0.7500 |      0.8000 |   0.7059 |         0.7273 |            0.9091 |
| Contact-EC      | novel_complete      |       3 |  33 |     0.5484 |      0.6071 |   0.5000 |         0.5152 |            0.8485 |
| Contact-EC      | novel_complete      |       4 |   0 |   nan      |    nan      | nan      |       nan      |          nan      |
| Contact-EC-Hier | all_prefix_scorable |       1 | 466 |     0.8562 |      0.8884 |   0.8263 |         0.8348 |            0.9421 |
| Contact-EC-Hier | all_prefix_scorable |       2 | 387 |     0.8016 |      0.8694 |   0.7437 |         0.7545 |            0.8708 |
| Contact-EC-Hier | all_prefix_scorable |       3 | 359 |     0.7474 |      0.8278 |   0.6812 |         0.6936 |            0.8412 |
| Contact-EC-Hier | all_prefix_scorable |       4 | 124 |     0.5690 |      0.8354 |   0.4314 |         0.4435 |            0.6371 |
| Contact-EC-Hier | known_complete      |       1 | 124 |     0.8988 |      0.9250 |   0.8740 |         0.8952 |            0.9677 |
| Contact-EC-Hier | known_complete      |       2 | 124 |     0.8583 |      0.8908 |   0.8281 |         0.8548 |            0.9597 |
| Contact-EC-Hier | known_complete      |       3 | 124 |     0.8750 |      0.9459 |   0.8140 |         0.8468 |            0.8952 |
| Contact-EC-Hier | known_complete      |       4 | 124 |     0.5690 |      0.8354 |   0.4314 |         0.4435 |            0.6371 |
| Contact-EC-Hier | partial             |       1 | 309 |     0.8409 |      0.8776 |   0.8071 |         0.8091 |            0.9256 |
| Contact-EC-Hier | partial             |       2 | 230 |     0.7685 |      0.8610 |   0.6940 |         0.6957 |            0.8130 |
| Contact-EC-Hier | partial             |       3 | 202 |     0.6975 |      0.7853 |   0.6275 |         0.6287 |            0.8069 |
| Contact-EC-Hier | partial             |       4 |   0 |   nan      |    nan      | nan      |       nan      |          nan      |
| Contact-EC-Hier | novel_complete      |       1 |  33 |     0.8358 |      0.8485 |   0.8235 |         0.8485 |            1.0000 |
| Contact-EC-Hier | novel_complete      |       2 |  33 |     0.8000 |      0.8387 |   0.7647 |         0.7879 |            0.9394 |
| Contact-EC-Hier | novel_complete      |       3 |  33 |     0.5484 |      0.6071 |   0.5000 |         0.5152 |            0.8485 |
| Contact-EC-Hier | novel_complete      |       4 |   0 |   nan      |    nan      | nan      |       nan      |          nan      |
