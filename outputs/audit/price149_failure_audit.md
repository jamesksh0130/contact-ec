# Price-149 failure audit

## Coverage

| category        | measure                           |   value |
|:----------------|:----------------------------------|--------:|
| raw_price149    | proteins                          |     149 |
| encoded_subset  | proteins                          |     136 |
| raw_price149    | encoded_in_sp2018_l4_vocab_all_ec |     139 |
| raw_price149    | encoded_in_sp2018_l4_vocab_any_ec |     139 |
| assets          | embedding_exists                  |     149 |
| assets          | contact_map_exists                |     149 |
| assets          | contact_map_nonzero               |     149 |
| sequence_length | median_raw                        |     389 |
| sequence_length | max_raw                           |     700 |
| sequence_length | raw_len_gt_1024                   |       0 |
| label_space     | unique_l4_raw                     |      56 |
| label_space     | unique_l4_in_sp2018_vocab         |      51 |
| label_space     | unique_l4_out_of_sp2018_vocab     |       5 |

## Metrics

| model                     | setting                 |   n | level   |   micro_f1 |   weighted_f1 |   precision |   recall |
|:--------------------------|:------------------------|----:|:--------|-----------:|--------------:|------------:|---------:|
| B1 (ESM-2, SP-2018)       | canonical flat Level-4  | 136 | L4      |     0.0324 |        0.0196 |      0.0612 |   0.0221 |
| B3 (Contact, SP-2018)     | canonical flat Level-4  | 136 | L4      |     0      |        0      |      0      |   0      |
| Contact-EC flat (SP-2018) | canonical flat Level-4  | 136 | L4      |     0.25   |        0.2304 |      0.3182 |   0.2059 |
| Contact-EC ESMFold-map    | ESMFold contact maps    | 139 | L4      |     0.1218 |        0.108  |    nan      | nan      |
| Contact-EC hierarchical   | hierarchical diagnostic | 145 | L1      |     0.9379 |        0.9371 |    nan      | nan      |
| Contact-EC hierarchical   | hierarchical diagnostic | 145 | L2      |     0.7862 |        0.794  |    nan      | nan      |
| Contact-EC hierarchical   | hierarchical diagnostic | 145 | L3      |     0.7655 |        0.7775 |    nan      | nan      |
| Contact-EC hierarchical   | hierarchical diagnostic | 145 | L4      |     0.0508 |      nan      |      0.102  |   0.0338 |

## Level-1 distribution

|   l1 |   n |   median_len |   mean_len |
|-----:|----:|-------------:|-----------:|
|    1 |  54 |        383   |    412.926 |
|    2 |  28 |        397   |    376.929 |
|    3 |  20 |        310   |    326.5   |
|    4 |  22 |        405   |    441.909 |
|    5 |   4 |        254.5 |    262     |
|    6 |   8 |        458   |    460.25  |

## Interpretation

- Contact-EC improves over the SP-2018 ESM-2 baseline on canonical Price-149 Level-4 evaluation (0.2500 vs. 0.0324 micro F1), but remains below CLEAN-Contact.
- Contact-only transfer is 0.0000, so the current structural branch is not robust on this external bacterial RefSeq benchmark.
- Hierarchical diagnostics show strong coarse-family signal (L1 0.9379; L2 0.7862; L3 0.7655) but weak exact Level-4 specificity (0.0508).
- ESMFold-derived contact maps do not recover the gap, supporting the interpretation that external structure-source/preprocessing shift requires retraining or calibration.
