# HIT-EC vs Contact-EC Case-wise Comparison

N = 124 known-label Swiss-Prot 2023-01 proteins.
Contact-EC recomputed micro F1 = 0.6032.
HIT-EC reported micro F1 = 0.8471.

## Case Status Counts

| Status | Count |
|---|---:|
| both_correct | 63 |
| hitec_only | 45 |
| contactec_only | 2 |
| both_wrong | 14 |

## Level-1 Status Counts

|   true_l1 |   both_correct |   both_wrong |   contactec_only |   hitec_only |
|----------:|---------------:|-------------:|-----------------:|-------------:|
|         1 |             12 |            0 |                1 |            5 |
|         2 |             13 |            3 |                1 |            8 |
|         3 |             29 |            2 |                0 |           11 |
|         4 |              5 |            7 |                0 |           20 |
|         5 |              4 |            2 |                0 |            1 |

## Representative Contact-EC-only Cases

- A0A3G9HRC2: true=1.14.14.1;1.6.2.4; Contact-EC=1.14.14.1;1.6.2.4; HIT-EC=3.2.1.183
- A0A6C0WW38: true=2.1.1.160; Contact-EC=2.1.1.160; HIT-EC=2.1.1.159

## Representative HIT-EC-only Cases

- C0HLB8: true=3.1.1.4; HIT-EC=3.1.1.4; Contact-EC top5=3.1.1.4:0.4931;3.2.1.8:0.0169;3.2.1.14:0.0139;3.2.1.4:0.0059;3.2.1.17:0.0038
- P0DSN5: true=3.1.1.4; HIT-EC=3.1.1.4; Contact-EC top5=1.4.3.2:0.1990;1.11.1.7:0.0628;2.7.7.6:0.0461;2.1.3.3:0.0297;2.7.7.48:0.0189
- A0A3L6G998: true=4.2.3.57; HIT-EC=4.2.3.57; Contact-EC top5=4.2.3.47:0.2375;4.2.3.75:0.1390;4.2.3.13:0.1375;4.2.3.61:0.1284;4.2.3.87:0.0939
- A0A2S1WBY6: true=2.1.1.117; HIT-EC=2.1.1.117; Contact-EC top5=2.1.1.46:0.1954;2.1.1.116:0.1845;2.1.1.4:0.1717;2.1.1.212:0.1145;2.1.1.146:0.0743
- A0A2I7G3B3: true=1.1.1.144;1.1.1.347; HIT-EC=1.1.1.144;1.1.1.347; Contact-EC top5=1.1.1.1:0.9190;1.1.1.284:0.8166;1.1.1.347:0.0189;1.1.1.354:0.0026;1.1.1.2:0.0006
