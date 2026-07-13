# Contact-EC Threshold Sensitivity

N = 124 known-label Swiss-Prot 2023-01 temporal proteins.
Checkpoint = `/home/user/Desktop/unlv/outputs/checkpoints/ecbench_b4_flatfc_best.pt`.

## Key Results

| Setting | Threshold | Micro F1 | Weighted F1 | Precision | Recall | Avg preds/protein | Empty predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed global | 0.50 | 0.6032 | 0.5026 | 0.7677 | 0.4967 | 0.80 | 45 |
| Best global by micro F1 | 0.65 | 0.6167 | 0.4895 | 0.8506 | 0.4837 | 0.70 | 54 |
| 0.5 or top-1 if empty | 0.50 | 0.5589 | 0.5413 | 0.5764 | 0.5425 | 1.16 | 0 |
| Top-1 only | - | 0.5126 | 0.4628 | 0.5726 | 0.4641 | 1.00 | 0 |

## Near Misses at Threshold 0.5

| UID | Top-1 EC | Probability |
|---|---|---:|
| C0HLB8 | 3.1.1.4 | 0.4931 |
| A0A3G1QTS7 | 4.2.3.19 | 0.4195 |
| P9WEW0 | 4.2.3.135 | 0.3554 |
| A0A2K9RFY0 | 4.2.3.19 | 0.3160 |
| P9WEW1 | 4.2.3.135 | 0.2574 |
| P9WEW2 | 4.2.3.135 | 0.2235 |
| P0DUB9 | 3.1.1.3 | 0.0645 |
