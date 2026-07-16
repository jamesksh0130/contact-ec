# Late fusion baseline audit

| dataset                  | model                     |   n |   micro_f1 |   weighted_f1 |   macro_f1 |   precision |   recall |
|:-------------------------|:--------------------------|----:|-----------:|--------------:|-----------:|------------:|---------:|
| Swiss-Prot 2023 known EC | B1 (ESM-2)                | 124 |     0.4216 |        0.3178 |     0.0031 |      0.8431 |   0.281  |
| Swiss-Prot 2023 known EC | B3 (Contact)              | 124 |     0.3646 |        0.2734 |     0.0021 |      0.8974 |   0.2288 |
| Swiss-Prot 2023 known EC | Contact-EC                | 124 |     0.6032 |        0.5026 |     0.0065 |      0.7677 |   0.4967 |
| Swiss-Prot 2023 known EC | Late mean B1+B3           | 124 |     0.377  |        0.2769 |     0.0022 |      0.9474 |   0.2353 |
| Swiss-Prot 2023 known EC | Late max B1+B3            | 124 |     0.4673 |        0.3606 |     0.0035 |      0.8197 |   0.3268 |
| Swiss-Prot 2023 known EC | Late weighted 0.7B1+0.3B3 | 124 |     0.4322 |        0.3207 |     0.003  |      0.9348 |   0.281  |
| Swiss-Prot 2023 known EC | Late weighted 0.3B1+0.7B3 | 124 |     0.3598 |        0.2674 |     0.0021 |      0.9444 |   0.2222 |
| Price-149                | B1 (ESM-2)                | 136 |     0.0324 |        0.0196 |     0.0006 |      0.0612 |   0.0221 |
| Price-149                | B3 (Contact)              | 136 |     0      |        0      |     0      |      0      |   0      |
| Price-149                | Contact-EC                | 136 |     0.25   |        0.2304 |     0.0023 |      0.3182 |   0.2059 |
| Price-149                | Late mean B1+B3           | 136 |     0.0118 |        0.0074 |     0.0002 |      0.0294 |   0.0074 |
| Price-149                | Late max B1+B3            | 136 |     0.0312 |        0.0196 |     0.0006 |      0.0536 |   0.0221 |
| Price-149                | Late weighted 0.7B1+0.3B3 | 136 |     0.0231 |        0.0147 |     0.0004 |      0.0541 |   0.0147 |
| Price-149                | Late weighted 0.3B1+0.7B3 | 136 |     0      |        0      |     0      |      0      |   0      |

## Interpretation

- Learned Contact-EC improves over simple B1/B3 posterior averaging on the temporal known-EC subset.
- Late max can increase recall on Price-149 but remains an OOD workaround, not a solution.
- The result supports keeping learned fusion in the manuscript while acknowledging that a separately trained concat/gated-MLP baseline remains future work.
