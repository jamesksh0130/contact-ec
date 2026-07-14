# Temporal Leakage Audit

Audit against the full evaluable Swiss-Prot 2023-01 temporal subset.

| Training corpus | Train rows | Test rows | Accession overlap | Exact sequence overlap | Train L4 labels | Test L4 labels |
|---|---:|---:|---:|---:|---:|---:|
| SP-2018 EC-Bench train | 201865 | 124 | 0 | 0 | 4284 | 60 |
| ExpA recent Swiss-Prot train | 243198 | 124 | 0 | 0 | 5145 | 60 |

Interpretation: accession and exact-sequence overlap are necessary but not sufficient leakage checks.
They do not replace homolog- or fold-disjoint evaluation.
