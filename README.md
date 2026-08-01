# Contact-EC

Code and reproducibility artifacts for:

**Dissecting Sequence, Structure, and Data Recency Effects in Enzyme Commission Prediction under Temporal Distribution Shift**

Contact-EC is a sequence-structure fusion framework for hierarchical, multi-label Enzyme Commission (EC) prediction. The repository contains training/evaluation code, model definitions, configuration files, paper figures, audit reports, and the current Bioinformatics-style manuscript draft.

## What is included

- `models/`: PyTorch model components for ESM-2 baselines, contact-map encoders, and fusion models.
- `scripts/`: data preparation, embedding extraction, training, evaluation, ablation, statistics, and figure-generation scripts.
- `configs/`: experiment configurations for the main split, EC-Bench, 3B ESM-2, and ExpA experiments.
- `outputs/results/`: selected JSON result files used in the manuscript.
- `outputs/results/casewise_hitec_contactec.*`: per-protein comparison between HIT-EC and Contact-EC on the 124 known-label temporal proteins.
- `outputs/results/contactec_threshold_sensitivity.*`: fixed-threshold, global-threshold, and top-1 fallback calibration analysis on the same temporal subset.
- `outputs/audit/`: overlap, asset-coverage, label-statistics, Foldseek-disjoint,
  seed-repeat, fusion-baseline, recency-intersection, open-vocabulary, and
  reliability audit outputs.
- `outputs/figures/`: manuscript figures.
- `paper/pdf/`: current main manuscript and supplementary PDF.
- `paper/source/`: LaTeX source for the current manuscript draft.

Large raw datasets, ESM-2 embedding caches, AlphaFold/PDB structures, contact-map arrays, and trained checkpoints are intentionally not included.

## Main reported results

Selected Level-4 micro F1 values from the current manuscript:

| Setting | Model | Micro F1 |
|---|---:|---:|
| Temporal Swiss-Prot 2023-01 holdout | ESM-2 650M only | 0.4508 +/- 0.0203 |
| Temporal Swiss-Prot 2023-01 holdout | Contact map only | 0.4244 +/- 0.0207 |
| Temporal Swiss-Prot 2023-01 holdout | Contact-EC fusion | 0.6241 +/- 0.0170 |
| Temporal Swiss-Prot 2023-01 holdout | Contact-EC 3B | 0.6316 |
| Temporal Swiss-Prot 2023-01 holdout | HIT-EC baseline | 0.8471 |
| Temporal Swiss-Prot 2023-01 holdout | Concat fusion control | 0.5922 +/- 0.0150 |
| Temporal recency intersection, N=99 | Contact-EC SP-2018 | 0.6467 +/- 0.0210 |
| Temporal recency intersection, N=99 | Contact-EC-ExpA SP-2026 | 0.7417 +/- 0.0182 |
| Sequence-disjoint EC-Bench hard validation | ESM-2 650M only | 0.7655 |
| Sequence-disjoint EC-Bench hard validation | Contact map only | 0.7650 |
| Sequence-disjoint EC-Bench hard validation | Contact-EC fusion | 0.8879 |
| Foldseek/TM-score 0.50 test, validation-selected threshold | ESM-2 650M only | 0.0607 +/- 0.0009 |
| Foldseek/TM-score 0.50 test, validation-selected threshold | Contact map only | 0.0654 +/- 0.0016 |
| Foldseek/TM-score 0.50 test, validation-selected threshold | Contact-EC fusion | 0.1685 +/- 0.0334 |

See `outputs/audit/reliability_report.md`,
`outputs/audit/paper_consistency_audit.md`, and
`outputs/audit/recency_intersection_eval.md` before making new claims from these
results.

The case-wise HIT-EC comparison shows that both HIT-EC and Contact-EC recover at
least one correct EC label for 63/124 temporal proteins, HIT-EC alone recovers 45,
Contact-EC alone recovers 2, and both miss 14. This is used in the manuscript to
frame Contact-EC as a decomposition model rather than a replacement for HIT-EC.

The threshold sensitivity analysis shows that Contact-EC reaches micro F1 0.6032
with the fixed 0.5 cutoff and 0.6167 with the best global cutoff of 0.65. The
simple fusion seed-repeat audit shows that concat/sum/gated-MLP controls remain
below Contact-EC, while the Foldseek audit shows that absolute closed-set
Level-4 performance under fold-level shift is still low.

## Installation

Create a Python environment, then install the core dependencies:

```bash
pip install -r requirements.txt
```

External tools and resources used by some scripts:

- CUDA-capable PyTorch environment for training and embedding extraction.
- ESM-2 weights from Hugging Face or FAIR ESM.
- AlphaFold/PDB structure files for contact-map construction.
- MMseqs2 for sequence-disjoint split checks where applicable.
- Foldseek for the TM-score-disjoint structural split audit.

## Typical workflow

The end-to-end workflow used for the paper is:

1. Download and preprocess UniProt/Swiss-Prot EC annotations.
2. Build temporal, random, and sequence-disjoint splits.
3. Download or map available protein structures.
4. Build 3-channel contact maps.
5. Extract ESM-2 embeddings.
6. Train ESM-only, contact-only, and fusion models.
7. Evaluate on temporal and EC-Bench-style splits.
8. Run reliability audits, statistical tests, and figure-generation scripts.

Representative commands are documented in `docs/REPRODUCIBILITY.md`.

## Current caveats

- This release is a clean upload package, not the full local training workspace.
- Checkpoints are not included because they are large. Add download links in `docs/DATA_AND_CHECKPOINTS.md` after hosting them.
- Raw data and derived embedding/contact-map caches are not included. Recreate them from scripts or provide external archive links.
- The paper source contains the current author metadata and repository URL.

## Citation

If this repository is useful, cite the manuscript after publication. `CITATION.cff` includes the current author metadata and repository URL, and should be updated with the final DOI after publication.

## License

This repository is released under the MIT License. See `LICENSE`.
