# Reproducibility Notes

This document summarizes the intended reproduction workflow. Paths may need adjustment depending on where external data and checkpoints are stored.

## 1. Prepare data

```bash
python scripts/01_download_uniprot.py --config configs/config.yaml
python scripts/02_preprocess_labels.py --config configs/config.yaml
python scripts/03_download_pdb_parallel.py --config configs/config.yaml
python scripts/04_build_contact_maps.py --config configs/config.yaml
python scripts/05_extract_esm2_embeddings.py --config configs/config.yaml
python scripts/06_combine_datasets.py --config configs/config.yaml
python scripts/06_extract_contact_pair_embeddings.py --config configs/config.yaml
```

For EC-Bench-style experiments, use `configs/config_ecbench.yaml` and the `scripts/ecbench_*` pipeline scripts.

## 2. Train representative models

```bash
bash scripts/train_b1_multilabel.sh
bash scripts/train_b3_multilabel.sh
bash scripts/train_v2_multilabel.sh
```

For cached phase-2 or end-to-end experiments:

```bash
python scripts/train_phase2_cached.py --config configs/config_ecbench.yaml
python scripts/train_flatfc_e2e_cached.py --config configs/config_expa_e2e.yaml
```

## 3. Evaluate

```bash
python scripts/eval_full_testset.py --config configs/config.yaml
python scripts/eval_ecbench_full468.py --config configs/config_ecbench.yaml
python scripts/eval_new392_clean.py
python scripts/run_stats.py --config configs/config.yaml
```

## 4. Audit reliability

```bash
python scripts/audit_reliability.py
python scripts/collect_result_tables.py
python scripts/audit_recency_homology_coverage.py --threads 24
```

Read these outputs before updating the manuscript:

- `outputs/audit/reliability_report.md`
- `outputs/audit/paper_consistency_audit.md`
- `outputs/audit/all_result_metrics.csv`
- `outputs/audit/paper_metric_candidates.csv`
- `outputs/audit/recency_homology_coverage.md`

## 5. Generate figures

```bash
python scripts/generate_figures.py
python scripts/regenerate_journal_figures.py
python scripts/gradcam_visualize.py
```

Generated manuscript figures are included under `outputs/figures/`.

## Important note

The exact commands used during local experimentation may include machine-specific paths and long-running GPU jobs. Before public release, verify each command from a fresh clone with externally hosted data paths.
