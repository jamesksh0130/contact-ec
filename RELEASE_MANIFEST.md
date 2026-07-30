# GitHub Upload Manifest

Prepared on: 2026-07-30

Package:

- Folder: `github_release/contact-ec/`
- Zip: `github_release/contact-ec-github-upload.zip`
- Approximate folder size before compression: 48 MB
- Files in zip package: 352

Contents:

- Source code: `train.py`, `evaluate.py`, `visualize.py`, `models/`, `scripts/`
- Configurations: `configs/`
- Manuscript PDFs and LaTeX source: `paper/`
- Selected results: `outputs/results/`
- HIT-EC vs Contact-EC case-wise comparison: `outputs/results/casewise_hitec_contactec.*`
- Contact-EC threshold sensitivity analysis: `outputs/results/contactec_threshold_sensitivity.*`
- Reliability audit artifacts: `outputs/audit/`
- Price-149 failure-mode audit: `scripts/audit_price149_failure_analysis.py`,
  `outputs/audit/price149_failure_*`, and
  `outputs/figures/price149_failure_breakdown.png`
- Late-fusion baseline audit: `scripts/audit_late_fusion_baselines.py`,
  `outputs/audit/late_fusion_baseline_*`, and
  `outputs/figures/late_fusion_baselines.png`
- Contact-pair nearest-neighbor audit: `scripts/audit_contact_pair_neighbors.py`,
  `outputs/audit/contact_pair_neighbor_*`, and
  `outputs/figures/contact_pair_neighbor_hist.png`
- Raw contact-map topology nearest-neighbor audit:
  `scripts/audit_contact_topology_neighbors.py`,
  `outputs/audit/contact_topology_neighbor_*`, and
  `outputs/figures/contact_topology_neighbor_audit.png`
- Partial-EC hierarchical prefix audit: `scripts/audit_partial_ec_hierarchical.py`,
  `outputs/audit/partial_ec_hierarchical_*`, and
  `outputs/figures/partial_ec_hierarchical_f1.png`
- Contact-EC calibration and prediction-cardinality audit:
  `scripts/audit_contactec_calibration.py`,
  `outputs/audit/contactec_calibration_audit.*`, and
  `outputs/figures/contactec_calibration_diagnostics.png`
- Foldseek/TM-score-disjoint split and robustness audit:
  `scripts/foldseek_structure_split.py`,
  `scripts/build_foldseek_tmscore_sweep_splits.sh`,
  `scripts/run_foldseek_tmscore_sweep_training.sh`,
  `scripts/collect_foldseek_tmscore_sweep.py`,
  `scripts/audit_foldseek_*`,
  `outputs/audit/foldseek_tmscore50_cc_*`, and
  `outputs/audit/foldseek_tmscore_sweep_*`
- Three-seed temporal known-label repeats:
  `scripts/run_temporal_known_seed_repeats.sh`,
  `scripts/collect_temporal_known_seed_repeats.py`, and
  `outputs/audit/temporal_known_seed_repeats.*`
- Trainable simple fusion architecture controls:
  `models/fusion_simple_baselines.py`,
  `scripts/run_simple_fusion_baseline_seed_repeats.sh`,
  `scripts/collect_simple_fusion_seed_repeats.py`, and
  `outputs/audit/simple_fusion_seed_repeats.*`
- Recency cutoff and matched-intersection audit:
  `scripts/audit_recency_cutoff_decomposition.py`,
  `scripts/build_recency_intersection_ids.py`,
  `scripts/collect_recency_intersection_eval.py`,
  `outputs/audit/recency_cutoff_decomposition.*`,
  `outputs/audit/recency_intersection_ids.md`, and
  `outputs/audit/recency_intersection_eval.*`
- Open-vocabulary, abstention, label-difficulty, and fusion-rescue audits:
  `outputs/audit/foldseek_tmscore50_cc_open_set_confidence.*`,
  `outputs/audit/foldseek_tmscore50_cc_fusion_abstention_fallback.*`,
  `outputs/audit/foldseek_tmscore50_cc_label_difficulty_audit.*`, and
  `outputs/audit/foldseek_tmscore50_cc_fusion_rescue_case_studies.*`
- Manuscript figures: `outputs/figures/`
- GitHub documentation: `README.md`, `docs/`, `CITATION.cff`, `requirements.txt`, `.gitignore`

Excluded by design:

- Raw datasets
- Structure files
- ESM-2 embedding caches
- Contact-map caches
- Model checkpoints
- Third-party baseline repositories
- Logs and temporary files

Required updates before public release:

- MIT license included.
- Author/email/repository metadata have been populated; update only if the public repository path changes.
- Add DOI or archive links for data and checkpoints.
- Run a fresh-clone smoke test after external data paths are configured.
