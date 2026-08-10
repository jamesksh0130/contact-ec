# HIT-EC SP-2024 Comparison Audit Manifest

This manifest records the files needed to reproduce and inspect the HIT-EC
comparison used for the SP-2024 temporal evaluation in the Contact-EC
manuscript. The purpose is to make the reported HIT-EC decrease auditable with
the exact prediction outputs, label-vocabulary stratification, and evaluation
scripts used in the paper.

## Scope

- Evaluation set: SP-2024 complete known-EC temporal holdout.
- Reported comparison: Contact-EC versus HIT-EC under the mapped Level-4 EC
  evaluation protocol.
- Key reported result: HIT-EC micro F1 of 0.4578 on SP-2024, with
  vocabulary-stratified analysis showing that the decrease is not explained
  solely by HIT-EC label-vocabulary mismatch.

## Primary Output Files

- `outputs/results/hitec_sp2024.json`
  - HIT-EC SP-2024 prediction/evaluation output.
- `outputs/results/hitec_sp2024_union.json`
  - HIT-EC SP-2024 evaluation using the union label space used for mapped
    comparison.
- `outputs/results/hitec_sp2024_vocab_stratified.json`
  - HIT-EC vocabulary-intersection audit for SP-2024.
- `outputs/results/sp2024_seed_repeats.json`
  - Contact-EC SP-2024 seed-repeat summary used for the three-seed mean.
- `outputs/results/sp2024_bootstrap_ci.json`
  - Bootstrap confidence interval summary for SP-2024.
- `outputs/results/sp2024_mmseqs_top1_metrics.json`
  - MMseqs2 top-hit homology-transfer baseline on SP-2024.
- `outputs/results/sp2024_mmseqs_top1_per_sample.csv`
  - Per-protein MMseqs2 top-hit predictions for SP-2024.

## Related SP-2023 HIT-EC Audit Files

- `outputs/results/hitec_eval.json`
  - HIT-EC evaluation on the Swiss-Prot 2023-01 complete known-EC temporal
    subset.
- `outputs/results/casewise_hitec_contactec.csv`
  - Per-protein HIT-EC versus Contact-EC case-wise comparison for SP-2023-01.
- `outputs/results/casewise_hitec_contactec_summary.json`
  - Machine-readable summary of the case-wise comparison.
- `outputs/results/casewise_hitec_contactec_summary.md`
  - Human-readable case-wise comparison summary.

## Supporting Audit Files

- `outputs/audit/recency_intersection_eval.csv`
- `outputs/audit/recency_intersection_eval.json`
- `outputs/audit/recency_intersection_eval.md`
- `outputs/audit/recency_intersection_ids.md`
- `outputs/audit/recency_cutoff_decomposition.csv`
- `outputs/audit/recency_cutoff_decomposition.json`
- `outputs/audit/recency_cutoff_decomposition.md`
- `outputs/audit/temporal_leakage_audit_final.csv`
- `outputs/audit/temporal_leakage_audit_final.json`
- `outputs/audit/temporal_leakage_audit_final.md`

These files document the data-recency decomposition, accession/sequence leakage
checks, and intersection-based controls that support the temporal comparison.

## Evaluation Scripts

- `scripts/run_hitec_inference.py`
  - HIT-EC inference on the SP-2023-01 EC-Bench temporal set.
- `scripts/casewise_hitec_contactec.py`
  - Per-protein HIT-EC versus Contact-EC comparison on SP-2023-01.
- `scripts/ecbench_eval.py`
  - Shared EC-Bench Level-4 multilabel evaluation utility.
- `scripts/eval_ecbench_full468.py`
  - Evaluation helper for the full EC-Bench temporal set.
- `scripts/evaluate_mmseqs_homology_baseline.py`
  - MMseqs2 top-hit homology-transfer baseline evaluation.
- `scripts/collect_recency_intersection_eval.py`
  - Recency/intersection evaluation collection.
- `scripts/audit_recency_cutoff_decomposition.py`
  - Data-recency and cutoff decomposition audit.
- `scripts/audit_temporal_leakage_final.py`
  - Final temporal leakage audit.

## Reproduction Notes

The paper uses the repository tag `bmc-submission-v1` as the archived code
identifier. Run commands from the repository root after preparing the local data
and model checkpoints described in the README.

Representative commands:

```bash
python scripts/run_hitec_inference.py
python scripts/casewise_hitec_contactec.py
python scripts/collect_recency_intersection_eval.py
python scripts/audit_recency_cutoff_decomposition.py
python scripts/audit_temporal_leakage_final.py
python scripts/evaluate_mmseqs_homology_baseline.py
```

Expected key outputs:

- HIT-EC SP-2023-01 Level-4 micro F1: `0.8471`
- HIT-EC SP-2024 Level-4 micro F1: `0.4578`
- Contact-EC SP-2024 three-seed Level-4 micro F1:
  `0.6819 +/- 0.0026`
- HIT-EC all-in-vocabulary SP-2024 stratum: `0.5496`

## Interpretation Guardrail

These files are provided to audit the mapped comparison rather than to claim
that HIT-EC and Contact-EC differ only by modality. HIT-EC and Contact-EC also
differ in training data, label space, inference protocol, and calibration. The
manuscript therefore interprets the SP-2024 result as evidence that temporal EC
benchmarks must report label-vocabulary coverage, temporal cutoff, and
homology/structure controls, not as a causal proof that structure alone explains
the observed HIT-EC decrease.
