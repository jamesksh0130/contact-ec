# Paper Consistency Audit

Last checked: 2026-08-01

## Canonical Evaluation Rules

1. Use the full evaluable Swiss-Prot 2023-01 temporal set (`N=124`) for main temporal conclusions.
2. Treat older `N=101` evaluations as historical partial runs only.
3. Use multi-label thresholded Level-4 evaluation for main tables.
4. Exclude auxiliary argmax-only or stdout-only evaluations from main conclusions unless explicitly labeled.
5. Treat New-392 as a published comparability benchmark, not a clean temporal external benchmark, unless the evaluated model is cutoff-controlled.
6. Treat Price-149 as an OOD failure-mode benchmark for this pipeline because AlphaFold contact maps are unavailable for the bacterial RefSeq identifiers.

## Canonical Main Results

| Item | Canonical Value | Source |
|---|---:|---|
| B1 temporal micro F1 | 0.4216 | `outputs/results/eval_full_testset.json` |
| B3 temporal micro F1 | 0.3646 | `outputs/results/eval_full_testset.json` |
| Contact-EC flat temporal micro F1 | 0.6032 | `outputs/results/eval_full_testset.json` |
| Contact-EC-Hier temporal micro F1 | 0.5690 | `outputs/results/eval_full_testset.json` |
| Contact-EC-3B temporal micro F1 | 0.6316 | `outputs/results/contact_ec_3b_full_eval.json` |
| Contact-EC-ExpA recency-intersection micro F1 | 0.7417 +/- 0.0182 | `outputs/audit/recency_intersection_eval.md` |
| Contact-EC-E2E temporal micro F1 | 0.6703 | `outputs/results/expa_e2e_full_eval.json` |
| HIT-EC temporal micro F1 | 0.8471 | `outputs/results/hitec_eval.json` |
| Contact-EC flat val_hard micro F1 | 0.8879 | `outputs/results/ecbench_eval_ecbench_b4_flatfc_best.json` |
| B1 val_hard micro F1 | 0.7655 | `outputs/results/ecbench_eval_ecbench_b1_best.json` |
| B3 val_hard micro F1 | 0.7650 | `outputs/logs/ecbench_b3_phase1_resume.log` |

## Resolved Inconsistencies

### Contact-EC-3B

Several 3B numbers exist:

- `0.6316`: full `N=124` multi-label temporal evaluation; this is canonical.
- `0.6480`: older `N=101` partial evaluation; not used for main conclusions.
- `0.5565`: alternate full evaluation file for a different/older 3B evaluation path; not used.
- `0.5246`: rare-EC value in an auxiliary stdout-only argmax evaluation, not the main temporal micro F1.

Paper action:

- Main table and conclusion use `0.6316`.
- Caption now states that earlier `N=101` and auxiliary argmax-only 3B evaluations are excluded.

### B3 Contact-Only Validation

Two B3 hard-validation values exist:

- `0.7786`: interrupted training log reached this value before DataLoader termination.
- `0.7650`: completed resumed run with auditable checkpoint/result pair.

Paper action:

- Main and supplementary tables use `0.7650`.
- Supplementary reproducibility section explains why the completed resumed checkpoint is the paper source.

## Remaining Reviewer-Sensitive Points

1. HIT-EC remains stronger in absolute temporal performance (`0.8471` vs. Contact-EC flat `0.6032` and Contact-EC-3B `0.6316`).
2. Macro F1 is very low on temporal testing because the Level-4 EC label space is long-tailed.
3. Contact-EC-ExpA `0.7417 +/- 0.0182` is a matched `N=99` recency-intersection diagnostic, not a fair full-`N=124` temporal benchmark result.
4. New-392 overlaps the current Swiss-Prot 2026 corpus and should not be called a clean temporal benchmark unless cutoff-controlled.
5. Price-149 lacks AlphaFold contact maps in the current pipeline, so it mainly tests fallback/OOD behavior.
