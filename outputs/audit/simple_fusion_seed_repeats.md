# Simple Fusion Baseline Seed Repeats

Evaluation subset: complete Level-4 Swiss-Prot 2023-01 temporal proteins (N=124).
Training corpus/config: SP-2018 EC-Bench setup (`config_ecbench.yaml`).

## Summary

| Model             | Completed seeds   | Micro F1          | Weighted F1       | Macro F1          | Precision         | Recall            | Rare EC F1        |
|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| Concat flat FC    | 3/3               | 0.5922 +/- 0.0150 | 0.4777 +/- 0.0116 | 0.0063 +/- 0.0003 | 0.8269 +/- 0.0479 | 0.4619 +/- 0.0136 | 0.4729 +/- 0.0113 |
| Sum flat FC       | 3/3               | 0.5505 +/- 0.0230 | 0.4818 +/- 0.0067 | 0.0064 +/- 0.0002 | 0.6638 +/- 0.0433 | 0.4706 +/- 0.0131 | 0.3970 +/- 0.0112 |
| Gated MLP flat FC | 3/3               | 0.5543 +/- 0.0525 | 0.4532 +/- 0.0413 | 0.0057 +/- 0.0005 | 0.7768 +/- 0.0692 | 0.4314 +/- 0.0458 | 0.4132 +/- 0.0368 |

## Per-Seed Results

| model_label       | model                   |   seed | status   | path                                                                                                        |   n_samples |   micro_f1 |   weighted_f1 |   macro_f1 |   precision |   recall |   rare_ec_f1 |   rare_ec_classes |
|:------------------|:------------------------|-------:|:---------|:------------------------------------------------------------------------------------------------------------|------------:|-----------:|--------------:|-----------:|------------:|---------:|-------------:|------------------:|
| Concat flat FC    | fusion_concat_flatfc    |     42 | complete | outputs/results/simple_fusion_seed_repeats/simple_fusion_concat_flatfc_seed42_known124_hier_results.json    |         124 |     0.6058 |        0.4889 |     0.0063 |      0.8295 |   0.4771 |       0.4808 |              3397 |
| Concat flat FC    | fusion_concat_flatfc    |     43 | complete | outputs/results/simple_fusion_seed_repeats/simple_fusion_concat_flatfc_seed43_known124_hier_results.json    |         124 |     0.5948 |        0.4658 |     0.006  |      0.8734 |   0.451  |       0.46   |              3397 |
| Concat flat FC    | fusion_concat_flatfc    |     44 | complete | outputs/results/simple_fusion_seed_repeats/simple_fusion_concat_flatfc_seed44_known124_hier_results.json    |         124 |     0.5761 |        0.4783 |     0.0065 |      0.7778 |   0.4575 |       0.4779 |              3397 |
| Sum flat FC       | fusion_sum_flatfc       |     42 | complete | outputs/results/simple_fusion_seed_repeats/simple_fusion_sum_flatfc_seed42_known124_hier_results.json       |         124 |     0.5603 |        0.4783 |     0.0062 |      0.6923 |   0.4706 |       0.4    |              3397 |
| Sum flat FC       | fusion_sum_flatfc       |     43 | complete | outputs/results/simple_fusion_seed_repeats/simple_fusion_sum_flatfc_seed43_known124_hier_results.json       |         124 |     0.567  |        0.4895 |     0.0063 |      0.6852 |   0.4837 |       0.4065 |              3397 |
| Sum flat FC       | fusion_sum_flatfc       |     44 | complete | outputs/results/simple_fusion_seed_repeats/simple_fusion_sum_flatfc_seed44_known124_hier_results.json       |         124 |     0.5243 |        0.4776 |     0.0066 |      0.614  |   0.4575 |       0.3846 |              3397 |
| Gated MLP flat FC | fusion_gated_mlp_flatfc |     42 | complete | outputs/results/simple_fusion_seed_repeats/simple_fusion_gated_mlp_flatfc_seed42_known124_hier_results.json |         124 |     0.6017 |        0.4812 |     0.0062 |      0.8554 |   0.4641 |       0.451  |              3397 |
| Gated MLP flat FC | fusion_gated_mlp_flatfc |     43 | complete | outputs/results/simple_fusion_seed_repeats/simple_fusion_gated_mlp_flatfc_seed43_known124_hier_results.json |         124 |     0.5633 |        0.4725 |     0.0056 |      0.75   |   0.451  |       0.4112 |              3397 |
| Gated MLP flat FC | fusion_gated_mlp_flatfc |     44 | complete | outputs/results/simple_fusion_seed_repeats/simple_fusion_gated_mlp_flatfc_seed44_known124_hier_results.json |         124 |     0.4979 |        0.4058 |     0.0052 |      0.725  |   0.3791 |       0.3774 |              3397 |
