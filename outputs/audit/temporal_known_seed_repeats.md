# Temporal Known-124 Seed Repeats

Evaluation subset: complete Level-4 Swiss-Prot 2023-01 temporal proteins (N=124).
Training corpus/config: SP-2018 EC-Bench setup (`config_ecbench.yaml`).

## Summary

| Model              | Completed seeds   | Micro F1          | Weighted F1       | Macro F1          | Precision         | Recall            | Rare EC F1        |
|:-------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|:------------------|
| B1 ESM-2           | 3/3               | 0.4508 +/- 0.0203 | 0.3387 +/- 0.0194 | 0.0034 +/- 0.0004 | 0.9035 +/- 0.0445 | 0.3006 +/- 0.0173 | 0.3292 +/- 0.0272 |
| B3 contact         | 3/3               | 0.4244 +/- 0.0207 | 0.3325 +/- 0.0237 | 0.0037 +/- 0.0002 | 0.8300 +/- 0.0220 | 0.2854 +/- 0.0200 | 0.3354 +/- 0.0254 |
| Contact-EC flat FC | 3/3               | 0.6241 +/- 0.0170 | 0.5278 +/- 0.0222 | 0.0068 +/- 0.0001 | 0.7997 +/- 0.0226 | 0.5120 +/- 0.0200 | 0.5257 +/- 0.0313 |

## Per-Seed Results

| model_label        | model            |   seed | status   | path                                                                                                          |   n_samples |   micro_f1 |   macro_f1 |   weighted_f1 |   precision |   recall |   rare_ec_f1 |   rare_ec_classes |
|:-------------------|:-----------------|-------:|:---------|:--------------------------------------------------------------------------------------------------------------|------------:|-----------:|-----------:|--------------:|------------:|---------:|-------------:|------------------:|
| B1 ESM-2           | b1_esm2_fc       |     42 | complete | outputs/results/temporal_known_seed_repeats/temporal_known_b1_esm2_fc_seed42_known124_hier_results.json       |         124 |     0.4706 |     0.0037 |        0.3534 |      0.9412 |   0.3137 |       0.3371 |              3397 |
| B1 ESM-2           | b1_esm2_fc       |     43 | complete | outputs/results/temporal_known_seed_repeats/temporal_known_b1_esm2_fc_seed43_known124_hier_results.json       |         124 |     0.4519 |     0.0035 |        0.3461 |      0.8545 |   0.3072 |       0.3516 |              3397 |
| B1 ESM-2           | b1_esm2_fc       |     44 | complete | outputs/results/temporal_known_seed_repeats/temporal_known_b1_esm2_fc_seed44_known124_hier_results.json       |         124 |     0.43   |     0.0029 |        0.3167 |      0.9149 |   0.281  |       0.2989 |              3397 |
| B3 contact         | b3_contact       |     42 | complete | outputs/results/temporal_known_seed_repeats/temporal_known_b3_contact_seed42_known124_hier_results.json       |         124 |     0.408  |     0.0037 |        0.3123 |      0.8542 |   0.268  |       0.3111 |              3397 |
| B3 contact         | b3_contact       |     43 | complete | outputs/results/temporal_known_seed_repeats/temporal_known_b3_contact_seed43_known124_hier_results.json       |         124 |     0.4476 |     0.0039 |        0.3586 |      0.8246 |   0.3072 |       0.3617 |              3397 |
| B3 contact         | b3_contact       |     44 | complete | outputs/results/temporal_known_seed_repeats/temporal_known_b3_contact_seed44_known124_hier_results.json       |         124 |     0.4175 |     0.0036 |        0.3265 |      0.8113 |   0.281  |       0.3333 |              3397 |
| Contact-EC flat FC | fusion_v2_flatfc |     42 | complete | outputs/results/temporal_known_seed_repeats/temporal_known_fusion_v2_flatfc_seed42_known124_hier_results.json |         124 |     0.6098 |     0.0067 |        0.5023 |      0.8065 |   0.4902 |       0.4909 |              3397 |
| Contact-EC flat FC | fusion_v2_flatfc |     43 | complete | outputs/results/temporal_known_seed_repeats/temporal_known_fusion_v2_flatfc_seed43_known124_hier_results.json |         124 |     0.6429 |     0.0068 |        0.5429 |      0.8182 |   0.5294 |       0.5345 |              3397 |
| Contact-EC flat FC | fusion_v2_flatfc |     44 | complete | outputs/results/temporal_known_seed_repeats/temporal_known_fusion_v2_flatfc_seed44_known124_hier_results.json |         124 |     0.6196 |     0.0069 |        0.5381 |      0.7745 |   0.5163 |       0.5517 |              3397 |
