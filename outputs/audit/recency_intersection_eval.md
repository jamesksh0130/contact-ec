# Recency Intersection Evaluation

All models are evaluated on the same temporal known subset that is Level-4 evaluable under the 2018, 2022, and ExpA/2026 encoders (N=99).
This avoids comparing the ExpA result against the broader 2018-only N=124 denominator.

## Per-run Results

| model                |   seed |   n |   micro_f1 |   weighted_f1 |   precision |   recall |   rare_micro_f1 |
|:---------------------|-------:|----:|-----------:|--------------:|------------:|---------:|----------------:|
| Contact-EC 2018      |     42 |  99 |     0.6386 |        0.5492 |      0.791  |   0.5354 |          0.4375 |
| Contact-EC 2018      |     43 |  99 |     0.6706 |        0.5917 |      0.8028 |   0.5758 |          0.4638 |
| Contact-EC 2018      |     44 |  99 |     0.631  |        0.5666 |      0.7681 |   0.5354 |          0.4545 |
| Contact-EC ExpA 2026 |     42 |  99 |     0.7209 |        0.6578 |      0.8493 |   0.6263 |          0.5    |
| Contact-EC ExpA 2026 |     43 |  99 |     0.75   |        0.6897 |      0.8571 |   0.6667 |          0.625  |
| Contact-EC ExpA 2026 |     44 |  99 |     0.7543 |        0.683  |      0.8684 |   0.6667 |          0.5172 |

## Summary

- Contact-EC 2018 micro F1: 0.6467 +/- 0.0210 (3 seeds).
- Contact-EC ExpA 2026 micro F1: 0.7417 +/- 0.0182 (3 run(s), N=99).
- Fair-subset recent-corpus delta: +0.0950 micro F1.

Interpretation:
- The original single-seed ExpA value of 0.7209 was a 99-sample encoder-evaluable result, not a 124-sample known-set result.
- The three-seed ExpA mean remains positive on the fair subset and is smaller than the naive comparison against the 124-sample 2018 result.
- ExpA should be described as a recent-corpus/vocabulary diagnostic because sample count, class frequencies, label coverage, and homolog coverage also change.
