# collision-risk

Predict the final log10 collision risk of an event from CDMs issued at least two days before TCA.

**Data:** ESA Collision Avoidance Challenge

**Split:** By event: 75% development (4-fold out-of-fold tuning), 12.5% conformal calibration, 12.5% test.

**Baseline:** Latest visible CDM risk carried forward. This model beats it on held-out data.

| Metric | Value |
|---|---|
| baseline.score | 0.6914 |
| baseline.mse_hr | 0.3601 |
| baseline.f2 | 0.5208 |
| baseline.missed_high_risk | 3 |
| baseline.false_alarms | 34 |
| baseline_tuned_line.score | 0.6842 |
| baseline_tuned_line.mse_hr | 0.3601 |
| baseline_tuned_line.f2 | 0.5263 |
| baseline_tuned_line.missed_high_risk | 3 |
| baseline_tuned_line.false_alarms | 33 |
| baseline_tuned_line.shrink | 0.0 |
| baseline_tuned_line.cut | -5.7 |
| lightgbm_full.score | 0.6026 |
| lightgbm_full.mse_hr | 0.3172 |
| lightgbm_full.f2 | 0.5263 |
| lightgbm_full.missed_high_risk | 3 |
| lightgbm_full.false_alarms | 33 |
| lightgbm_full.shrink | 0.1 |
| lightgbm_full.cut | -5.7 |
| lightgbm_serving.score | 0.6266 |
| lightgbm_serving.mse_hr | 0.3298 |
| lightgbm_serving.f2 | 0.5263 |
| lightgbm_serving.missed_high_risk | 3 |
| lightgbm_serving.false_alarms | 33 |
| lightgbm_serving.shrink | 0.1 |
| lightgbm_serving.cut | -5.7 |
| transformer.score | 0.6842 |
| transformer.mse_hr | 0.3601 |
| transformer.f2 | 0.5263 |
| transformer.missed_high_risk | 3 |
| transformer.false_alarms | 33 |
| transformer.shrink | 0.0 |
| transformer.cut | -5.7 |
| deep_ensemble.score | 0.7025 |
| deep_ensemble.mse_hr | 0.3363 |
| deep_ensemble.f2 | 0.4787 |
| deep_ensemble.missed_high_risk | 4 |
| deep_ensemble.false_alarms | 33 |
| deep_ensemble.shrink | 0.1 |
| deep_ensemble.cut | -5.7 |
| deep_ensemble.mean_predictive_std | 0.0128 |
| serving_interval.half_width | 1.7995 |
| serving_interval.calibration_events | 35 |
| serving_interval.test_events | 46 |
| serving_interval.coverage_90 | 1.0 |
| onnx_max_abs_diff | 0.0001 |

## Limitations

- Live events are scored from public TLE data, whose covariance is modelled, not measured.
- High-risk events are rare; the test set holds few of them, so the score moves between seeds.
- The serving model uses only features Ashvini can compute live, and is weaker than the full model.
- Models predict the change from the latest risk; shrink and line are chosen on out-of-fold predictions. shrink 0 with cut -6 is the baseline, so a tie means the model added nothing on validation.
