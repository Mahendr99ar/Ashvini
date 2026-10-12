# kp-forecast

Forecast the planetary K index 3, 6, 12 and 24 hours ahead from upstream solar wind.

**Data:** NASA OMNI

**Split:** By time: train before 2017-01-01, validate to 2020-01-01, test after.

**Baseline:** Persistence (Kp now) and 27-day recurrence. This model beats it on held-out data.

| Metric | Value |
|---|---|
| h3.persistence.rmse | 0.8642 |
| h3.persistence.storm_tss | 0.5064 |
| h3.recurrence_27d.rmse | 1.6389 |
| h3.recurrence_27d.storm_tss | 0.0271 |
| h3.lightgbm.rmse | 0.7587 |
| h3.lightgbm.storm_tss | 0.4518 |
| h3.lightgbm.storm_cut | 3.1 |
| h3.lightgbm.storm_tss_tuned | 0.8085 |
| h3.tcn.rmse | 0.7807 |
| h3.tcn.storm_tss | 0.2683 |
| h3.tcn.storm_cut | 2.9 |
| h3.tcn.storm_tss_tuned | 0.7874 |
| h6.persistence.rmse | 1.1044 |
| h6.persistence.storm_tss | 0.3595 |
| h6.recurrence_27d.rmse | 1.639 |
| h6.recurrence_27d.storm_tss | 0.0271 |
| h6.lightgbm.rmse | 0.9553 |
| h6.lightgbm.storm_tss | 0.2416 |
| h6.lightgbm.storm_cut | 3.0 |
| h6.lightgbm.storm_tss_tuned | 0.6753 |
| h6.tcn.rmse | 0.9744 |
| h6.tcn.storm_tss | 0.1223 |
| h6.tcn.storm_cut | 2.6 |
| h6.tcn.storm_tss_tuned | 0.6277 |
| h12.persistence.rmse | 1.3228 |
| h12.persistence.storm_tss | 0.2229 |
| h12.recurrence_27d.rmse | 1.6391 |
| h12.recurrence_27d.storm_tss | 0.0271 |
| h12.lightgbm.rmse | 1.1105 |
| h12.lightgbm.storm_tss | 0.0444 |
| h12.lightgbm.storm_cut | 2.8 |
| h12.lightgbm.storm_tss_tuned | 0.461 |
| h12.tcn.rmse | 1.1324 |
| h12.tcn.storm_tss | 0.0417 |
| h12.tcn.storm_cut | 2.5 |
| h12.tcn.storm_tss_tuned | 0.4102 |
| h24.persistence.rmse | 1.5102 |
| h24.persistence.storm_tss | 0.1209 |
| h24.recurrence_27d.rmse | 1.6392 |
| h24.recurrence_27d.storm_tss | 0.0271 |
| h24.lightgbm.rmse | 1.2291 |
| h24.lightgbm.storm_tss | 0.0033 |
| h24.lightgbm.storm_cut | 2.5 |
| h24.lightgbm.storm_tss_tuned | 0.3332 |
| h24.tcn.rmse | 1.2599 |
| h24.tcn.storm_tss | 0.0095 |
| h24.tcn.storm_cut | 2.7 |
| h24.tcn.storm_tss_tuned | 0.1728 |

## Limitations

- OMNI is propagated to the bow shock, so lead time beyond about an hour comes from the model, not the data.
- Storms are rare; storm TSS on a few years of test data has wide error bars.
- Chronos runs zero-shot on Kp alone and does not see the solar wind.
- Regression pulls forecasts toward the mean; storm calls use a cut tuned on validation, reported separately.
