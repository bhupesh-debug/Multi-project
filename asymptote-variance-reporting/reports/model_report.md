# Model report — forecast & anomaly detection (synthetic data)

## Forecast model comparison (6-month time-based hold-out, Jul–Dec 2025)

| Series | Best model | MAPE by model |
|---|---|---|
| Revenue | Ridge (trend + seasonality) | Seasonal naive: 43.1%; Holt linear trend: 11.0%; Ridge (trend + seasonality): 9.0% |
| Total cost | Ridge (trend + seasonality) | Seasonal naive: 5.4%; Holt linear trend: 2.7%; Ridge (trend + seasonality): 2.3% |

## Anomaly detection (Isolation Forest + material-variance rules)

| month   | rule_flag   | line_material_flag   | model_anomaly   |   anomaly_score |   revenue_variance_pct |   cost_variance_pct | verdict   |
|:--------|:------------|:---------------------|:----------------|----------------:|-----------------------:|--------------------:|:----------|
| 2024-01 | True        | False                | False           |           0.17  |                  0.446 |              -0.002 | Normal    |
| 2024-02 | True        | False                | False           |           0.193 |                  0.226 |              -0.042 | Normal    |
| 2024-03 | True        | False                | False           |           0.216 |                  0.247 |              -0.003 | Normal    |
| 2024-04 | True        | False                | False           |           0.026 |                  0.945 |               0.032 | Normal    |
| 2024-05 | True        | False                | True            |          -0.005 |                 -0.562 |              -0.03  | ANOMALY   |
| 2024-06 | True        | False                | False           |           0.213 |                  0.154 |               0.003 | Normal    |
| 2024-07 | True        | False                | False           |           0.148 |                  0.178 |               0.057 | Normal    |
| 2024-08 | False       | False                | False           |           0.177 |                 -0.037 |              -0.034 | Normal    |
| 2024-09 | True        | False                | False           |           0.075 |                 -0.363 |              -0.001 | Normal    |
| 2024-10 | True        | False                | False           |           0.124 |                 -0.142 |               0.006 | Normal    |
| 2024-11 | True        | False                | False           |           0.001 |                  1.184 |              -0.048 | Normal    |
| 2024-12 | True        | False                | True            |          -0.023 |                 -0.548 |               0.05  | ANOMALY   |
| 2025-01 | True        | False                | False           |           0.112 |                  0.239 |               0.041 | Normal    |
| 2025-02 | True        | False                | False           |           0.064 |                  0.142 |               0.078 | Normal    |
| 2025-03 | True        | False                | True            |          -0.004 |                  0.471 |              -0.032 | ANOMALY   |
| 2025-04 | True        | False                | False           |           0.07  |                  0.148 |              -0.036 | Normal    |
| 2025-05 | True        | False                | False           |           0.201 |                  0.294 |              -0.023 | Normal    |
| 2025-06 | False       | False                | False           |           0.168 |                  0.016 |              -0.005 | Normal    |
| 2025-07 | True        | True                 | False           |           0.169 |                 -0.142 |              -0.014 | ANOMALY   |
| 2025-08 | False       | False                | False           |           0.195 |                  0.092 |              -0.042 | Normal    |
| 2025-09 | False       | False                | False           |           0.176 |                  0.065 |              -0.009 | Normal    |
| 2025-10 | True        | False                | False           |           0.187 |                  0.157 |              -0.046 | Normal    |
| 2025-11 | True        | False                | False           |           0.212 |                  0.177 |               0.014 | Normal    |
| 2025-12 | True        | False                | False           |           0.218 |                  0.219 |               0.002 | Normal    |

## Honest read

- Forecast MAPEs are measured on synthetic data with smooth, seeded patterns — they show the pipeline works, not real forecasting skill.
- The injected shocks (Mar 2025 credit redemption, Jul 2025 technology spike, Nov 2024 macro performance surge) are the ground truth the anomaly layer is checked against.
- 24 monthly observations is a small sample for unsupervised detection; every flag is a prompt for human review, not an automated conclusion.
- KNOWN MISS: the injected Nov 2024 macro performance surge (+118% revenue variance, the largest single month) was NOT flagged ANOMALY — its Isolation Forest score (+0.001) fell just short of the threshold. The detector caught 2 of 3 injected shocks (Mar 2025, Jul 2025). A single-threshold rule would have caught Nov 2024; the joint-pattern model did not. That is a real limitation, not a tuning oversight to hide.
- `rule_flag` means a material variance fired; `verdict` is ANOMALY only when the model agrees or a single cost line is material — so a month can show rule_flag=True and still be Normal.