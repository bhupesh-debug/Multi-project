# VerityOps — Reliability Monitoring (90-day simulation)

**All data is synthetic.** Fresh stream (seed 7, n=2,700), scored by the champion model (Logistic Regression, threshold 0.3).

- Overall hallucination rate: **29.5%**; flag rate: **22.7%**
- Overall precision: **0.845**; recall: **0.652**
- Alert days (hallucination rate > 35% or precision < 80% or detection gap > 10pp): **17** — days [10, 11, 12, 20, 21, 22, 23, 24, 25, 26, 32, 34, 50, 51, 52, 54, 68]
- Drift episode (days 45–59, generator 'upgrade' emitting terser, overconfident summaries that omit checkable fields; measured on audited labels, which land ~7 days later): recall fell to 0.534 (baseline 0.648), precision 0.864 (baseline 0.832), and the detection gap widened to 11.1% (baseline 6.2%). Precision alone would have looked fine — the recall/gap pair is what catches it.
- Alert anatomy: 4 days breached the hallucination-rate threshold, 6 breached precision, 7 breached the gap. Days 20–26 breached the detection gap for a full week with NO underlying change — a false alarm from small daily volumes (~30 summaries/day makes 14-day gap estimates noisy). Lesson: pair the threshold with a minimum-sample rule and a 2-consecutive-day confirmation before paging.

## Bias check — flag rate / precision by technician tenure band (synthetic)

                  n  flag_rate  precision  recall
tenure_band                                      
1-3 yrs      1209.0      0.219      0.842   0.616
3+ yrs        786.0      0.246      0.813   0.680
<1 yr         705.0      0.221      0.891   0.685

## Bias check — by device complexity (1 = simple, 5 = complex)

                       n  flag_rate  precision
device_complexity                             
1                  502.0      0.241      0.835
2                  522.0      0.228      0.840
3                  529.0      0.214      0.796
4                  561.0      0.228      0.875
5                  586.0      0.227      0.872

## What an analyst would do next

1. Page the on-call reviewer on alert days; sample-audit flagged summaries the same day.
2. After a generator upgrade, re-validate on a fresh labelled batch before trusting the old threshold.
3. If a tenure band or complexity tier shows a precision gap > 0.10, escalate as a bias-mitigation review, per the JD's bias-mitigation tracking requirement.
