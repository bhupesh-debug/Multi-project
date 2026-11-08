# Research Memo — Signal Discovery & Validation (Synthetic Data)

**Author:** Bhupesh Chandra Dimri | **Firm context:** Quantitative Researcher Intern, Asymptote Capital (New York, NY) | **Date:** November 2026

> ⚠️ **ALL DATA IS SYNTHETIC** (seed 2026). No real market data, no real Asymptote data. This memo documents a *research process*, not a trading recommendation. Nothing here should be traded on.

## 1. Question
Which of 17 candidate quantitative signals show statistically robust, out-of-sample predictive power for 5-day forward returns — after controlling for multiple testing and data snooping — and which are just noise that looks good in-sample?

## 2. Method (in one paragraph)
Daily synthetic prices for 12 SYN_ assets (2020–2025, 1,512 days). 17 signals in 5 families (momentum, mean reversion, volatility, trend, cross-sectional). Each signal scored daily, cross-sectionally z-scored. Validation: daily Spearman IC vs 5-day forward returns, Newey-West t-statistics (lag 5), Bonferroni / Holm / Benjamini-Hochberg corrections, deflated Sharpe (Bailey & Lopez de Prado), signal-correlation clustering, and 3-fold expanding-window walk-forward backtest with purge (5d) + embargo (5d). Long-short tercile portfolios net of 5 bps costs.

## 3. Findings

| Signal | Mean IC | NW t | p (raw) | p (BH) | Passes FDR 5%? |
|---|---|---|---|---|---|
| MOM_120d | 0.0446 | 2.72 | 0.006455 | 0.05666 | no |
| MOM_60d | 0.0422 | 2.71 | 0.006666 | 0.05666 | no |
| TREND_MACD | 0.0350 | 2.14 | 0.03236 | 0.11 | no |
| TREND_breakout | 0.0329 | 1.88 | 0.06078 | 0.1722 | no |
| MOM_12m1 | 0.0235 | 1.43 | 0.1518 | 0.3226 | no |
| XSEC_ret1d | 0.0211 | 2.50 | 0.01253 | 0.062 | no |
| VOL_ratio | 0.0205 | 1.53 | 0.1265 | 0.3071 | no |
| TREND_MA20 | 0.0182 | 1.21 | 0.2256 | 0.3568 | no |
| MOM_20d | 0.0175 | 1.15 | 0.2518 | 0.3568 | no |
| XSEC_skew20 | 0.0153 | 1.05 | 0.2942 | 0.3847 | no |
| MOM_5d | 0.0069 | 0.50 | 0.6205 | 0.6205 | no |
| REV_5d | -0.0069 | -0.50 | 0.6205 | 0.6205 | no |
| VOL_20d | -0.0105 | -0.69 | 0.489 | 0.5542 | no |
| VOL_absret | -0.0133 | -0.87 | 0.3843 | 0.4667 | no |
| XSEC_RSI14 | -0.0172 | -1.31 | 0.1915 | 0.3568 | no |
| REV_20d | -0.0175 | -1.15 | 0.2518 | 0.3568 | no |
| REV_MA60 | -0.0392 | -2.44 | 0.01459 | 0.062 | no |

**Survivors after BH-FDR 5%: NONE.**

Top 5 by IC:
- **MOM_120d**: IC 0.0446, NW t 2.72, BH p 0.05666, long-short Sharpe 0.70 (net of costs), deflated Sharpe 0.43.
- **MOM_60d**: IC 0.0422, NW t 2.71, BH p 0.05666, long-short Sharpe 0.94 (net of costs), deflated Sharpe 0.63.
- **TREND_MACD**: IC 0.0350, NW t 2.14, BH p 0.11, long-short Sharpe 0.60 (net of costs), deflated Sharpe 0.37.
- **TREND_breakout**: IC 0.0329, NW t 1.88, BH p 0.1722, long-short Sharpe 0.95 (net of costs), deflated Sharpe 0.60.
- **MOM_12m1**: IC 0.0235, NW t 1.43, BH p 0.3226, long-short Sharpe 0.24 (net of costs), deflated Sharpe 0.10.

## 4. Walk-forward (out-of-sample) — the honest test

| signal         |   fold |   ic_train |   ic_test |   ic_decay |   ls_sharpe_test |
|:---------------|-------:|-----------:|----------:|-----------:|-----------------:|
| MOM_120d       |      1 |     0.0191 |    0.0828 |    -0.0636 |           2.3383 |
| MOM_120d       |      2 |     0.0374 |    0.0844 |    -0.0471 |           1.2081 |
| MOM_120d       |      3 |     0.0516 |    0.0286 |     0.023  |           0.1259 |
| MOM_60d        |      1 |     0.0187 |    0.0303 |    -0.0116 |           1.4317 |
| MOM_60d        |      2 |     0.0199 |    0.1141 |    -0.0942 |           2.294  |
| MOM_60d        |      3 |     0.041  |    0.0513 |    -0.0103 |           0.8987 |
| REV_MA60       |      1 |    -0.0119 |   -0.0272 |     0.0153 |          -1.29   |
| REV_MA60       |      2 |    -0.0145 |   -0.1207 |     0.1062 |          -1.6778 |
| REV_MA60       |      3 |    -0.0375 |   -0.0495 |     0.012  |          -1.0921 |
| TREND_MACD     |      1 |     0.0114 |    0.0109 |     0.0006 |           0.6791 |
| TREND_MACD     |      2 |     0.0094 |    0.1225 |    -0.1131 |           2.4064 |
| TREND_MACD     |      3 |     0.0327 |    0.0481 |    -0.0154 |          -0.1568 |
| TREND_breakout |      1 |    -0.009  |    0.0575 |    -0.0666 |           1.9183 |
| TREND_breakout |      2 |     0.0111 |    0.0689 |    -0.0578 |           1.8493 |
| TREND_breakout |      3 |     0.0284 |    0.054  |    -0.0256 |           0.6725 |

Read the decay column first: a signal whose train IC collapses out-of-sample was fit to noise, however impressive its in-sample p-value.

## 5. What I would NOT do
- Do not trade the full 17-signal library, and do not trade any signal that fails FDR correction — most of the library is noise by construction.
- Do not re-run the search with new signal variants until a survivor fails; each new variant is another trial and inflates the false-discovery rate.
- Do not use the in-sample Sharpe as a sizing input; use the walk-forward test Sharpe, and even that on synthetic data.

## 6. Next steps
1. Re-test survivors on a fresh synthetic seed (out-of-sample *data*, not just out-of-sample *time*) — a true signal should survive a new seed.
2. Add realistic frictions: borrow costs, capacity, market-impact proxy.
3. Combine survivors only if their correlation is low (< 0.5); otherwise the 'combination' is the same bet twice.
4. Pre-register the next signal family *before* looking at results.

## 7. Limitations (stated openly)
- Synthetic data with only two embedded true effects; real markets have regimes, jumps, and microstructure this generator lacks.
- 12 assets is a narrow cross-section; IC estimates are noisy.
- Cost model is a simple turnover × 5 bps proxy, not a fill simulator.
- Walk-forward uses 3 folds only; a single split would be weaker still.
