# Slide Deck — ML Project for Business Analyst Intern at Asymptote Capital

## Slide 1 — Title

**Systematic Signal Engine: Regime-Aware Forecasting for a Quant Asset Manager**
Business Analyst Intern project — Asymptote Capital, New York, NY
Bhupesh Chandra Dimri — November 2026
✅ All data synthetic (seed 42); demonstration of method, not a live strategy.

## Slide 2 — Problem Overview

**One-liner:** Can a disciplined, testable signal tell a systematic manager
when to be long, flat, or short — and can the answer be reported as KPIs a
manager can actually use?

- Asymptote runs systematic strategies across global markets; every business
  decision needs quantitative evidence, not opinion.
- The intern's job is exactly this loop: collect → clean → model → dashboard
  → present. This project is that loop, end to end, on reproducible data.
- ✅ Why it matters: a signal that cannot survive a backtest with costs, or
  cannot be explained to management, is not a signal — it is a story.

## Slide 3 — JD Breakdown

| Must-have | Evidence in this project |
|---|---|
| Excel/Sheets models, dashboards, reports | Native-Excel KPI dashboard with charts |
| Collect/clean/analyse large datasets | 1,827 days × 4 assets → 6,224 model rows |
| Market & competitor research | Regime analysis, cross-asset breadth |
| KPIs, revenue, costs, ops metrics | Synthetic firm KPIs: AUM, margin, headcount |
| Forecasting & scenario analysis | 20-day models + bull/base/bear + Monte Carlo |
| Python / SQL / Tableau / Power BI (preferred) | Python, pandas, scikit-learn; Excel BI layer |

Hidden expectations uncovered: autonomy (no hand-holding in the JD),
communication (reports "for management"), and comfort with ambiguity —
the JD never names an asset class, so the project covers four.

## Slide 4 — ML Approach

- **Problem type:** binary classification (will this asset be higher in 20
  trading days?) + 3-class regime classification (bull/range/bear).
- **Input:** 12 trailing-only features — momentum, volatility, trend, RSI,
  drawdown, distance from high, market breadth.
- **Output:** probability of a positive 20-day return → position (+1 / 0 / −1).
- **Algorithms:** logistic regression (interpretable), random forest,
  histogram gradient boosting — vs a no-learning momentum rule.
- **Design decision (honest-fit gate):** this is an analytics + forecasting
  project, not deep learning. The JD asks for Excel, research and scenario
  analysis — a transformer would be tone-deaf; a reproducible forecasting
  pipeline is the right ambition.

## Slide 5 — Data Strategy

- **Source:** synthetic, generated in code (seed 42). Regime-switching
  prices for 4 proxies (SYN_EQ / SYN_BD / SYN_CMD / SYN_FX) + synthetic
  monthly firm KPIs (AUM, revenue, costs, headcount).
- **Why synthetic:** no real Asymptote data exists publicly; inventing
  "real" data would be fabrication. Synthetic data is labelled everywhere.
- **Preprocessing:** trailing-only features (no look-ahead), chronological
  train/test split (2019–2023 / 2024–2025), rows with incomplete windows
  dropped.
- **Schema:** one row per (date, asset); 12 features; forward 20-day targets.

## Slide 6 — Model Design

- **Baseline:** momentum rule — long if 60-day return > 0 and price above
  its 60-day average, else short. F1 0.456.
- **Best model:** logistic regression, F1 0.596, ROC-AUC 0.560.
- **Feature engineering highlights:** volatility ratio (current vs 60-day
  vol), distance from 252-day high, market breadth (share of assets above
  their 60-day average).
- **Evaluation:** accuracy, ROC-AUC, precision/recall, F1 on a strict
  time-based split; backtest with 5 bps costs; overfitting controlled by
  the time split, simple models, and a rule baseline that must be beaten.
- **Top drivers:** 20-day vol, vol ratio, distance from high, 60-day return.

## Slide 7 — Results & Insights

- **Backtest (test period):** ML signal Sharpe 0.54, CAGR +3.8%, max
  drawdown −9.3% — vs momentum rule 0.24 / +1.8% / −9.6% and buy-and-hold
  −0.83 / −7.9% / −22.4%.
- **Insight 1:** the edge is mostly drawdown avoidance in the test-period
  bear regime, not precise forecasting (AUC ≈ 0.56).
- **Insight 2:** volatility features outrank momentum — regime shows up in
  vol before it shows up in price.
- **Insight 3:** the class balance shifted (53% up-days in train, 47% in
  test) — a live model would need monitoring and retraining, not a single fit.

## Slide 8 — Business Impact

- **Risk framing:** Monte Carlo (2,000 paths) — median $1 → $1.03 in a year;
  29% chance of ending below $1; 1.4% chance of a >10% drawdown.
- **Scenario framing:** bear −10.5% / base +4.1% / bull +21.1% expected
  1-year return — the same table a manager wants before allocating.
- **Efficiency framing:** one command (`python run.py`) regenerates data,
  models, backtest, charts, dashboard and report — the whole analyst loop
  is reproducible and auditable.
- **Stakeholder metric:** every number in the dashboard traces to code and
  a test; nothing is hand-typed.

## Slide 9 — Future Improvements

1. Replace synthetic prices with real, survivorship-bias-free data
   (point-in-time, corporate-action-adjusted).
2. Purged walk-forward validation instead of a single chronological split.
3. Realistic costs: slippage, borrow costs, capacity and turnover limits.
4. Add fundamentals/macro features (rates, credit spreads, positioning).
5. Production checklist: model monitoring, drift detection, retraining
   cadence, independent backtest review, kill-switch rules.

## Slide 10 — About This Project

- Created by **Bhupesh Chandra Dimri** — Financial Analyst, JPMorgan Chase
  (Jul 2025–present); M.S. Accounting Analytics & Data Technology, Pace
  University (Dec 2025); MBA Finance & Accounting.
- GitHub: https://github.com/bhupesh-debug
- Contact: bhupesh7cd@gmail.com | 917-818-6198
- Related work: quant-ipo-engine (2,047 IPOs), Jane Street Market Prediction,
  Parallel-Run Reconciler, Upstox trading terminal.
- Run it: `python -m venv .venv && .venv/bin/pip install -r requirements.txt
  && .venv/bin/python run.py`
