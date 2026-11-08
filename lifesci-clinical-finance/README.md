# Risk-Adjusted Valuation & Budget Forecasting for Clinical Programs

**LifeSci NYC Internship Program — Finance sample role  •  SYNTHETIC DATA ONLY**

> **Eligibility note (one line, as required):** the candidate appears **ineligible** for the LifeSci NYC Internship Program on its own terms — the program excludes applicants requiring the OPT STEM Extension, and completing a placement into 2027 would require his STEM extension. This project is a portfolio piece built from the program's finance work; it does not imply eligibility.

> **Synthetic-data statement:** ALL data in this project is synthetic, generated in code (`src/generate_data.py`, seed 42). No real clinical-program data, no real pharma/biotech data, no patient data, and no production use. Phase success probabilities are illustrative simulation assumptions, not published industry rates and not real-world claims. Do not use this project for investment, budgeting, or clinical decisions.

## One-liner

A reproducible pipeline that prices synthetic clinical programs by risk (risk-adjusted NPV), forecasts their budgets, and stress-tests both with Monte Carlo simulation — the exact Finance work described in the LifeSci NYC posting: *"Develop new financial projections and budget model for existing and new clinical programs. Develop risk-adjusted valuation models for clinical programs."*

## What it does (3 problems)

| # | Problem | Type | Output |
|---|---|---|---|
| 1 | Predict risk-adjusted NPV (rNPV, $M) for a clinical program | Regression | rNPV estimate + model comparison |
| 2 | Predict next-phase transition success (yes/no) | Classification | Probability + ROC-AUC / precision / recall |
| 3 | Budget & timeline scenarios for a representative program | Monte Carlo (10,000 draws) | P50 / P80 / P90 cost, probability of breaching the approved budget |

**Honest-fit design decision (jd-to-ml-project, Step 2):** this is a forecasting / valuation *analytics* project, not deep learning. The role's technical bar is statistical modelling + scenario engines; a transformer/LLM demo forced onto a Finance JD would be tone-deaf. Baselines (Ridge / Logistic Regression) vs intermediates (Random Forest / Gradient Boosting).

## Results (measured, 2026-11-08 run — see `reports/model_report.md`)

| Model | Held-out test | 5-fold CV |
|---|---|---|
| **rNPV — Gradient Boosting (best)** | R² **0.964**, RMSE $61.36M, MAE $33.03M | R² mean 0.50, std 0.71 — **unstable** |
| rNPV — Random Forest | R² 0.949, RMSE $72.70M | R² mean 0.28, std 1.02 — **unstable** |
| rNPV — Ridge (baseline) | R² 0.477, RMSE $233.31M | R² mean 0.29 |
| **Transition — Logistic Regression (best)** | ROC-AUC **0.702**, accuracy 0.667, recall 0.316 | ROC-AUC mean 0.44 — **weak** |
| Transition — Random Forest | ROC-AUC 0.545 | ROC-AUC mean 0.50 |

- rNPV (synthetic): median $108.94M, mean $256.79M, 92.1% of programs positive (skewed by the illustrative peak-sales distribution).
- Monte Carlo (representative synthetic Phase II program: $36M / 30 months, $40M approved budget): P50 $35.88M, P80 $44.20M, P90 $49.57M; **probability of breaching budget 33.2%**; P90 timeline 36.0 months.
- **Read the results honestly:** the held-out rNPV R² looks strong but CV is unstable on only 240 synthetic programs, and the top feature (`log_peak_sales`, 61.6% of Gradient Boosting importance) is an input to the rNPV formula itself — the regression partly recovers its own algebra. The classifier, which *excludes* those inputs to avoid leakage, is weak. Both findings are reported as findings, not hidden. Full discussion in `reports/model_report.md`.

## How to run

```bash
python3 -m pip install -r requirements.txt   # pandas, numpy, scikit-learn, matplotlib, joblib, python-pptx
python3 src/run_all.py        # regenerates data, trains models, runs Monte Carlo, writes charts + reports/metrics.json
python3 src/make_deck.py      # rebuilds slides/project_deck.pptx from reports/metrics.json (run after run_all.py)
```

Python 3.12. No network, no API keys, no credentials. Deterministic (seed 42).

## Project structure

```
ml-project-lifesci-clinical-finance/
├── README.md
├── requirements.txt
├── .gitignore
├── data/synthetic/          # clinical_programs.csv (240 programs), monthly_burn.csv (60 x 24 mo) — SYNTHETIC
├── data/raw/  data/processed/   # empty placeholders (no real data is ever placed here)
├── src/
│   ├── generate_data.py     # synthetic data generator (seed 42)
│   ├── data_pipeline.py     # load / validate / clean (asserts is_synthetic)
│   ├── feature_engineering.py  # interpretable features; leakage-controlled classifier set
│   ├── model.py             # Ridge / RF / GB regression; Logistic / RF classification
│   ├── monte_carlo.py       # 10,000-draw budget & timeline scenarios
│   ├── evaluate.py          # charts (all labelled SYNTHETIC)
│   ├── run_all.py           # end-to-end pipeline
│   └── make_deck.py         # builds the 10-slide deck
├── notebooks/               # 01_eda, 02_feature_engineering, 03_modeling, 04_evaluation
├── models/                  # best_rnpv_model.pkl, best_transition_model.pkl
├── reports/
│   ├── model_report.md      # full results, insights, limitations
│   ├── metrics.json         # machine-readable results of the latest run
│   ├── dashboard.html       # static dashboard (open in a browser)
│   └── figures/             # 5 PNG charts
└── slides/project_deck.pptx # 10-slide deck (skill Step 8)
```

## Data schema (synthetic)

`clinical_programs.csv` — `program_id, therapeutic_area, modality, phase, indication_complexity_1_5, unmet_need_1_5, competitive_intensity_1_5, enrollment_target, sites, duration_months, monthly_burn_usd_m, peak_sales_usd_m, cost_to_complete_usd_m, discount_rate, pos_next_phase_illustrative, pos_to_approval_illustrative, rNPV_usd_m, transition_success, is_synthetic`

`monthly_burn.csv` — `program_id, phase, therapeutic_area, month (1–24), planned_spend_usd_m, actual_spend_usd_m, is_synthetic`

**rNPV formula (fully disclosed, illustrative):** `rNPV = pos_to_approval × peak_sales × 0.35 × Σ 1/(1+r)^t (t=1..8) − cost_to_complete / (1+r)^(duration/24)`. The 0.35 margin proxy, 8-year horizon, and phase probabilities are invented simulation assumptions.

## JD breakdown (LifeSci NYC — Finance)

- **Core work:** financial projections & budget models for existing and new clinical programs; risk-adjusted valuation models for clinical programs.
- **Must-have:** forecasting / budgeting / variance analysis; financial modelling; translating scientific risk into dollars.
- **Good-to-have:** Python, SQL, Power BI / Tableau, Excel VBA.
- **Hidden expectation:** present a valuation *and* its uncertainty to a non-technical committee — hence P50/P80/P90 and breach probabilities, not a single point estimate.
- **Resume alignment (candidate context):** Bhupesh Chandra Dimri — experienced financial analyst, 9+ years forecasting/budgeting/variance analysis; JPMorgan Chase (Jul 2025–present); M.S. Accounting Analytics & Data Technology (Pace, Dec 2025); MBA Finance & Accounting; SQL, Python, Power BI, Tableau, Excel VBA. Match ~7/10 for the finance work itself; gaps: no evidenced life-sciences / clinical-trial domain experience, no clinical valuation precedent — this project is the bridge evidence, built on synthetic data and labelled as such.

## Differentiation (skill Step 10)

1. **Budget at P80, not at the point estimate** — the Monte Carlo breach probability (33.2%) is the number a committee can act on.
2. **Leakage control stated openly** — the classifier excludes rNPV algebra inputs; its weakness is reported, not hidden.
3. **Reproducibility as a feature** — one command regenerates data, models, charts, and report; seed 42.

**Interview questions this project prepares you for:** (1) How would you validate a valuation model with no real outcome data? (2) When is a P80 budget the right ask vs a P50? (3) What breaks first in this rNPV formula if peak-sales assumptions are wrong? (4) Why did the transition classifier fail, and what data would fix it? (5) How would you present a 33% budget-breach probability to a program lead without killing the program?

## Limitations (honest)

- All data synthetic; nothing here generalises to real programs, companies, or drugs.
- n = 240 programs; CV unstable (see results table); classification signal weak.
- rNPV regression partly recovers its own formula (peak-sales dominance).
- Monte Carlo volatility (lognormal σ = 0.25 cost, ±15% timeline) is an invented assumption.
- No time-based validation of the burn forecast; no real-world backtest is possible on synthetic data.
- Apply by Nov 20, 2026 (program deadline, per the posting) — this project does not change the eligibility note above.

## Author

Bhupesh Chandra Dimri — bhupesh7cd@gmail.com — GitHub: github.com/bhupesh-debug
Built with the jd-to-ml-project skill (10-step execution package). Local deliverable only; nothing published or pushed to GitHub as part of this build.
