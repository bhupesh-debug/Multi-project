# JD → ML Project Brief — Business Analyst Intern, Asymptote Capital

Complete 10-step execution package per the `jd-to-ml-project` skill.
Role: Business Analyst Intern, Asymptote Capital, New York, NY (10–50
employees). Full-time internship, hybrid NYC, $80–90/hr. Posted ~Oct 25,
2026; **apply by Nov 18, 2026 at 10:59 AM**.
Candidate: Bhupesh Chandra Dimri (see Step 6).

---

## Executive Summary

- **Built:** a regime-aware systematic signal engine — 20-day return
  prediction → daily long/flat/short positions → backtest vs momentum rule
  and buy-and-hold → Excel KPI dashboard → bull/base/bear + Monte Carlo
  scenario engine. One command (`python run.py`) regenerates everything.
- **Result (test 2024–2025):** best model (logistic regression) F1 0.596,
  ROC-AUC 0.560; backtest Sharpe 0.54 vs 0.24 (rule) vs −0.83 (buy-hold).
- **Honest headline:** the signal is weak (AUC ≈ 0.56) and the backtest edge
  is mostly bear-drawdown avoidance. On synthetic data that is the correct
  finding — and stating it is the point.
- **All data synthetic** (seed 42, `SYN_`-prefixed tickers), labelled
  everywhere. Nothing here is real market data or Asymptote data.
- **Resume match: 8/10** on skills — with one eligibility flag: the JD asks
  for candidates *currently pursuing* a degree; he finished his M.S. in
  Dec 2025. Confirm with the recruiter before applying.

---

## Step 1 — JD Deconstruction

### Core responsibilities (day-to-day)

1. Quantitative + qualitative analysis for business/strategic decisions
2. Collect, clean, organise, analyse data
3. Build/maintain Excel/Google Sheets models, dashboards, analytical reports
4. Market, industry, competitor research
5. Analyse business performance — KPIs, revenue, costs, operational metrics
6. Identify trends, opportunities, risks
7. Prepare presentations and concise reports for management
8. Support financial/business modelling, forecasting, scenario analysis

### Required vs preferred

| Must-have | Good-to-have / preferred |
|---|---|
| Pursuing Bachelor's/Master's in quantitative or business field | Python, SQL, R, Tableau, Power BI |
| Strong analytical & problem-solving skills | Financial modelling / business analytics experience |
| Strong Excel / Google Sheets | Familiarity with financial markets, investment research, tech companies |
| Attention to detail with large datasets | Market research / consulting / finance / analytics experience |
| Strong communication skills | Presentation & data-visualisation skills |
| Independent research; fast-paced entrepreneurial environment | |

### Hidden expectations

- **Autonomy:** "independent research" + a 10–50-person firm = no
  hand-holding; the intern is expected to scope work themselves.
- **Ambiguity tolerance:** the JD names no asset class, no strategy, no
  dataset — the intern must choose sensible proxies and defend them.
- **Communication as a deliverable:** "reports for management" means the
  output is a decision document, not a notebook.
- **Scale:** "large datasets" at a systematic firm means time-series
  discipline (no look-ahead, chronological splits) — the classic intern
  failure mode is shuffling time-series data.

### KPIs / success metrics for the role

Quality of analysis, dashboard/report usefulness to management, forecast
accuracy vs naive baselines, speed of ad-hoc research turnaround.

### Seniority

Intern (student) level on paper — but the work described is
analyst-grade. The candidate is an experienced analyst (9+ years), which
is a strength for the work and a flag for eligibility (see Step 6).

### Dependency map

Skills → tools → outcomes: Excel/Sheets → models & dashboards →
management decisions; Python/SQL → clean data → trustworthy forecasts;
market research → regime context → scenario analysis.

### Risk factors

- Eligibility: "currently pursuing" wording vs completed M.S. (Dec 2025).
- Overqualification: 9+ years' experience for an intern title — frame as
  immediate productivity, not as a stepping stone.
- No real firm data available — synthetic data must be labelled honestly.

---

## Step 2 — ML Problem Framing

**Honest-fit gate (explicit design decision):** this JD has no ML or
deep-learning requirement. Forcing a transformer onto it would be
tone-deaf. The artifact is an **analytics + forecasting project**
(statistical modelling, regime classification, scenario engine) matched to
the role's actual technical bar — Excel, research, forecasting, scenarios.

| Problem | Input | Output / target | Type |
|---|---|---|---|
| P1 — 20-day direction signal | 12 trailing price features per asset | P(asset higher in 20 days); position +1/0/−1 | Binary classification |
| P2 — Market regime | Same features | bull / range / bear next 20 days | 3-class classification |
| P3 — Business KPI outlook | Synthetic monthly firm KPIs | AUM / revenue / margin trajectory + scenarios | Forecasting + scenario analysis |

Constraints: interpretability (a manager must understand it), reproducibility
(one command), no look-ahead (trailing features only, chronological split),
honest labelling of synthetic data. Real-world dataset sources for a live
version: CRSP/Compustat or Yahoo Finance (prices), FRED (macro), Kaggle
financial time-series — all replaced here by a seeded synthetic generator
so nothing is misrepresented as real.

---

## Step 3 — Data Strategy + Simulation

**Schema (model frame, 6,224 rows):** one row per (date, asset).
Features: `ret_5d, ret_20d, ret_60d, vol_20d, vol_ratio, ma_ratio, rsi_14,
drawdown_60d, trend_slope, dist_252d_high, breadth, eq_minus_bd`.
Targets: `target_up` (binary), `target_regime` (bull/range/bear),
`target_ret` (raw forward 20-day return).

**Collection plan (live version):** survivorship-bias-free prices
(CRSP / vendor), point-in-time fundamentals, FRED macro series; stored in
`data/raw/` → cleaned to `data/processed/`.

**Preprocessing:** trailing-only windows; RSI (Wilder); rolling vol
annualised; rows with incomplete trailing (252d) or forward (20d) windows
dropped; chronological split 2019–2023 train / 2024–2025 test.

**Synthetic data:** 1,827 business days of regime-switching prices for
SYN_EQ / SYN_BD / SYN_CMD / SYN_FX (Markov bull/range/bear regimes with
regime-dependent drift/vol + volatility clustering) and 84 months of
synthetic firm KPIs (AUM $4.17bn → $7.27bn, margin 11.7–51.4%). Full
dictionary: `data/synthetic/DATA_DICTIONARY.md`.

**Simulated scenarios:** bull/base/bear drift-vol shifts; 2,000-path
block-bootstrap Monte Carlo; single-day stress shocks (equity −10%,
rates +100 bp, vol spike, FX gap).

---

## Step 4 — Model Strategy

| Tier | Models | Role here |
|---|---|---|
| Baseline | Momentum rule (no learning) | Must-be-beaten sanity check — F1 0.456 |
| Intermediate | Logistic regression, random forest, hist. gradient boosting | Primary models; logreg wins on F1 |
| Advanced | LSTM / transformer | **Not used** — unjustified for this JD and data scale |
| Cutting edge | LLM fine-tuning / RL | **Not used** — would signal misunderstanding of the role |

**Feature engineering (top 5):** volatility ratio (current vs 60-day vol);
distance from 252-day high; market breadth (share of assets above 60-day
MA); equity-minus-bond 20-day spread; trend slope (OLS slope of log-price).

**Evaluation:** accuracy, ROC-AUC, precision/recall, F1 on the time-based
test set; backtest Sharpe/Sortino/max-drawdown/Calmar/win-rate with 5 bps
costs. **Overfitting controls:** chronological split (no shuffling), simple
models preferred, rule baseline as a hurdle, and the honest reporting of
weak AUC rather than tuning until the test set looks good.

**Results:** logreg F1 0.596 / AUC 0.560; RF F1 0.548 / AUC 0.558; HGB F1
0.532 / AUC 0.553; regime classifier accuracy 0.460 (chance ≈ 0.33).

---

## Step 5 — Visual Intelligence

All charts in `reports/charts/` (7 PNGs), each with a business reading:

1. **Feature importance** — vol features dominate; a manager reads this as
   "regime shows up in volatility before price."
2. **Model comparison** — grouped bars; all models cluster at AUC ≈ 0.55,
   i.e. the signal is weak and model choice barely matters.
3. **Equity curves** — ML signal vs rule vs buy-hold; the visual story is
   drawdown avoidance in the bear test period.
4. **Drawdown chart** — buy-hold gives back 22%; the signal caps it at 9%.
5. **Correlation heatmap** — momentum features overlap heavily; vol features
   add independent information.
6. **Prices + regimes** — the synthetic equity proxy with bull/bear bands.
7. **Monte Carlo distribution** — terminal wealth of $1: p5 $0.94, median
   $1.03, p95 $1.14 — the honest range of outcomes.

---

## Step 6 — Resume Alignment

Candidate: Bhupesh Chandra Dimri — experienced financial analyst (9+ years
forecasting/budgeting/variance analysis), Financial Analyst at JPMorgan
Chase (Jul 2025–present), M.S. Accounting Analytics & Data Technology,
Pace (Dec 2025), MBA Finance & Accounting, NY Series 17-52 license
No. 1991975. Skills: SQL, Python, Power BI, Tableau, Excel VBA, Oracle
Fusion, SAP, NetSuite, Bloomberg. GitHub: quant-ipo-engine, Jane Street
Market Prediction, Parallel-Run Reconciler, Upstox trading terminal.
Forage sims: JPMorgan Quantitative Research, Citi Markets Quantitative
Analysis (most on-point here).

| JD requirement | Resume evidence | Gap? | Fix |
|---|---|---|---|
| Excel / Sheets models | JPMorgan FP&A forecasting, budgeting, variance analysis | No | — |
| Large-dataset attention to detail | Parallel-Run Reconciler (mapping-error detector, 0.988 ROC AUC) | No | — |
| Python (preferred) | quant-ipo-engine, Jane Street prediction, this project | No | — |
| SQL (preferred) | SQL at JPMorgan; Oracle Fusion / SAP / NetSuite | No | — |
| Power BI / Tableau (preferred) | Both listed; this project's Excel dashboard | No | — |
| Financial markets familiarity | JPMorgan + quant GitHub work + Forage JPM Quant Research & Citi MQA sims | No | — |
| Market research / consulting experience | Partial — Forage BCG sims only | Partial | This project is the evidence; cite regime + scenario work |
| Currently pursuing a degree | M.S. completed Dec 2025 | **Yes — eligibility flag** | Confirm with recruiter before applying; do not claim current-student status |
| Intern-level seniority | 9+ years' experience | Positioning | Frame as immediate productivity + systematic-strategies fluency |

**Match score: 8/10** on skills and tools; the two deductions are the
eligibility wording and the thin consulting/market-research record.

**Top 3 missing/partial items:** (1) confirmed current-student eligibility;
(2) direct systematic-strategies / quant-firm experience (this project is
the closest evidence); (3) public market-research writing sample.

**Bullet upgrades (STAR + JD keywords):**

- "Built forecasting and variance-analysis models at JPMorgan Chase used
  for budgeting decisions, cutting forecast cycle time by automating
  Excel/VBA reconciliations" → add the quantified outcome from his record.
- "Designed a mapping-error detection model (Parallel-Run Reconciler,
  0.988 ROC AUC) that flags reconciliation breaks between parallel
  reporting runs" → evidence for large-dataset attention to detail.
- "Built a regime-aware systematic signal engine (Python, scikit-learn):
  20-day return forecasting, walk-forward backtest with transaction costs,
  KPI dashboard and Monte Carlo scenarios" → this project, tailored to the
  JD's exact vocabulary.

**New project description for the resume:** "Systematic Signal Engine —
Regime-aware forecasting for a quantitative asset manager. Predicted 20-day
asset direction across four asset proxies (logistic regression, random
forest, gradient boosting) on a chronological split; backtested long/flat/
short signals vs momentum and buy-and-hold benchmarks with transaction
costs; delivered an Excel KPI dashboard (AUM, revenue, margin, headcount)
and bull/base/bear + Monte Carlo scenario analysis. All data synthetic and
labelled; results reported with limitations."

---

## Step 7 — Architecture

```
[Synthetic generator] → [Ingestion / CSV] → [Feature store (model_frame.csv)]
      → [Training pipeline] → [Model registry (models/best_model.pkl)]
      → [Backtest + scenario engine] → [Excel KPI dashboard + report]
      → [Monitoring: retrain on new data, track AUC drift]
```

**Tech stack:** Python 3.12, pandas, NumPy, scikit-learn, matplotlib,
openpyxl, SciPy. No experiment tracker needed at this scale — the run is
deterministic (seed 42) and the report is generated from the run itself.
**Deployment idea:** a scheduled batch job (weekly) that regenerates the
dashboard and emails the one-page KPI summary to management — exactly the
intern's deliverable loop. A Streamlit dashboard is the natural next step
if the team wants it interactive.

---

## Step 8 — Slide Deck Content

Full 10-slide content in `slides/project_deck.md` (title → problem →
JD breakdown → approach → data → model design → results → business impact
→ future improvements → about). Story-driven, ✅ instead of ELI5.

---

## Step 9 — GitHub Packaging

Structure as built (see README.md). Ready to push as
`bhupesh-debug/ml-project-asymptote-systematic-signal` — **not pushed**;
nothing has been published. Suggested repo description: "Regime-aware
systematic signal engine + KPI dashboard + scenario analysis (synthetic
data) — built for the Asymptote Capital Business Analyst Intern role."
README includes title, problem, dataset, run instructions, results,
limitations, author.

---

## Step 10 — Differentiation Layer

**1. What differentiates this from 99% of candidates' projects?**

- It is built to the JD's actual vocabulary (dashboards, KPIs, scenarios)
  instead of a generic Kaggle classifier with a finance label.
- It reports a *weak* result honestly (AUC 0.56) with the reason why —
  most portfolios only show wins.
- One command reproduces everything, including the report and the Excel
  dashboard — auditability is the wow factor for a quant firm.

**2. Insights that impress a hiring manager:**

- Volatility features beat momentum features — regime arrives in vol first.
- The backtest edge is drawdown avoidance, not forecasting skill — and the
  report says so.
- The class balance shifted between train and test (53% → 47% up-days),
  which is why a live model needs monitoring, not a single fit.

**3. Five questions to ask in the interview:**

1. Which strategies and asset classes would this role actually support —
   and what does the current research-to-decision loop look like?
2. Where does the team's data live, and what is the biggest data-quality
   pain point today?
3. How are scenarios currently built — is there a standard bull/base/bear
   framework, or does each analysis start from scratch?
4. What does a great intern deliverable look like here — a dashboard, a
   memo, a model? Who reads it?
5. How is model/signal performance monitored once something ships — is
   there a retraining or kill-switch process?
