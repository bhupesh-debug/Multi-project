# JD Breakdown — Business Analyst, Asymptote Capital (New York, NY)

Role: Business Analyst, full-time, hybrid NYC. $70–80/hr + signing bonus +
bonus + equity. Posted ~Oct 25, 2026; apply by Nov 18, 2026 at 10:59 AM.
Firm: quantitative asset management firm (10–50 employees) investing across
global markets and asset classes; systematic strategies; engineers and
researchers as one integrated unit.

## Step 1 — Deep JD deconstruction

### Core responsibilities (top 7)

| # | Responsibility | Project component |
|---|---|---|
| 1 | Analyse business performance, financial data, operational metrics | Firm + strategy variance tables, KPI dashboard |
| 2 | Build financial models, dashboards, management reports | Excel workbook (7 sheets), Markdown variance report, 4 charts |
| 3 | Market / industry / competitive analysis | Out of scope for synthetic data — flagged as a gap, not faked |
| 4 | Track KPIs; identify trends, risks, opportunities | Margin/AUM trend chart, anomaly flags, forecast outlook |
| 5 | Revenue, cost, profitability & variance analysis | Revenue decomposition + profit bridge (exact tie-out) |
| 6 | Support budgeting, forecasting, strategic planning | 3-model forecast comparison, 3-month outlook with intervals |
| 7 | Develop automated reporting & analytical workflows | `run.py` — one command reproduces the whole monthly pack |

### Requirement hierarchy

**Must-have:** Bachelor's in Business/Finance/Economics/Statistics/Maths/CS;
strong analytical & quantitative skills; advanced Excel/Google Sheets;
strong communication & presentation; large datasets → actionable insights.

**Good-to-have (preferred):** SQL, Python, Power BI, Tableau; financial
modelling, forecasting, valuation; BI / data analytics; familiarity with
financial markets, investment management, fintech; consulting / finance /
business-analytics experience.

**Hidden expectations:** ownership of the monthly reporting cadence (the JD
says "develop automated reporting workflows" — i.e. stop doing it by hand);
comfort explaining variances to management, not just computing them; working
in a small (10–50 person) firm where one analyst covers the whole P&L, so
breadth beats narrow depth; quantitative-firm numeracy (fees, AUM, performance
fees) without needing to be a quant researcher.

**KPIs / success metrics:** on-time, accurate monthly reporting; variances
explained, not just reported; forecast accuracy improving over time; fewer
manual reporting hours; management acting on flagged risks.

**Seniority:** mid-level experienced hire (full-time, $70–80/hr ≈ $145–166k
annualised) — NOT entry level. The project is scoped accordingly: full P&L
ownership, not a tutorial exercise.

**Risk factors:** no real firm data is available, so everything is synthetic
(stated everywhere); performance-fee lumpiness makes revenue variance noisy;
a 24-month history is thin for anomaly detection (see README limitations).

### Dependency map

```
JD skills → tools → outcomes
Variance analysis → Python/pandas → exact revenue decomposition + profit bridge
Forecasting → Holt / Ridge → 3-month outlook with 95% intervals
Reporting automation → run.py + openpyxl → monthly pack in one command
KPI tracking → Excel dashboard + charts → margin/AUM trends, anomaly flags
Communication → Markdown narrative → executive summary a non-technical
  manager can act on in 60 seconds
```

## Step 2 — ML problem framing (honest-fit gate)

Per the skill's honest-fit gate: this JD has **no ML / deep-learning
requirement**. Forcing a transformer or LLM demo onto a Business Analyst role
would read as tone-deaf. The project is framed honestly as an **analytics +
forecasting project** (statistical modelling, variance decomposition,
unsupervised anomaly detection). That is the design decision, stated in the
README.

| Problem | Type | Input | Output | Target |
|---|---|---|---|---|
| 1. Explain the revenue variance | Deterministic decomposition | Budget vs actual AUM, fee rates, perf fees | AUM / rate / perf-fee effects ($) | Exact tie-out to total variance |
| 2. Forecast next-quarter revenue & cost | Time-series regression | 24 months of revenue/cost | 3-month forecast + 95% interval | Minimise hold-out MAPE |
| 3. Flag unusual variance months | Anomaly detection | Monthly variance features | ANOMALY / WATCH / Normal + reason | Catch injected shocks (ground truth) |

## Step 6 — Resume alignment (candidate: Bhupesh Chandra Dimri)

**Match score: 8.5 / 10** — unusually strong fit; the JD reads like his
current JPMorgan FP&A work in an asset-management setting.

| JD requirement | Resume evidence | Gap? | Fix / project proof |
|---|---|---|---|
| Variance analysis | 9+ yrs forecasting/budgeting/variance analysis | No | This project — exact decomposition |
| Financial modelling, forecasting | JPMorgan Financial Analyst; M.S. Accounting Analytics & Data Technology | No | Forecast module, hold-out MAPE 9.0% revenue |
| SQL, Python, Power BI, Tableau | All listed; Python/SQL evidenced by GitHub (quant-ipo-engine, Upstox terminal) | No | This project (Python/pandas/sklearn); Excel workbook as BI output |
| Advanced Excel | Excel VBA listed | No | Workbook build is programmatic; add one Excel-native model note in interview |
| Large datasets → insights | 2,047-IPO dataset; Parallel-Run Reconciler (0.988 ROC AUC) | No | 96 strategy-months → 1-page executive summary |
| Financial markets / investment mgmt | JPMorgan; Jane Street Market Prediction project | Partial | No direct asset-management employment — the two Asymptote projects (signal engine + this one) are the bridge |
| Consulting / business-analytics experience | Forage sims: BCG Data Science, BCG Data for Decision Makers, Deloitte Data Analytics, Citi MQA, Goldman Controllers | No | Cite sims only where role-relevant (per standing instruction) |
| Communication / presentation | — (not evidenced on resume excerpt) | Soft gap | This project's narrative report + slide deck are the evidence |

**Top 3 missing skills:** (1) direct asset-management / fund-accounting
employment — mitigated by the two Asymptote projects; (2) Power BI / Tableau
*delivery* artefacts (skills listed, no public dashboard) — the Excel
workbook partially covers it; (3) market/competitive analysis output — not
built here because faking it on synthetic data would be dishonest; say so in
interview.

**Bullet upgrade suggestions (STAR + JD keywords):**
- "Own the monthly variance-analysis cycle for a $1.4bn synthetic AUM book:
  decomposed revenue variance into AUM, fee-rate and performance-fee effects
  with an exact tie-out, cutting unexplained variance to zero."
- "Automated the management reporting pack (variance report + 7-sheet Excel
  workbook + forecast outlook) into a single reproducible command,
  replacing manual month-end reporting."
- "Built a 3-model forecasting comparison on a 6-month hold-out (best MAPE
  9.0% revenue, 2.3% cost) and a rules + Isolation Forest anomaly layer that
  caught 2 of 3 injected shocks, with the one miss reported openly."

**Relevant Forage sims for this role:** JPMorgan Quantitative Research,
Citi MQA, BCG Data Science, Goldman Controllers, Deloitte Data Analytics,
BCG Data for Decision Makers. (Per standing instruction, add only
role-relevant sims; the personality profile is not resume content.)
