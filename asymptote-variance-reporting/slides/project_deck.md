# Slide Deck Content — ML Project for Business Analyst at Asymptote Capital

## Slide 1 — Title
**Variance, Explained: An Automated Reporting Engine for a Systematic Asset Manager**
ML Project for Business Analyst at Asymptote Capital · Bhupesh Chandra Dimri · November 2026
✅ One line: every dollar of variance, explained — automatically, every month.

## Slide 2 — Problem Overview
- A 10–50 person quant manager lives or dies on knowing *why* the P&L moved.
- Budget-vs-actual reports that just print the difference leave management
  guessing: was it flows, fees, or performance?
- ✅ One-liner: this project turns the monthly close into a one-command,
  fully explained variance pack.

## Slide 3 — JD Breakdown
- Must-haves: variance analysis, budgeting/forecasting, KPI tracking,
  management reporting, automated workflows, advanced Excel.
- Hidden expectation: own the monthly cadence in a small firm — breadth over
  narrow depth; explain variances to management, don't just compute them.
- Requirement hierarchy: analytics + forecasting first; no deep learning
  required — and none forced in.

## Slide 4 — ML Approach
- Honest-fit gate: analytics + forecasting project, not a transformer demo.
- Three problems: exact revenue decomposition (deterministic), 3-month
  revenue/cost forecast (Holt vs Ridge, hold-out MAPE), anomaly flagging
  (rules + Isolation Forest, explainable by design).

## Slide 5 — Data Strategy
- 100% synthetic, seed 42: 24 months × 4 strategies (96 strategy-months),
  firm P&L, 5 opex lines, headcount. All `SYN_`-prefixed.
- Three injected shocks as ground truth: Mar 2025 redemption, Jul 2025 tech
  spike, Nov 2024 performance surge.
- Preprocessing: budget/forecast/actual aligned monthly; rounding handled so
  bridges tie out exactly.

## Slide 6 — Model Design
- Baseline: seasonal naive (revenue MAPE 43.1% — the bar to beat).
- Intermediate: Holt linear trend (11.0%) and Ridge trend + seasonality
  (**9.0% revenue, 2.3% cost** — winner).
- Feature engineering: trend + month dummies; variance features for anomaly
  detection (revenue/cost var %, AUM momentum, margin change).
- Overfitting control: time-based hold-out (no shuffling), 6-month test window.

## Slide 7 — Results & Insights
- Dec 2025: revenue +$825k (+21.9%) vs budget; profit +$819k; margin 25.4%.
- Top insight: performance-fee effect dominates revenue variance — AUM effect
  was only +$2.9k in Dec 2025. The story is fees, not flows.
- Top insight: costs are well controlled (+0.2%); the variance is a revenue
  story, not a cost story.
- Top insight: anomaly layer caught 2 of 3 injected shocks — and missed the
  biggest one (Nov 2024), which is reported, not hidden.

## Slide 8 — Business Impact
- One command (`python run.py --month YYYY-MM`) replaces manual month-end
  reporting: Markdown report + 7-sheet Excel workbook + 4 charts.
- Every variance is explained with a named cause (AUM / rate / performance
  fee), so management meetings start from answers, not questions.
- Forecast outlook with 95% intervals gives planning a range, not a point
  guess — revenue ~$4.19–4.32m/month for Jan–Mar 2026.

## Slide 9 — Future Improvements
- Add real flow data and a high-water-mark performance-fee model.
- Add macro drivers (rates, volatility) to the forecast.
- Recalibrate anomaly detection on a longer history; add a threshold-rule
  backstop so a +118% month can never slip through again.
- Power BI / Tableau version of the KPI dashboard; SQL staging layer.
- Production readiness: scheduling (cron), data validation checks, alerting.

## Slide 10 — About This Project
- Created by **Bhupesh Chandra Dimri** — Financial Analyst, JPMorgan Chase;
  M.S. Accounting Analytics & Data Technology, Pace University.
- GitHub: github.com/bhupesh-debug (suggested repo: `asymptote-variance-reporting`)
- Contact: 917-818-6198 · bhupesh7cd@gmail.com · New York, NY
- All data synthetic; no real Asymptote Capital figures used.
