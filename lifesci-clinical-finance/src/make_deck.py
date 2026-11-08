"""make_deck.py — builds slides/project_deck.pptx (10 slides, per skill Step 8).

Run after run_all.py so the deck carries the real measured metrics.
"""
import json
import pathlib
from pptx import Presentation
from pptx.util import Inches, Pt

ROOT = pathlib.Path(__file__).resolve().parent.parent
metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
reg = metrics["regression_rnpv_usd_m"]
clf = metrics["classification_transition_success"]
mc = metrics["monte_carlo"]
best_reg = metrics["regression_best"]
best_clf = metrics["classification_best"]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide(title, bullets):
    s = prs.slides.add_slide(BLANK)
    tx = s.shapes.add_textbox(Inches(0.6), Inches(0.3), Inches(12.1), Inches(0.8)).text_frame
    tx.word_wrap = True
    tx.paragraphs[0].text = title
    tx.paragraphs[0].font.size = Pt(30)
    tx.paragraphs[0].font.bold = True
    body = s.shapes.add_textbox(Inches(0.6), Inches(1.3), Inches(12.1), Inches(5.7)).text_frame
    body.word_wrap = True
    for i, b in enumerate(bullets):
        p = body.paragraphs[0] if i == 0 else body.add_paragraph()
        p.text = "•  " + b
        p.font.size = Pt(18)
        p.space_after = Pt(10)
    return s


slide("Risk-Adjusted Valuation & Budget Forecasting for Clinical Programs",
      ["LifeSci NYC Internship Program — Finance sample role  •  SYNTHETIC DATA ONLY",
       "Bhupesh Chandra Dimri  •  November 2026",
       "One-liner: a reproducible pipeline that prices clinical programs by risk, "
       "forecasts their budgets, and stress-tests both — on fully synthetic data."])
slide("2 — Problem Overview",
      ["Finance teams in life sciences must answer two questions for every clinical program: "
       "what is it worth after adjusting for the risk it never reaches patients, and what will it cost to get there?",
       "This project answers both with one pipeline: (1) a risk-adjusted NPV (rNPV) model, "
       "(2) a next-phase transition-success classifier, (3) a Monte Carlo budget engine.",
       "Why it matters: capital allocation across a portfolio is the highest-leverage finance decision in biotech."])
slide("3 — JD Breakdown (LifeSci NYC Finance)",
      ["Core work: new financial projections & budget models for existing and new clinical programs; "
       "risk-adjusted valuation models for clinical programs.",
       "Must-have: forecasting / budgeting / variance analysis, Excel & financial modelling, "
       "comfort with clinical-stage uncertainty. Good-to-have: Python, SQL, Power BI / Tableau.",
       "Hidden expectation: translate scientific risk (phase, indication, enrollment) into dollars a committee can act on.",
       "Eligibility context: the LifeSci program excludes applicants requiring the OPT STEM Extension — "
       "this is a portfolio piece built from its finance work, not a claim of eligibility."])
slide("4 — ML Approach (honest-fit)",
      ["Design decision: this is a forecasting / valuation analytics project, NOT deep learning — "
       "the role's technical bar is statistical modelling + scenario engines, and a transformer demo would be tone-deaf.",
       "Problem 1 — Regression: predict rNPV ($M) from program features. "
       "Problem 2 — Classification: predict next-phase transition success (yes/no). "
       "Problem 3 — Simulation: Monte Carlo cost / timeline scenarios (10,000 draws).",
       "Baselines (Ridge / Logistic) vs intermediates (Random Forest / Gradient Boosting); 80/20 hold-out + 5-fold CV."])
slide("5 — Data Strategy (all synthetic)",
      [f"{metrics['data']['n_programs']} synthetic programs × 24-month synthetic monthly burn for 60 programs. "
       "Seed 42 — fully reproducible with one command.",
       "Schema: therapeutic area, modality, phase, enrollment, sites, duration, monthly burn, "
       "peak sales, cost-to-complete, discount rate, illustrative phase success probabilities.",
       "No real clinical, pharma, or patient data anywhere. Phase success rates are illustrative "
       "simulation assumptions, not published industry rates."])
slide("6 — Model Design",
      ["Feature engineering (all one-sentence explainable): log peak sales, log cost-to-complete, "
       "burn × duration, cost per patient, patients per site, unmet-need minus competitive intensity, phase ordinal.",
       "Leakage control: the classifier EXCLUDES peak-sales / cost inputs that feed the rNPV algebra.",
       f"Regression winner: {best_reg} — R² {reg[best_reg]['r2_test']}, RMSE ${reg[best_reg]['rmse_test_usd_m']}M. "
       f"Classifier winner: {best_clf} — ROC-AUC {clf[best_clf]['roc_auc_test']}.",
       "Evaluation: R² / RMSE / MAE for regression; ROC-AUC + precision / recall / F1 for classification."])
slide("7 — Results & Insights",
      [f"rNPV: median ${metrics['rnpv_summary_usd_m']['median']}M, "
       f"{metrics['rnpv_summary_usd_m']['share_positive']:.0%} of synthetic programs positive — "
       "the model separates value-creating from value-destroying programs on held-out data.",
       f"Monte Carlo (representative synthetic Phase II program, $36M / 30 mo, $40M approved budget): "
       f"P50 ${mc['cost_p50_usd_m']}M, P80 ${mc['cost_p80_usd_m']}M, P90 ${mc['cost_p90_usd_m']}M; "
       f"probability of breaching budget {mc['prob_budget_breach']:.1%}.",
       "Business insight: budget at P80, not at the point estimate — the P50-to-P80 gap is the contingency "
       "a Finance lead should defend in committee.",
       "Honest caveat (do not hide): held-out R² looks strong (0.96) but 5-fold CV is unstable "
       "(mean 0.50, std 0.71) on only 240 synthetic programs, and the transition classifier is weak "
       "(best ROC-AUC 0.70 test, 0.44 CV) — these are synthetic-data findings, not production claims.",
       "Full numbers, charts, and limitations: reports/model_report.md"])
slide("8 — Business Impact",
      ["Portfolio lens: rank programs by predicted rNPV and transition probability together — "
       "fund the high-rNPV / high-probability quadrant first, challenge the rest.",
       "Budget lens: a P80 budget with a stated breach probability replaces a single-point forecast "
       "that is wrong by construction.",
       "Efficiency: one command (python src/run_all.py) regenerates data, models, charts, and report — "
       "no manual spreadsheet rebuilds."])
slide("9 — Future Improvements",
      ["Replace illustrative phase-success assumptions with a documented, sourced assumption set "
       "and sensitivity tables (never present synthetic rates as industry facts).",
       "Add time-based validation on the monthly-burn series (forecast next-6-months spend, backtest vs actual).",
       "Add SHAP explanations for individual program valuations for committee review.",
       "Add a portfolio optimiser: maximise expected rNPV subject to a capital budget and a risk cap.",
       "Production checklist: versioned data, MLflow tracking, scheduled re-run, drift checks on burn variance."])
slide("10 — About This Project",
      ["Created by Bhupesh Chandra Dimri — experienced financial analyst (9+ years forecasting / "
       "budgeting / variance analysis; JPMorgan Chase; M.S. Accounting Analytics & Data Technology, Pace).",
       "Skills applied: Python, SQL, Power BI, Tableau, Excel VBA — forecasting, valuation, scenario modelling.",
       "All data synthetic. No real clinical-program or pharma data. Not for investment or clinical decisions.",
       "Contact: bhupesh7cd@gmail.com  •  GitHub: github.com/bhupesh-debug"])

out = ROOT / "slides" / "project_deck.pptx"
prs.save(out)
print(f"Saved {out} — {len(prs.slides.__iter__.__self__._sldIdLst)} slides")
