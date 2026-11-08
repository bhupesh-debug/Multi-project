# Multi-project

Five machine-learning portfolio projects built on November 8, 2026, each from a real
job description. All data is synthetic and labeled as such; nothing here uses
production data or live trading.

| Project | What it is | Built for |
|---|---|---|
| [canon-ai-operations-ml](canon-ai-operations-ml/) | Hallucination detection for AI-generated field-service summaries (champion: Logistic Regression, ROC-AUC 0.8929) | Canon USA — AI Operations & Data Analyst |
| [asymptote-systematic-signal](asymptote-systematic-signal/) | Regime-aware systematic signal engine (logistic regression, backtest Sharpe 0.54, CAGR +3.8%) | Asymptote Capital — Business Analyst Intern |
| [asymptote-variance-reporting](asymptote-variance-reporting/) | Variance decomposition + automated reporting engine (Ridge forecast MAPE 9.0%) | Asymptote Capital — Business Analyst (full-time) |
| [lifesci-clinical-finance](lifesci-clinical-finance/) | Clinical-program rNPV regression + phase-transition classifier with Monte Carlo budgeting | LifeSci NYC Internship Program — Finance roles |
| [asymptote-quant-signal-research](asymptote-quant-signal-research/) | Signal discovery + validation pipeline: Newey-West IC, multiple-testing corrections, deflated Sharpe, purged walk-forward backtest | Asymptote Capital — Quantitative Researcher Intern |

Each project is self-contained: `README.md`, `src/`, `tests/`, `data/`, `reports/`,
and a `requirements.txt`. Run any project from its own directory (see its README).

## Notes

- **Binaries not included in this upload.** Trained model files (`.pkl`), chart
  images (`.png`), spreadsheets (`.xlsx`), and slide decks (`.pptx`) cannot be
  pushed through the API used here — they can be added via the GitHub web UI.
  Every project regenerates its own outputs from `run.py` / `run_all.py`.
- All results are reported with their honest limitations in each project's README
  and model report (for example, the quant-signal project finds no signal that
  survives multiple-testing correction — that is the finding, not a failure to report).
- By Bhupesh Chandra Dimri, 2026.
