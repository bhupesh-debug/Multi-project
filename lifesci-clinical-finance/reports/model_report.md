# Model Report — Risk-Adjusted Valuation & Budget Forecasting for Clinical Programs
**LifeSci NYC Internship Program, Finance sample role  •  Run date: 2026-11-08  •  ALL DATA SYNTHETIC**

## 1. Executive summary
- Pipeline (`python src/run_all.py`) regenerates 240 synthetic clinical programs + 1,440 synthetic monthly-burn rows (60 programs × 24 months), trains regression + classification models, runs a 10,000-draw Monte Carlo budget simulation, and writes 5 charts — end to end, seed 42, no network.
- Best rNPV model: Gradient Boosting, held-out R² 0.964, RMSE $61.36M. Best transition model: Logistic Regression, held-out ROC-AUC 0.702.
- **The headline numbers overstate confidence.** 5-fold CV for the rNPV models is unstable (GB mean R² 0.50 ± 0.71; RF 0.28 ± 1.02) and the classifier's CV ROC-AUC is ~0.44–0.50 (no better than chance). With n = 240 and heavy-tailed synthetic peak sales, a single 80/20 split is a lucky draw away from a different story. This report leads with the CV numbers, not just the test numbers.

## 2. Data (synthetic)
- `data/synthetic/clinical_programs.csv`: 240 rows. Phase mix — Preclinical 65, Phase I 52, Phase II 60, Phase III 63. Therapeutic areas: Oncology, Neurology, Immunology, Cardiology, Rare Disease, Infectious Disease. Modalities: Small Molecule, Monoclonal Antibody, Cell Therapy, Gene Therapy, RNA/Oligonucleotide, Vaccine.
- `data/synthetic/monthly_burn.csv`: 1,440 rows. Mean planned spend $1.163M/month vs mean actual (synthetic) $1.224M/month — actuals run ~5% above plan on average in the simulation, with 5% of months carrying a 1.4–1.8× cost spike.
- Transition-success base rate (synthetic label): 38.8%.
- rNPV (synthetic): median $108.94M, mean $256.79M, 92.1% positive — right-skewed by the lognormal peak-sales distribution. This is a property of the invented generator, not a market fact.

## 3. Method
- **rNPV formula (illustrative, disclosed):** `rNPV = pos_to_approval × peak_sales × 0.35 × Σ 1/(1+r)^t (t=1..8) − cost_to_complete / (1+r)^(duration/24)`. Phase approval probabilities are illustrative assumptions (Preclinical 0.08, Phase I 0.12, Phase II 0.22, Phase III 0.55).
- **Features (all one-sentence explainable):** phase ordinal, complexity / unmet need / competitive intensity (1–5), enrollment, sites, duration, monthly burn, log peak sales, log cost-to-complete, discount rate, illustrative POS features, cost per patient, burn × duration, patients per site, unmet-minus-competitive, plus one-hot area/modality (29 features total).
- **Leakage control:** the transition classifier excludes peak sales and cost-to-complete (inputs to the rNPV algebra). The regression keeps them as log drivers and this is flagged as a limitation (§6).
- **Splits:** 80/20 hold-out (seed 42; stratified for classification) + 5-fold CV on the training set. No hyperparameter search — defaults, deliberately, so the comparison is about model class, not tuning.
- **Monte Carlo:** representative synthetic Phase II program, $36M / 30 months, $40M approved budget; lognormal cost (σ = 0.25), normal timeline (±15%); 10,000 draws, seed 42.

## 4. Results

### 4.1 rNPV regression ($M)
| Model | R² test | RMSE test | MAE test | R² CV mean | R² CV std |
|---|---|---|---|---|---|
| Linear Regression (baseline) | 0.469 | 235.07 | 179.84 | 0.266 | 0.285 |
| Ridge (baseline) | 0.477 | 233.31 | 177.55 | 0.287 | 0.278 |
| Random Forest | 0.949 | 72.70 | 35.53 | 0.282 | 1.022 |
| **Gradient Boosting** | **0.964** | **61.36** | **33.03** | 0.499 | 0.708 |

### 4.2 Transition-success classification
| Model | ROC-AUC test | Accuracy | Precision | Recall | F1 | ROC-AUC CV mean |
|---|---|---|---|---|---|---|
| **Logistic Regression (baseline)** | **0.702** | 0.667 | 0.667 | 0.316 | 0.429 | 0.436 |
| Random Forest | 0.545 | 0.604 | 0.500 | 0.158 | 0.240 | 0.502 |

### 4.3 Feature importance (Gradient Boosting, rNPV)
1. `log_peak_sales` 0.616 — dominates; it is an rNPV formula input (see §6).
2. `sites` 0.112; 3. `enrollment_target` 0.093; 4. `pos_to_approval_illustrative` 0.028; 5. `burn_x_duration` 0.027; 6. `duration_months` 0.025; 7. `area_Immunology` 0.021; 8. `log_cost_to_complete` 0.020; 9. `phase_ordinal` 0.020; 10. `cost_per_patient_usd_m` 0.017.
(Chart: `reports/figures/feature_importance.png` — Random Forest importances; same story: peak-sales dominance.)

### 4.4 Monte Carlo budget scenarios
| Metric | Value |
|---|---|
| P50 cost | $35.88M |
| P80 cost | $44.20M |
| P90 cost | $49.57M |
| Mean cost | $37.06M |
| P50 timeline | 30.0 months; P90 36.0 months |
| **Probability of breaching the $40M approved budget** | **33.2%** |
| Probability of >20% cost overrun | 22.8% |

## 5. Business insights (hiring-manager framing)
1. **Budget at P80, not at the point estimate.** The $36M base case has a 1-in-3 chance of breaching a $40M budget; the P80 ($44.20M) is the defensible ask, and the $8.3M P50→P80 gap is the contingency to defend in committee.
2. **Rank on two axes, not one.** rNPV alone rewards big peak sales; pairing predicted rNPV with transition probability surfaces the fund-first quadrant and exposes high-rNPV / low-probability programs that need challenge, not cash.
3. **The classifier's failure is informative.** Operational features (enrollment, sites, duration, burn) barely predict transition success on this synthetic data (CV AUC ~0.5). In a real setting that would point the next data investment at scientific features (biomarker, mechanism, prior data quality) — not at a bigger model.

## 6. Limitations (honest)
- All data synthetic; no result generalises to real programs, companies, or drugs.
- n = 240; CV unstable; single-split test metrics are not robust — do not quote R² 0.96 without the CV caveat.
- rNPV regression partly recovers its own formula (log peak sales = 61.6% importance). The honest predictive test is the classifier, which is weak.
- Monte Carlo volatility assumptions are invented (σ = 0.25, ±15% timeline); real programs need measured variance.
- No time-based validation of burn forecasts; no real-world backtest is possible on synthetic data.
- Phase success probabilities are illustrative, not published industry rates.
- Eligibility: the candidate appears ineligible for the LifeSci program on its own terms (OPT STEM Extension exclusion); this project does not imply eligibility.

## 7. Future improvements
1. Replace illustrative POS assumptions with a documented assumption set + sensitivity tables.
2. Time-based validation: forecast next-6-months burn and backtest against synthetic actuals.
3. SHAP explanations per program for committee review.
4. Portfolio optimiser: maximise expected rNPV subject to capital budget + risk cap.
5. MLflow tracking, versioned data, scheduled re-run, drift checks on burn variance.

## 8. Reproducibility
`python3 src/run_all.py` → `reports/metrics.json` (machine-readable results of this run). Models: `models/best_rnpv_model.pkl` (Gradient Boosting), `models/best_transition_model.pkl` (Logistic Regression pipeline). Deck: `slides/project_deck.pptx` (10 slides, rebuilt by `src/make_deck.py`).
