# VerityOps — Hallucination / Risk Classifier for LLM Field-Service Summaries

**ML project for the Canon U.S.A., Inc. — AI Operations & Data Analyst role
(Melville, NY; hybrid; $69,300–$103,770; apply by Jan 16, 2027).**

> **ALL DATA IS SYNTHETIC.** Every work order, customer, technician, device
> serial, part code, cost, and date in this repo was invented by
> `src/generate_data.py` (seed 42). No real Canon data is used anywhere.
> The model would power **VerityOps**, an AI-output validation platform for
> field-service operations (see the companion PRD in `../prd/`).

## Problem

Field-service teams use an LLM to draft visit summaries and customer updates
from structured work orders. The LLM sometimes hallucinates: wrong part
numbers, wrong dates, invented costs, a fabricated next step. A wrong number
in a customer-facing update is a billing dispute; a wrong part code is a
second truck roll. This project builds the validator the JD describes:

1. **A classifier** that scores every LLM-generated summary as
   *safe to auto-send* vs *flag for human review*, by checking the summary
   against the structured source work order.
2. **A monitoring layer** that tracks real-time reliability, bias, and
   accuracy metrics for the deployed models — including a detection-gap
   metric that catches silent recall degradation.

## Results (test set, n = 360, 29.9% hallucinated)

| Model | Threshold | ROC-AUC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression (champion)** | 0.30 | **0.8929** | 0.8367 | 0.7664 | **0.8000** | 0.0774 |
| Gradient Boosting | 0.65 | 0.8992 | 0.9136 | 0.6916 | 0.7872 | 0.1019 |
| Random Forest | 0.65 | 0.8877 | 0.9178 | 0.6262 | 0.7444 | 0.1206 |
| Rules baseline (any grounding check fails) | 0.50 | 0.6464 | 0.3964 | 0.8224 | 0.5350 | 0.4250 |
| TF-IDF text-only (no source record) | 0.05 | 0.6414 | 0.2972 | 1.0000 | 0.4582 | 0.1883 |

Champion test confusion: **TP 82 · FP 16 · TN 237 · FN 25**.
5-fold CV ROC-AUC: RF 0.8607 ± 0.0089, GB 0.8458 ± 0.0114.
Time-based split (train ≤ 2026-03-31): ROC-AUC 0.8241 — lower than the
random split, i.e. expect drift; monitor and retrain.

### Where it is strong, and where it is weak (the honest part)

Recall by corruption type on the test set:

| Corruption type | Recall |
|---|---:|
| Wrong date / device / part code / technician / SLA / serial, inflated hours, fabricated next step (9 types) | **1.000** |
| Invented cost | 0.818 |
| Subtle issue swap (plausible but wrong problem description) | 0.586 |
| Wrong part name (right code, wrong name) | 0.429 |

All 25 missed hallucinations are subtle fabrications (24) or an invented
cost hidden by a field omission (1). **The system is a near-perfect
contradiction detector and a weak subtle-fabrication detector.** Grounding
checks catch factual contradictions; plausible paraphrase is the irreducible
blind spot — which is why human audit stays in the loop. 9 of the 16 false
positives are faithful summaries with a digit-transposition typo: no single
check is decisive, the model weighs evidence.

### Monitoring (90-day simulated production stream, n = 2,700)

Overall hallucination rate 29.5%, flag rate 22.7%, precision 0.845,
recall 0.652. **17 alert days.** During a simulated generator upgrade
(days 45–59: terser, overconfident summaries that omit checkable fields),
recall fell to 0.534 (baseline 0.648) and the detection gap widened to
11.1% (baseline 6.2%) — while precision stayed at 0.864. **Precision alone
would have looked fine; the recall/gap pair is what catches silent
degradation.** One false-alarm lesson: days 20–26 breached the gap threshold
for a week with no underlying change — at ~30 summaries/day the estimates
are noisy, so thresholds need a minimum-sample rule and a 2-day
confirmation before paging. Bias check: precision 0.81–0.89 and recall
0.62–0.70 across technician tenure bands; no band differs by > 0.10, so no
bias-mitigation escalation is triggered on synthetic data.

## How to run (fresh checkout)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

Verified with **Python 3.12.3** (`scikit-learn 1.9.1, pandas 3.1.1,
numpy 2.5.4, matplotlib 3.11.2`). `run_all.py` regenerates the data,
trains and compares all five models, and runs the monitoring simulation.
Outputs: `models/metrics.json`, `models/monitoring_metrics.json`,
`models/best_model.pkl`, `reports/model_report.md`,
`reports/monitoring.md`, `reports/figures/*.png`.

Individual steps:

```bash
python -m src.generate_data   # data/synthetic/field_service_summaries.csv
python -m src.features        # data/processed/features.csv
python -m src.train           # models/, reports/model_report.md, figures
python -m src.monitor         # reports/monitoring.md, monitoring.png
```

## Project structure

```
ml-project/
├── README.md
├── requirements.txt
├── .gitignore
├── run_all.py
├── data/
│   ├── raw/                  # (empty — no real data is ever ingested)
│   ├── processed/features.csv
│   └── synthetic/field_service_summaries.csv
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── src/
│   ├── generate_data.py      # synthetic work orders + LLM summaries (seed 42)
│   ├── features.py           # 29 grounding/text/metadata features
│   ├── train.py              # 5 models, threshold tuning, figures, report
│   └── monitor.py            # 90-day reliability simulation + bias checks
├── models/
│   ├── best_model.pkl        # champion (Logistic Regression, thr 0.30)
│   ├── metrics.json
│   └── monitoring_metrics.json
└── reports/
    ├── model_report.md
    ├── monitoring.md
    ├── build_report.py       # builds the PDF report (../your_files/)
    └── figures/
        ├── class_balance.png
        ├── feature_importance.png
        ├── model_comparison.png
        ├── roc_curves.png
        └── monitoring.png
```

## Data schema (synthetic)

One row per work order: `work_order_id, service_date, device_model,
serial_no, technician, customer, issue_type, part_code, part_name,
labor_hours, parts_cost, total_cost, sla_status, next_step,
source_completeness, device_complexity, llm_summary, is_hallucinated,
corruption_types, n_corruptions, omitted_fields, has_typo,
llm_confidence, llm_model_version`.

29 engineered features in three families: **grounding checks** (does the
summary agree with the source record — part code, date, technician,
serial, device, SLA, cost, hours; each split into "stated" vs "match", so
an omitted field is missing evidence, not evidence of error),
**text-shape signals** (length, hedge/certainty language, entity overlap),
**generator metadata** (self-reported confidence, model version, source
completeness, device complexity). `corruption_types` is analysis-only and
is never a feature.

## Design decisions (why this, not something fancier)

- **Logistic Regression as champion, not the gradient booster.** Best
  test F1 (0.8000 vs 0.7872), best Brier score (0.0774), and — decisive
  for an audit tool — its coefficients are explainable to a service ops
  lead and to an auditor. The booster wins AUC by 0.006 and loses
  everywhere a human has to defend a decision.
- **Grounding features, not text embeddings.** The text-only TF-IDF model
  scores AUC 0.64: without the source record you cannot catch a wrong
  number. The product insight is that validation is a *comparison*
  problem, not a *reading* problem.
- **Threshold tuned for max F1 on validation**, not 0.5 — because a missed
  hallucination (wrong cost to a customer) costs more than a false flag
  (30 seconds of reviewer time), the operating point favours recall.
- **No LangChain/AWS production claims.** The author's real AI experience
  is the AI Engineering project and the Parallel-Run Reconciler
  (data-validation tooling); this project demonstrates the same validation
  thinking in the field-service domain. AWS/LangChain familiarity is
  stated as familiarity, not production experience.

## Limitations

- Synthetic corruptions are template-based; real LLM errors are more
  varied (omitted caveats, unit confusion, multi-sentence drift).
- ~30% of synthetic hallucinations are subtle by design; recall is capped
  by them. Treat all metrics as a lower bound on real-world difficulty.
- Simulated confidence overlaps heavily between classes and is the
  top-ranked Random Forest feature — in production it should be
  down-weighted, because a vendor can game self-reported confidence.
- No real Canon, customer, or technician data was used or is needed to
  reproduce this project.

## Author

Bhupesh Chandra Dimri — Financial Analyst (JPMorgan Chase), M.S.
Accounting Analytics & Data Technology, Pace University. GitHub:
github.com/bhupesh-debug (AI-Engineering, Parallel-Run Reconciler).
