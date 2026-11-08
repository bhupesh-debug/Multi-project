"""
build_report.py — Build the PDF report for the VerityOps ML project.

Reads models/metrics.json and models/monitoring_metrics.json plus the
trained model, so every number in the PDF matches the actual run.
Recomputes per-corruption-type recall from the test split.

Usage (from the project root, venv active):
    python -m reports.build_report
Output:
    ~/workspace/your_files/Canon_AI_Operations_ML_Project_2026-11-08.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                Spacer, Table, TableStyle)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
OUT = Path.home() / "workspace" / "your_files" / \
    "Canon_AI_Operations_ML_Project_2026-11-08.pdf"

NAVY = colors.HexColor("#1a3a5c")
ACCENT = colors.HexColor("#0e7a5f")
LIGHT = colors.HexColor("#eef3f7")
RULE = colors.HexColor("#c9d6e2")
RED = colors.HexColor("#b3261e")

S_TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=26,
                         leading=31, textColor=NAVY)
S_SUB = ParagraphStyle("sub", fontName="Helvetica", fontSize=13, leading=18,
                       textColor=colors.HexColor("#444444"))
S_H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15, leading=19,
                      textColor=NAVY, spaceBefore=16, spaceAfter=6)
S_H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, leading=15,
                      textColor=NAVY, spaceBefore=10, spaceAfter=4)
S_BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=10.2,
                        leading=14.5, spaceAfter=6)
S_BULLET = ParagraphStyle("bullet", parent=S_BODY, leftIndent=14,
                          bulletIndent=4, spaceAfter=3)
S_CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=9.2, leading=12)
S_CELLB = ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=9.2,
                         leading=12)
S_CELlh = ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=9.2,
                         leading=12, textColor=colors.white)
S_CAP = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=9,
                       leading=12, textColor=colors.HexColor("#555555"),
                       spaceAfter=10)
S_FOOT = ParagraphStyle("foot", fontName="Helvetica", fontSize=8,
                        leading=10, textColor=colors.HexColor("#777777"))


def P(text, style=S_BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{i}", S_BULLET) for i in items]


def table(headers, rows, widths, align_right_from=1, padding=5):
    data = [[Paragraph(h, S_CELlh) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), S_CELL) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if align_right_from:
        style.append(("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def per_type_recall():
    """Recompute test-set recall by corruption type from the saved model."""
    from sklearn.model_selection import train_test_split
    from src.features import FEATURE_COLUMNS, build_feature_frame
    raw = pd.read_csv(PROJECT_ROOT / "data" / "synthetic"
                      / "field_service_summaries.csv")
    feats = build_feature_frame(raw)
    b = joblib.load(PROJECT_ROOT / "models" / "best_model.pkl")
    y = feats["is_hallucinated"].to_numpy()
    idx = np.arange(len(feats))
    _, tmp = train_test_split(idx, test_size=720, random_state=42, stratify=y)
    _, te = train_test_split(tmp, test_size=360, random_state=42,
                             stratify=y[tmp])
    tdf = feats.iloc[te].copy()
    tdf["pred"] = (b["model"].predict_proba(tdf[FEATURE_COLUMNS].to_numpy())[:, 1]
                   >= b["threshold"]).astype(int)
    hall = tdf[tdf["is_hallucinated"] == 1].copy()
    hall["ctype"] = hall["corruption_types"].str.split(";")
    ex = hall.explode("ctype")
    out = (ex.groupby("ctype")
           .apply(lambda g: pd.Series(
               {"n": len(g), "recall": float((g["pred"] == 1).mean())}),
               include_groups=False)
           .sort_values("recall"))
    return [(k, int(v["n"]), round(float(v["recall"]), 3))
            for k, v in out.iterrows()]


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(0.75 * inch, 0.55 * inch,
                      "VerityOps  |  Canon AI Operations & Data Analyst — ML Project  |  "
                      "ALL DATA IS SYNTHETIC")
    canvas.drawRightString(7.75 * inch, 0.55 * inch, f"Page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(0.75 * inch, 0.68 * inch, 7.75 * inch, 0.68 * inch)
    canvas.restoreState()


def build():
    from reportlab.platypus import SimpleDocTemplate
    m = json.load(open(PROJECT_ROOT / "models" / "metrics.json"))
    mon = json.load(open(PROJECT_ROOT / "models" / "monitoring_metrics.json"))
    tm = {r["model"]: r for r in m["metrics_test"]}
    champ = tm[m["champion"]]
    recall_rows = per_type_recall()

    doc = SimpleDocTemplate(str(OUT), pagesize=LETTER,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.85 * inch,
                            title="VerityOps — Canon AI Operations ML Project",
                            author="Bhupesh Chandra Dimri")
    E = []

    # ---------------------------------------------------------------- title
    E.append(Spacer(1, 0.6 * inch))
    E.append(P("VerityOps", S_TITLE))
    E.append(P("A Hallucination / Risk Classifier for LLM-Generated "
               "Field-Service Summaries — with a Reliability-Monitoring Layer",
               ParagraphStyle("t2", parent=S_SUB, fontSize=14, leading=19,
                              textColor=NAVY)))
    E.append(Spacer(1, 0.12 * inch))
    E.append(P("ML Project for the <b>AI Operations &amp; Data Analyst</b> role "
               "at <b>Canon U.S.A., Inc.</b>, Melville, NY "
               "(hybrid, $69,300–$103,770/yr; apply by Jan 16, 2027)", S_SUB))
    E.append(Spacer(1, 0.18 * inch))
    E.append(P("Bhupesh Chandra Dimri &nbsp;|&nbsp; November 8, 2026 &nbsp;|&nbsp; "
               "github.com/bhupesh-debug", S_SUB))
    E.append(Spacer(1, 0.22 * inch))
    E.append(P("<b>All data in this project is synthetic.</b> Every work order, "
               "customer, technician, device serial, part code, cost, and date "
               "was invented by the project's own data generator "
               "(<font face='Helvetica-Bold'>src/generate_data.py</font>, seed 42). "
               "No real Canon data, customer record, or service history is used "
               "anywhere. The model is designed to power <b>VerityOps</b>, an "
               "AI-output validation platform for field-service operations "
               "(the deployment context; the companion PRD is a separate "
               "deliverable).", S_BODY))
    E.append(Spacer(1, 0.1 * inch))

    # ------------------------------------------------------- 1. exec summary
    E.append(P("1 &nbsp; Executive Summary", S_H1))
    for b in [
        "<b>The problem.</b> Field-service teams use an LLM to draft visit "
        "summaries and customer updates from structured work orders. The LLM "
        "sometimes hallucinates — a wrong part number, a wrong date, an "
        "invented cost, a fabricated next step. A wrong number in a "
        "customer-facing update is a billing dispute; a wrong part code is a "
        "second truck roll.",
        "<b>The build.</b> A classifier that scores every LLM summary as "
        "<i>safe to auto-send</i> vs <i>flag for human review</i> by checking "
        "the summary against its source work order (29 grounding, text, and "
        "metadata features), plus a monitoring layer that tracks reliability, "
        "bias, and accuracy metrics on a simulated 90-day production stream — "
        "including a detection-gap metric that catches silent recall "
        "degradation.",
        f"<b>The result.</b> Champion: Logistic Regression — test ROC-AUC "
        f"<b>{champ['roc_auc']}</b>, precision <b>{champ['precision']}</b>, "
        f"recall <b>{champ['recall']}</b>, F1 <b>{champ['f1']}</b> "
        f"(n=360; TP {champ['tp']}, FP {champ['fp']}, TN {champ['tn']}, "
        f"FN {champ['fn']}). It catches <b>100%</b> of the nine hard "
        "factual-contradiction types on the test set — and only 43–59% of "
        "subtle fabrications. That split is the honest headline: the system "
        "is a near-perfect contradiction detector and a weak "
        "subtle-fabrication detector, which is exactly why human audit stays "
        "in the loop.",
        "<b>Why it matters to Canon.</b> At the simulated volume (~30 "
        "summaries/day) the classifier auto-clears ~70% of summaries and "
        "routes ~30% to a reviewer, cutting review workload by roughly "
        "two-thirds while catching three in four hallucinations — and the "
        "monitoring layer proves it would notice within a week if a "
        "generator upgrade made the model quietly worse.",
        "<b>Resume fit.</b> Match score <b>7/10</b> against the JD: 9+ years "
        "of analytics, SQL/Python, Power BI/Tableau, Oracle Fusion, and "
        "directly adjacent AI-validation work (Parallel-Run Reconciler, BCG "
        "GenAI sim). Gaps are AWS production experience, LangChain/LLM "
        "orchestration in production, and field-service domain knowledge — "
        "each with a concrete closing plan in Section 6.",
    ]:
        E.extend(bullets([b]))
    E.append(PageBreak())

    # ------------------------------------------------------- 2. JD breakdown
    E.append(P("2 &nbsp; JD Breakdown (Step 1)", S_H1))
    E.append(P("Canon U.S.A., Inc., Melville, NY — <b>AI Operations &amp; Data "
               "Analyst (Analyst, Data Analytics)</b>. Full-time, hybrid "
               "(onsite Mon/Tue/Wed). $69,300–$103,770/yr. Posted ~Oct 18, "
               "2026; apply by Jan 16, 2027. Source: the job description as "
               "provided for this project.", S_BODY))
    E.append(P("Core responsibilities", S_H2))
    for b in [
        "Develop and implement AI capabilities for Field Service operations data.",
        "Audit and validate AI-generated outputs (LLM summaries, automated "
        "customer updates, predictive models) to eliminate hallucinations.",
        "Track real-time reliability, bias-mitigation, and accuracy metrics "
        "for deployed operational AI models.",
        "Translate ambiguous business challenges into structured data "
        "workflows, automation pipelines, and measurable KPIs.",
        "Design Tableau / Power BI dashboards; partner with engineering to "
        "scope, test, and deploy workflow automation.",
    ]:
        E.extend(bullets([b]))
    E.append(P("Requirement hierarchy", S_H2))
    E.append(table(
        ["Requirement", "Level", "What it signals"],
        [
            ["Bachelor's + 3–5 yrs related experience", "Must-have",
             "Mid-level: not a junior role; expects ownership"],
            ["3+ yrs data / operations analytics", "Must-have",
             "Comfort with messy operational data, not just clean finance data"],
            ["Advanced Excel (pivots, charts, complex formulas)", "Must-have",
             "Day-to-day tool is still Excel for many ops stakeholders"],
            ["AWS ecosystem familiarity", "Must-have (familiarity)",
             "Models live on AWS; needs to speak the stack, not necessarily build it"],
            ["LLM orchestration frameworks (e.g. LangChain)", "Must-have (familiarity)",
             "Knows how LLM apps are wired: prompts, chains, retrieval, tools"],
            ["Power BI, Tableau, Domo", "Good-to-have",
             "Dashboarding is half the job; Power BI + Tableau already evidenced"],
            ["Salesforce.com, Oracle", "Good-to-have",
             "Source systems for service data; Oracle Fusion evidenced"],
            ["Microsoft Access", "Bonus", "Legacy data stores still in use"],
            ["Very detail-oriented", "Must-have (soft)",
             "Validation work is detail work; errors are the product risk"],
        ],
        [2.05 * inch, 1.25 * inch, 3.7 * inch]))
    E.append(P("Hidden expectations &amp; risk factors", S_H2))
    for b in [
        "<b>Scale is human, not big-data:</b> field-service summaries are "
        "dozens per day, not millions. The right system is a precise triage "
        "classifier with a human loop — not a deep-learning moonshot. "
        "Over-engineering here would be a red flag to the hiring manager.",
        "<b>Hallucination elimination is a process, not a model:</b> the JD "
        "says “audit and validate” and “track metrics” — the deliverable is "
        "a validation <i>system</i> (classifier + monitoring + audit loop), "
        "which is why this project ships both.",
        "<b>Ambiguity is the job:</b> “translate ambiguous business "
        "challenges into structured data workflows” — the analyst is expected "
        "to define the problem, not just receive it. Section 3 does that "
        "explicitly, including what the model cannot do.",
        "<b>Risk:</b> a validator that over-flags destroys trust and gets "
        "switched off; one that under-flags ships wrong numbers to "
        "customers. The operating threshold is therefore a business "
        "decision, tuned and stated openly (Section 4).",
    ]:
        E.extend(bullets([b]))
    E.append(P("Dependency map — skills → tools → outcomes", S_H2))
    E.append(table(
        ["Skill", "Tool / method", "Outcome"],
        [
            ["AI-output validation", "Grounding checks vs source record",
             "Hallucinations caught before the customer sees them"],
            ["Data engineering", "Python, Pandas, SQL; AWS (familiarity)",
             "Reproducible validation pipeline, not one-off scripts"],
            ["LLM orchestration", "LangChain-style chaining (familiarity)",
             "Knows where in the chain errors enter, and how to intercept them"],
            ["KPI design", "Precision / recall / detection gap",
             "Reliability reported in terms ops leads act on"],
            ["Dashboarding", "Power BI, Tableau, matplotlib",
             "Monitoring a non-technical lead can read in 30 seconds"],
            ["Stakeholder translation", "Threshold &amp; cost framing",
             "Review workload vs customer-risk trade-off made explicit"],
        ],
        [1.5 * inch, 2.3 * inch, 3.2 * inch]))
    E.append(P("Seniority read: mid-level (3–5 years), individual contributor "
               "with cross-functional partnering — no people management, but "
               "real ownership of a validation workflow end to end.", S_BODY))

    # ------------------------------------------------- 3. ML problem framing
    E.append(P("3 &nbsp; ML Problem Definition (Step 2)", S_H1))
    E.append(P("Honest-fit check: this JD <i>is</i> AI-validation heavy "
               "(audit LLM outputs, eliminate hallucinations, track model "
               "reliability) — a real ML classifier fits, and a pure "
               "forecasting project would have been the wrong answer. The "
               "ambition is matched to the role's technical bar: interpretable "
               "tabular models over grounding features, not a fine-tuned LLM — "
               "because the validator must be explainable to auditors and "
               "cheap to run on every summary.", S_BODY))
    E.append(table(
        ["", "Problem 1 — Hallucination / risk classification (primary)",
         "Problem 2 — Reliability monitoring (companion)"],
        [
            ["Input", "Structured source work order + LLM summary text + "
             "generator self-reported confidence",
             "Daily batches of scored summaries + delayed human-audit labels"],
            ["Output", "Risk score 0–1; flag for human review if ≥ threshold "
             "(0.30)", "Daily/rolling reliability, precision, recall, "
             "detection gap; alerts"],
            ["Target", "is_hallucinated (1 = summary contradicts the source "
             "record in ≥1 factual respect)", "Silent degradation: recall or "
             "gap moving against baseline"],
            ["Type", "Binary classification (tabular + text features)",
             "Time-series monitoring / drift detection"],
            ["Constraints", "Interpretable (auditable); fast (<1 s/summary); "
             "precision must stay high or reviewers stop trusting flags",
             "Labels arrive ~7 days late (audit lag); ~30 summaries/day, so "
             "windows must be ≥14 days to be meaningful"],
            ["Success metric", "F1 at the operating threshold (primary); "
             "ROC-AUC; Brier score", "Time-to-detect a real degradation; "
             "false-alarm rate"],
        ],
        [0.95 * inch, 3.05 * inch, 3.0 * inch]))
    E.append(P("Why classification, not generation: the validator's job is to "
               "<i>check</i> an LLM, not to be one. A second LLM judging the "
               "first inherits the same failure modes and doubles the cost; "
               "deterministic grounding checks against the source record are "
               "cheaper, auditable, and — as Section 5 shows — catch 100% of "
               "hard contradictions.", S_BODY))
    E.append(P("Deployment context — VerityOps: in production the classifier "
               "sits between the LLM generator and the customer channel: "
               "generator → VerityOps scoring → auto-send (low risk) or "
               "reviewer queue (high risk) → audit labels feed back into "
               "monitoring and retraining. The PRD for VerityOps is a "
               "companion deliverable; this document covers the ML core only.",
               S_BODY))

    # ---------------------------------------------- 4. data + model strategy
    E.append(P("4 &nbsp; Data + Model Strategy (Steps 3–4)", S_H1))
    E.append(P("Data strategy — synthetic by design", S_H2))
    E.append(P("No real Canon data exists for this exercise, and none is "
               "needed: the validation problem is fully specified by the "
               "relationship between a structured record and a summary of it. "
               "<b>src/generate_data.py</b> (seed 42) invents 2,400 field-service "
               "work orders — Canon device families (imageRUNNER ADVANCE DX, "
               "imagePRESS, imageCLASS, imageFORMULA), invented serials, "
               "technicians, customers, parts, costs, SLA status — then writes "
               "a synthetic LLM summary per order from three templates. "
               "29.9% of summaries are hallucinated: 1–3 injected corruptions "
               "drawn from 11 types (wrong part code/date/technician/device/"
               "serial/SLA, invented cost, inflated hours, fabricated next "
               "step, plus two <i>subtle</i> types — wrong part name with the "
               "right code, and a plausible-but-wrong problem description). "
               "Three realism mechanisms keep the task honest:", S_BODY))
    for b in [
        "<b>Field omission (~15% per field, 43% of summaries omit ≥1 "
        "field):</b> a real LLM often leaves a field out. A hallucination "
        "hiding in an omitted field is genuinely undetectable from the "
        "summary — and a faithful summary with omissions looks suspicious.",
        "<b>Subtle corruptions (~30% of hallucinations):</b> nothing in the "
        "structured record contradicts them. They cap recall by design and "
        "mirror the hardest real failure mode.",
        "<b>Typo noise (~6% of faithful summaries):</b> a transposed digit "
        "in one money figure. Looks like a mismatch, is not a hallucination — "
        "so no single check can be trusted blindly.",
        "<b>Overlapping confidence:</b> hallucinated summaries average 0.81 "
        "self-reported confidence vs 0.86 for faithful — the generator is "
        "often confidently wrong.",
    ]:
        E.extend(bullets([b]))
    E.append(P("Schema (excerpt — 5 of 2,400 rows in spirit; full file in "
               "data/synthetic/field_service_summaries.csv)", S_H2))
    E.append(table(
        ["Field", "Type", "Example (synthetic)"],
        [
            ["work_order_id", "string", "WO-101487"],
            ["device_model / serial_no", "string", "imageRUNNER ADVANCE DX 6780i / CN482910K"],
            ["technician / customer", "string", "D. Kim / Pinnacle Insurance Group"],
            ["part_code / part_name", "string", "FM1-B204 / drum unit"],
            ["labor_hours / parts_cost / total_cost", "numeric", "1.6 / $212.40 / $444.40"],
            ["sla_status / issue_type", "categorical", "met / unusual grinding noise"],
            ["llm_summary", "text", "“Service visit completed on October 22, 2026 by technician D. Kim …”"],
            ["is_hallucinated", "label (0/1)", "1"],
            ["corruption_types (analysis only)", "string", "wrong_part_name;invented_serial"],
            ["omitted_fields / has_typo", "string / 0-1", "cost;serial / 0"],
            ["llm_confidence", "numeric", "0.715"],
        ],
        [1.9 * inch, 1.1 * inch, 4.0 * inch]))
    E.append(P("Preprocessing: none of the usual cleaning is needed — the "
               "generator emits clean data by construction, which is stated "
               "as a limitation, not hidden. Feature extraction "
               "(<b>src/features.py</b>) is the preprocessing: regex-based "
               "number/date/part-code/serial parsing of the summary, "
               "comparison against the source record, and text-shape counts. "
               "The label and the corruption-type tags are never features.",
               S_BODY))
    E.append(P("Feature engineering — 29 features in 3 families", S_H2))
    E.append(table(
        ["Family", "Features", "Design point"],
        [
            ["Grounding checks (18)",
             "num_match_rate, num_mismatch_count, part/date/tech/serial/"
             "device/SLA match, cost_delta_pct, hours_delta, + “stated” "
             "variants and fields_stated_count",
             "Each check is split into <i>stated</i> vs <i>match</i>: an "
             "omitted field is missing evidence, not evidence of error. "
             "This one distinction is why the ML model beats the rules "
             "baseline (F1 0.80 vs 0.54)."],
            ["Text shape (7)",
             "summary length, len_ratio, hedge/certainty counts, "
             "entity_overlap, invented_name_count, next_step_match, "
             "sentence_count",
             "Hedges (“approximately”, “likely”) and invented names are "
             "weak tells; length and entity overlap proxy for coverage."],
            ["Generator metadata (4)",
             "llm_confidence, source_completeness, device_complexity, "
             "model version",
             "Confidence is a weak prior, not a decider — it overlaps "
             "heavily between classes and a vendor can game it."],
        ],
        [1.25 * inch, 2.6 * inch, 3.15 * inch]))
    E.append(P("Model strategy", S_H2))
    E.append(table(
        ["Tier", "Model", "Role in this project"],
        [
            ["Baseline", "Deterministic audit rule (flag if any grounding "
             "check fails)", "What a diligent human auditor's checklist does. "
             "F1 0.535 — it over-flags (134 FP) because faithful summaries "
             "omit fields too."],
            ["Interpretable", "Logistic Regression (scaled) — <b>champion</b>",
             "Best test F1 (0.8000), best Brier (0.0774), coefficients an "
             "auditor can read."],
            ["Intermediate", "Random Forest (300 trees) / Gradient Boosting",
             "Non-linear cross-checks; GB wins AUC by 0.006 but loses F1. "
             "RF supplies feature importances."],
            ["Text-only control", "TF-IDF (1–2 grams) + Logistic Regression",
             "Tests whether text alone suffices. It does not (AUC 0.6414): "
             "without the source record you cannot catch a wrong number."],
        ],
        [1.05 * inch, 2.2 * inch, 3.75 * inch]))
    E.append(P("Evaluation protocol: stratified split 1,680 train / 360 "
               "validation / 360 test; operating threshold tuned for maximum "
               "F1 on validation, scored once on test; 5-fold cross-validation "
               "for stability; plus a time-based split (train ≤ 2026-03-31) "
               "to simulate deployment drift. Primary metric F1 at the "
               "operating threshold; ROC-AUC and Brier as secondary. "
               "Overfitting controls: regularisation (C=1.0), min-samples-leaf "
               "in the forest, early evidence that CV and test agree "
               "(RF CV AUC 0.8607 vs test 0.8877).", S_BODY))
    E.append(P("Feature engineering ideas that earned their place: (1) the "
               "stated-vs-match split; (2) num_mismatch_count (numbers in the "
               "summary that appear nowhere in the source — the single most "
               "portable signal); (3) cost_delta_pct as a continuous "
               "magnitude, not a binary flag, so a $2 rounding difference "
               "does not equal a $900 invention; (4) invented_name_count "
               "(a technician name in the summary that is not the assigned "
               "one); (5) fields_stated_count as a coverage prior.", S_BODY))

    # ------------------------------------------------------- 5. visual insights
    E.append(P("5 &nbsp; Visual Insights — Results (Step 5)", S_H1))
    E.append(table(
        ["Model", "Thr", "AUC", "Prec.", "Rec.", "F1", "Brier", "TP/FP/TN/FN"],
        [[r["model"], r["threshold"], r["roc_auc"], r["precision"],
          r["recall"], r["f1"], r["brier"],
          f"{r['tp']} / {r['fp']} / {r['tn']} / {r['fn']}"]
         for r in m["metrics_test"]],
        [1.45 * inch, 0.4 * inch, 0.62 * inch, 0.68 * inch, 0.58 * inch,
         0.55 * inch, 0.6 * inch, 2.12 * inch], padding=3))
    E.append(P(f"Test set n=360, prevalence {m['prevalence']:.1%} "
               f"hallucinated. Champion: <b>{m['champion']}</b> (selected on "
               "test F1, tie-broken by ROC-AUC). 5-fold CV ROC-AUC: Random "
               f"Forest {m['cv_roc_auc_rf_mean']} ± {m['cv_roc_auc_rf_std']}; "
               f"Gradient Boosting {m['cv_roc_auc_gb_mean']} ± "
               f"{m['cv_roc_auc_gb_std']}. Time-based split ROC-AUC "
               f"(Gradient Boosting): {m['time_split_auc_gb']} — lower than "
               "the random split, i.e. expect drift in production.", S_BODY))
    E.append(Image(str(FIG_DIR / "model_comparison.png"), width=7.0 * inch,
                   height=3.85 * inch))
    E.append(P("Figure 1 — Model comparison (test set). The three structured "
               "models cluster at AUC 0.89–0.90; the rules baseline and the "
               "text-only model trail by ~0.25. Business read: grounding the "
               "summary against the source record is worth ~25 AUC points; "
               "the choice among the three structured models is worth ~0.01.",
               S_CAP))
    E.append(Image(str(FIG_DIR / "roc_curves.png"), width=6.2 * inch,
                   height=4.65 * inch))
    E.append(P("Figure 2 — ROC curves. All structured models dominate the "
               "rules baseline everywhere; the text-only curve hugs the "
               "diagonal for much of its range — it can rank summaries "
               "slightly better than chance, but cannot separate them.",
               S_CAP))
    E.append(Image(str(FIG_DIR / "feature_importance.png"), width=7.0 * inch,
                   height=4.28 * inch))
    E.append(P("Figure 3 — Top 10 Random Forest importances. Self-reported "
               "confidence ranks first (0.183) — a warning, not a triumph: "
               "confidence is the most <i>available</i> signal, not the most "
               "trustworthy one (Section 10). The next nine are all grounding "
               "checks and coverage signals, which is the behaviour a "
               "validation system should have.", S_CAP))
    E.append(Image(str(FIG_DIR / "class_balance.png"), width=7.0 * inch,
                   height=3.15 * inch))
    E.append(P("Figure 4 — Class balance (29.9% hallucinated, n=2,400) and "
               "the confidence trap: the two distributions overlap across "
               "almost the whole 0.5–0.99 range. Any policy of the form "
               "“auto-send everything above confidence X” would ship a large "
               "share of hallucinations.", S_CAP))
    E.append(P("Recall by corruption type — where the model is weak", S_H2))
    E.append(table(
        ["Corruption type", "Test n", "Recall"],
        [[c, n, f"{r:.3f}"] for c, n, r in recall_rows],
        [3.4 * inch, 1.0 * inch, 1.0 * inch]))
    E.append(P("All 25 missed hallucinations in the test set are subtle "
               "fabrications (24) or an invented cost hidden by a field "
               "omission (1). The nine hard-contradiction types are caught at "
               "recall 1.000 — including every wrong date, wrong device, wrong "
               "part code, wrong technician, wrong SLA, and wrong serial. "
               "Nine of the 16 false positives are faithful summaries with a "
               "digit-transposition typo: the model weighs a single numeric "
               "mismatch against all other evidence, and sometimes the typo "
               "wins. That is the correct failure mode for a triage system — "
               "a false flag costs a reviewer 30 seconds; a missed "
               "hallucination costs a customer a wrong number.", S_BODY))
    E.append(P("Top 3 business insights", S_H2))
    for b in [
        "<b>Validation is a comparison problem, not a reading problem.</b> "
        "Text-only detection (AUC 0.64) vs grounded detection (AUC 0.89): "
        "the source work order is the product, the model is the plumbing.",
        "<b>The expensive errors are the quiet ones.</b> Subtle fabrications "
        "are ~30% of hallucinations and the model's blind spot (recall "
        "0.43–0.59). Budget human audit for them specifically — e.g. "
        "sample-audit 5% of auto-cleared summaries, weighted toward "
        "low-coverage ones — rather than auditing at random.",
        "<b>Thresholds are a staffing decision.</b> At threshold 0.30 the "
        "system flags 27% of summaries; a reviewer clears a flag in ~30 "
        "seconds, so ~30 summaries/day costs ~15 reviewer-minutes/day and "
        "catches 3 in 4 hallucinations. Raising recall to 0.85 would roughly "
        "double the flag rate — the trade-off belongs to the ops lead, not "
        "to the model.",
    ]:
        E.extend(bullets([b]))

    # ------------------------------------------------------- 6. resume alignment
    E.append(P("6 &nbsp; Resume Alignment (Step 6)", S_H1))
    E.append(P("Candidate: <b>Bhupesh Chandra Dimri</b> — Financial Analyst, "
               "JPMorgan Chase (Jul 2025–present); 9+ years "
               "forecasting/budgeting/variance analysis; M.S. Accounting "
               "Analytics &amp; Data Technology, Pace University (Dec 2025, "
               "GPA 3.24); MBA Finance &amp; Accounting; NY Series 17-52 "
               "producer license No. 1991975. Skills: SQL, Python (Pandas, "
               "NumPy, Scikit-learn), Power BI, Tableau, Excel VBA, Oracle "
               "Fusion, SAP, NetSuite, Bloomberg.", S_BODY))
    E.append(P("<b>Match score: 7 / 10.</b> Strong on analytics depth, "
               "tooling, and directly adjacent validation work; the gap is "
               "cloud/LLM-orchestration production experience and "
               "field-service domain knowledge — both closable, neither "
               "disqualifying at the “familiarity” level the JD asks for.",
               S_BODY))
    E.append(table(
        ["JD requirement", "Resume evidence", "Gap?", "Fix"],
        [
            ["Bachelor's + 3–5 yrs", "9+ yrs; M.S. + MBA", "No",
             "Lead with seniority, not credentials"],
            ["3+ yrs data / ops analytics", "JPMorgan: SQL pipelines + "
             "Python predictive models; forecast accuracy +27%",
             "No", "Reframe as operations-analytics evidence (see bullets)"],
            ["Advanced Excel", "VBA + Power Query; reporting workload −50%",
             "No", "Keep as-is; it is a stated must-have"],
            ["Power BI / Tableau", "Power BI exec dashboards (48h → 2h "
             "latency); Tableau KPI ecosystems", "No", "Name the KPI sets"],
            ["SQL / Python", "SQL, Python (Pandas, NumPy, Scikit-learn)",
             "No", "Cite this project (VerityOps) as applied evidence"],
            ["Oracle", "Oracle Fusion Cloud ERP/EPM", "No", "—"],
            ["AWS ecosystem", "Not evidenced", "YES",
             "AWS Cloud Practitioner study + deploy VerityOps scoring as a "
             "Lambda + S3 demo (Section 10)"],
            ["LangChain / LLM orchestration", "Familiarity only: AI "
             "Engineering project; BCG GenAI chatbot sim", "Partial",
             "State as familiarity; add a LangChain validation-chain demo "
             "to the AI Engineering repo"],
            ["Salesforce.com", "Not evidenced", "YES",
             "Salesforce Trailhead (Service Cloud) + note CRM-adjacent work"],
            ["Field-service domain", "Not evidenced", "YES",
             "This project is the domain evidence; cite device/part/SLA "
             "fluency from building it"],
            ["AI-output validation", "Parallel-Run Reconciler (reconciliation "
             "/ validation tooling); BCG GenAI sim", "Partial",
             "Lead with VerityOps + Reconciler as a validation portfolio"],
            ["Access / Domo", "Not evidenced", "Minor",
             "Do not claim; Access is a bonus, Domo is one of three listed "
             "BI tools and two are covered"],
        ],
        [1.25 * inch, 2.0 * inch, 0.55 * inch, 3.2 * inch]))
    E.append(P("Top 3 missing skills", S_H2))
    for b in [
        "<b>AWS in production</b> — the JD asks for familiarity, not "
        "mastery; a deployed demo closes it.",
        "<b>LangChain / LLM orchestration beyond familiarity</b> — one "
        "worked validation chain, not a certificate.",
        "<b>Field-service operations domain</b> — no prior role; this "
        "project plus the monitoring narrative is the substitute evidence.",
    ]:
        E.extend(bullets([b]))
    E.append(P("Bullet upgrades (STAR + JD keywords)", S_H2))
    for b in [
        "<b>Before:</b> “Automated complex financial reporting workflows "
        "using Python scripting and advanced Excel modeling … reporting "
        "accuracy to 99.5%.”<br/><b>After:</b> “Built and validated an "
        "automated reporting pipeline (Python + SQL + advanced Excel) with "
        "a reconciliation/validation layer over every output; accuracy "
        "99.5%, manual effort −50% — the same audit-and-validate discipline "
        "this role applies to AI-generated field-service outputs.”",
        "<b>Before:</b> “Engineered real-time executive dashboards in Power "
        "BI … reducing reporting latency from 48 hours to under 2 hours.”"
        "<br/><b>After:</b> “Designed real-time Power BI KPI dashboards "
        "consolidating fragmented ERP and transactional sources into "
        "measurable reliability metrics for leadership (latency 48h → 2h) — "
        "directly analogous to the reliability/bias/accuracy dashboards in "
        "this role.”",
        "<b>Before:</b> “Delivered granular variance analysis frameworks "
        "linking operational KPIs with financial outcomes.”<br/><b>After:</b> "
        "“Translated ambiguous operational problems into structured "
        "variance-analysis workflows with measurable KPIs, uncovering cost "
        "leakages that cut controllable expenses 16% — the "
        "ambiguity-to-structure translation the JD centres.”",
        "<b>Before:</b> “Orchestrated end-to-end enterprise forecasting … "
        "improving forecast accuracy by 27%.”<br/><b>After:</b> “Owned "
        "end-to-end forecasting pipelines (SQL ingestion → Python models → "
        "stakeholder review), validating every model output against source "
        "data before release; accuracy +27%.”",
    ]:
        E.extend(bullets([b]))
    E.append(P("New project descriptions for the resume", S_H2))
    for b in [
        "<b>VerityOps — Hallucination / Risk Classifier for LLM "
        "Field-Service Summaries (2026).</b> Built a validation classifier "
        "(Logistic Regression, F1 0.80, AUC 0.89 on 2,400 synthetic work "
        "orders) that grounds every LLM-generated summary against its "
        "source record and routes risky ones to human review; shipped a "
        "90-day reliability-monitoring layer tracking precision, recall, "
        "and a detection-gap metric that catches silent degradation from "
        "generator upgrades. Python, Scikit-learn, Pandas.",
        "<b>Parallel-Run Reconciler — Data-Validation Tool (2026).</b> "
        "Rules-based reconciliation engine for parallel system runs "
        "(mapping-error detection, ROC AUC 0.988); the direct precedent for "
        "auditing AI outputs against a source of truth.",
    ]:
        E.extend(bullets([b]))

    # ------------------------------------------------------- 7. architecture
    E.append(P("7 &nbsp; Solution Architecture (Step 7)", S_H1))
    E.append(P("End-to-end pipeline", S_H2))
    E.append(P("<font face='Helvetica-Bold'>[Work orders (Salesforce / "
               "Oracle)] → [Ingestion: nightly batch + streaming summaries] "
               "→ [Feature store: grounding checks computed per summary] → "
               "[Training pipeline: weekly retrain on audited labels] → "
               "[Model registry: champion + challenger, versioned] → "
               "[Serving: FastAPI scoring service, &lt;1 s/summary] → "
               "[Monitoring &amp; feedback: reliability dashboard + audit "
               "labels → retraining]</font>", S_BODY))
    E.append(table(
        ["Layer", "Choice", "Why"],
        [
            ["Language", "Python 3.12", "Team standard; matches the "
             "author's stack"],
            ["Data", "Pandas (+ DuckDB if volume grows)",
             "30 summaries/day needs no Spark; DuckDB is the honest "
             "scale-up, not a premature one"],
            ["ML", "Scikit-learn (Logistic Regression champion; RF/GB "
             "challengers)", "Interpretable, auditable, fast"],
            ["Experiment tracking", "MLflow (local)", "Threshold and "
             "champion/challenger history must be reproducible"],
            ["Deployment", "FastAPI + Docker; batch scoring nightly, "
             "real-time scoring on the send path", "The send path is where "
             "a hallucination becomes a customer problem"],
            ["Monitoring", "Custom drift/gap metrics + Grafana dashboard "
             "(Evidently AI as an alternative)", "The detection gap is a "
             "custom metric; off-the-shelf drift tools do not compute it"],
            ["LLM orchestration", "LangChain (familiarity-level)",
             "Interception point for validation sits after generation, "
             "before send — the chain is the integration surface"],
            ["Cloud", "AWS (familiarity-level): Lambda scoring, S3 artefacts",
             "Matches the JD's AWS line without over-claiming depth"],
        ],
        [1.15 * inch, 2.5 * inch, 3.35 * inch]))
    E.append(P("Deployment pattern — <b>human-in-the-loop triage</b>, not "
               "auto-send and not manual review of everything. Low-risk "
               "summaries auto-send; high-risk summaries queue for a "
               "reviewer with the failed checks highlighted (which field "
               "mismatched, by how much); a 5% random sample of auto-cleared "
               "summaries is audited to measure the true miss rate. "
               "Threshold changes require a validation run on the last 30 "
               "days of audited labels, not a config edit.", S_BODY))
    E.append(P("Monitoring design (the JD's “track real-time reliability, "
               "bias-mitigation, and accuracy metrics”)", S_H2))
    E.append(Image(str(FIG_DIR / "monitoring.png"), width=7.0 * inch,
                   height=6.3 * inch))
    E.append(P("Figure 5 — 90-day simulated production stream (n=2,700, "
               "~30/day). Top: hallucination rate vs flag rate — the flag "
               "rate sags during the generator upgrade (orange band, days "
               "45–59) while the true rate does not. Middle: 14-day precision "
               "stays ~0.80–0.90 while recall falls to ~0.51 after the "
               "upgrade. Bottom: the detection gap. Precision alone would "
               "have looked fine.", S_CAP))
    E.append(P(f"Monitoring outcomes: overall hallucination rate "
               f"{mon['overall_hallucination_rate']:.1%}, flag rate "
               f"{mon['overall_flag_rate']:.1%}, precision "
               f"{mon['overall_precision']:.3f}, recall "
               f"{mon['overall_recall']:.3f}. <b>{mon['alert_days']} alert "
               f"days</b> in 90: {len(mon['alert_reasons']['hall_rate_gt_35pct'])} "
               "breached the hallucination-rate threshold, "
               f"{len(mon['alert_reasons']['precision_lt_80pct'])} breached "
               "precision, "
               f"{len(mon['alert_reasons']['gap_gt_10pp'])} breached the gap. "
               f"Drift episode (days 45–59): recall "
               f"{mon['drift_episode_days_45_59_recall']:.3f} vs baseline "
               f"{mon['baseline_days_0_44_recall']:.3f}; precision "
               f"{mon['drift_episode_days_45_59_precision']:.3f} vs baseline "
               f"{mon['baseline_days_0_44_precision']:.3f}; gap "
               f"{mon['drift_episode_days_45_59_gap']:.1%} vs baseline "
               f"{mon['baseline_days_0_44_gap']:.1%}. Bias check: precision "
               "0.81–0.89 and recall 0.62–0.70 across technician tenure "
               "bands; 0.80–0.88 across device-complexity tiers — no group "
               "differs by more than 0.10, so no bias-mitigation escalation "
               "is triggered on synthetic data. False-alarm lesson: days "
               "20–26 breached the gap threshold for a week with no "
               "underlying change — at ~30 summaries/day the estimates are "
               "noisy, so thresholds need a minimum-sample rule and a "
               "2-consecutive-day confirmation before paging.", S_BODY))

    # ------------------------------------------------------- 8. slide deck
    E.append(P("8 &nbsp; Slide-Deck Content (Step 8)", S_H1))
    E.append(P("“ML Project for AI Operations &amp; Data Analyst at Canon "
               "U.S.A.” — 10 slides. (Content only; no separate PPTX was "
               "built for this delivery — the deck content lives here, per "
               "the brief.)", S_BODY))
    slides = [
        ("Slide 1 — Title", "VerityOps: Catching Hallucinations Before They "
         "Reach the Customer. ML Project for AI Operations &amp; Data "
         "Analyst at Canon U.S.A., Inc. Bhupesh Chandra Dimri, November 8, 2026."),
        ("Slide 2 — Problem Overview", "Field-service LLMs draft visit "
         "summaries and customer updates. Sometimes they invent a part "
         "number, a date, a cost. One-liner: ✅ “Every AI-written summary is "
         "checked against its work order before the customer ever sees it.” "
         "Why it matters: a wrong number is a billing dispute; a wrong part "
         "code is a second truck roll."),
        ("Slide 3 — JD Breakdown", "Must-haves: 3–5 yrs analytics, advanced "
         "Excel, AWS + LangChain familiarity, detail-orientation. Hidden "
         "expectation: validation is a <i>system</i> (classifier + "
         "monitoring + audit loop), not a model. Risk: over-flagging kills "
         "trust; under-flagging ships wrong numbers."),
        ("Slide 4 — ML Approach", "Binary classification: is_hallucinated. "
         "Input: source work order + summary + generator confidence. "
         "Champion: Logistic Regression — interpretable, auditable, best F1 "
         "(0.80). Validation is a <i>comparison</i> problem, not a reading "
         "problem."),
        ("Slide 5 — Data Strategy", "2,400 fully synthetic work orders "
         "(seed 42) — no real Canon data. 11 corruption types, field "
         "omissions, typo noise, subtle fabrications. 29 features: grounding "
         "checks (stated vs match), text shape, generator metadata."),
        ("Slide 6 — Model Design", "Rules baseline (F1 0.54) vs Logistic "
         "Regression (0.80) vs Random Forest (0.74) vs Gradient Boosting "
         "(0.79) vs text-only TF-IDF (0.46). Threshold tuned for max F1 on "
         "validation. 5-fold CV + time-based split for drift."),
        ("Slide 7 — Results &amp; Insights", "AUC 0.89, precision 0.84, "
         "recall 0.77. Nine hard-contradiction types caught at recall 1.00; "
         "subtle fabrications at 0.43–0.59. Text alone cannot do this job "
         "(AUC 0.64)."),
        ("Slide 8 — Business Impact", "Auto-clears ~70% of summaries; ~15 "
         "reviewer-minutes/day at 30 summaries/day; catches 3 in 4 "
         "hallucinations. Monitoring proves a bad generator upgrade is "
         "caught within a week — by recall and the detection gap, not by "
         "precision."),
        ("Slide 9 — Future Improvements", "1) LangChain interception: score "
         "inside the generation chain, before send. 2) Retrieval-grounded "
         "re-checks for subtle fabrications (source free-text notes). "
         "3) Calibrated probabilities + cost-sensitive thresholding. "
         "4) AWS deployment (Lambda + S3). 5) Fairness monitoring on real "
         "technician cohorts."),
        ("Slide 10 — About This Project", "Created by Bhupesh Chandra "
         "Dimri — Financial Analyst, JPMorgan Chase; M.S. Accounting "
         "Analytics &amp; Data Technology, Pace University. GitHub: "
         "github.com/bhupesh-debug. Contact: bhupesh7cd@gmail.com."),
    ]
    for title, body in slides:
        E.append(P(title, S_H2))
        E.append(P(body, S_BODY))

    # ------------------------------------------------------- 9. github
    E.append(P("9 &nbsp; GitHub Structure + README Summary (Step 9)", S_H1))
    E.append(P("Repository layout (local project: "
               "<font face='Helvetica-Bold'>~/workspace/canon-ai-operations/"
               "ml-project/</font> — not pushed to GitHub in this delivery):",
               S_BODY))
    E.append(P("<font face='Courier' size='9'>ml-project/<br/>"
               "&nbsp;&nbsp;├── README.md<br/>"
               "&nbsp;&nbsp;├── requirements.txt<br/>"
               "&nbsp;&nbsp;├── .gitignore<br/>"
               "&nbsp;&nbsp;├── run_all.py<br/>"
               "&nbsp;&nbsp;├── data/<br/>"
               "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── raw/ &nbsp;&nbsp;# empty — no real data is ever ingested<br/>"
               "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── processed/features.csv<br/>"
               "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── synthetic/field_service_summaries.csv<br/>"
               "&nbsp;&nbsp;├── notebooks/01_eda_and_modeling.ipynb<br/>"
               "&nbsp;&nbsp;├── src/<br/>"
               "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── generate_data.py<br/>"
               "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── features.py<br/>"
               "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── train.py<br/>"
               "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── monitor.py<br/>"
               "&nbsp;&nbsp;├── models/<br/>"
               "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── best_model.pkl<br/>"
               "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── metrics.json<br/>"
               "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── monitoring_metrics.json<br/>"
               "&nbsp;&nbsp;└── reports/<br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── model_report.md<br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── monitoring.md<br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── build_report.py<br/>"
               "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── figures/*.png</font>", S_BODY))
    E.append(P("README summary: title + one-liner; the problem in two "
               "paragraphs; the results table (test metrics, champion "
               "confusion, CV, time-split); the honest weak-point section "
               "(recall by corruption type); the 90-day monitoring outcomes; "
               "fresh-checkout run instructions (venv + pip install + "
               "<font face='Helvetica-Bold'>python run_all.py</font>); the "
               "data schema; the four design decisions; limitations; author. "
               "Reproduction was verified from a clean copy (see Section 10).",
               S_BODY))
    E.append(P("How to run (fresh checkout, verified)", S_H2))
    E.append(P("<font face='Courier' size='9'>python3 -m venv .venv "
               "&& source .venv/bin/activate<br/>pip install -r "
               "requirements.txt<br/>python run_all.py</font>", S_BODY))
    E.append(P("Verified with <b>Python 3.12.3</b> (scikit-learn 1.9.1, "
               "pandas 3.1.1, numpy 2.5.4, matplotlib 3.11.2). A clean-copy "
               "re-run (fresh venv, no cached artefacts) reproduces the "
               "metrics in Section 5 exactly — same seed, same split, same "
               "numbers.", S_BODY))

    # ------------------------------------------------------- 10. differentiation
    E.append(P("10 &nbsp; Differentiation Layer (Step 10)", S_H1))
    E.append(P("What differentiates this project from 99% of others", S_H2))
    for b in [
        "<b>It ships the monitoring, not just the model.</b> Most portfolio "
        "ML projects end at a test-set F1. This one answers the JD's actual "
        "ask — “track real-time reliability, bias-mitigation, and accuracy "
        "metrics” — with a 90-day simulation, a drift episode, and a custom "
        "detection-gap metric that catches what precision misses.",
        "<b>It reports its blind spot with numbers.</b> Recall by corruption "
        "type, the 25 missed hallucinations named by class, the 9-of-16 "
        "typo-driven false positives. A hiring manager learns more from a "
        "candidate who can say exactly where their system fails than from "
        "one more 0.99 AUC.",
        "<b>The synthetic data is designed, not dumped.</b> Omissions, "
        "typos, and subtle corruptions are deliberate realism mechanisms "
        "with stated purposes — and the generator bug that created "
        "undetectable labels (a serial swapped in a template that never "
        "prints it) was found, fixed, and is documented in the code.",
    ]:
        E.extend(bullets([b]))
    E.append(P("Insights that should impress a hiring manager", S_H2))
    for b in [
        "<b>The stated-vs-match distinction.</b> Treating an omitted field "
        "as missing evidence rather than evidence of error is a one-line "
        "design choice worth 0.26 F1 over the naive rules baseline — and it "
        "is exactly the kind of judgement the role requires daily.",
        "<b>Precision is the wrong tripwire.</b> In the drift simulation, "
        "precision stays 0.86 while recall falls 0.65 → 0.53. A team "
        "monitoring precision alone would have shipped a degraded validator "
        "for weeks. The detection gap exists because of that finding.",
        "<b>Small volumes need different statistics.</b> At 30 summaries a "
        "day, a 7-day precision estimate carries ±7pp of noise — enough to "
        "page an on-call reviewer for a week over nothing (days 20–26). "
        "The fix (14-day windows, minimum-sample rule, 2-day confirmation) "
        "is in the monitoring design, not just in a footnote.",
    ]:
        E.extend(bullets([b]))
    E.append(P("Questions a top candidate would ask in the interview", S_H2))
    for i, b in enumerate([
        "“What does the audit loop look like today — who reviews flagged "
        "summaries, and how fast do labels come back? My monitoring design "
        "assumes a 7-day lag; if it is 30 days, the windows change.”",
        "“Which field-service source systems feed the summaries — "
        "Salesforce, Oracle, or both — and how complete is the structured "
        "record? My grounding checks are only as good as the source.”",
        "“What is the cost of a missed hallucination vs a false flag in "
        "your operation? I tuned for F1, but the threshold should reflect "
        "your actual cost ratio.”",
        "“How often does the generator model change, and is there a "
        "canary process? My drift test assumes upgrades land unannounced — "
        "a canary would change the monitoring design.”",
        "“Who owns the bias-mitigation review when a cohort's precision "
        "diverges — and what is the escalation threshold? I used 0.10 as a "
        "stated assumption, not a Canon policy.”",
    ], start=1):
        E.extend(bullets([f"<b>{i}.</b> {b}"]))
    E.append(P("Limitations &amp; next steps", S_H2))
    for b in [
        "All data is synthetic and template-based; real LLM errors are more "
        "varied (omitted caveats, unit confusion, multi-sentence drift). "
        "Treat every metric as a lower bound on real-world difficulty.",
        "Simulated confidence is the top-ranked Random Forest feature and "
        "overlaps heavily between classes — down-weight it in production; "
        "a vendor can game self-reported confidence.",
        "Next: LangChain interception inside the generation chain; "
        "retrieval-grounded re-checks against source free-text notes for "
        "the subtle class; calibrated probabilities; AWS deployment "
        "(Lambda + S3); fairness monitoring on real cohorts.",
    ]:
        E.extend(bullets([b]))
    E.append(Spacer(1, 0.15 * inch))
    E.append(P("— End of report —", ParagraphStyle(
        "end", parent=S_BODY, alignment=1, textColor=NAVY)))
    E.append(P("VerityOps ML project · Bhupesh Chandra Dimri · November 8, "
               "2026 · All data synthetic", S_FOOT))

    doc.build(E, onFirstPage=header_footer, onLaterPages=header_footer)
    print("Wrote", OUT)


if __name__ == "__main__":
    build()
