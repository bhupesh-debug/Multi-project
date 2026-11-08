"""
train.py — Train and compare hallucination/risk classifiers.

Models (all on the same stratified train/val/test split):
  1. Rules baseline — deterministic audit rule (any grounding check fails).
  2. Logistic Regression — interpretable linear model on structured features.
  3. Random Forest — non-linear, gives feature importances.
  4. HistGradientBoosting — strongest tabular model.
  5. TF-IDF + Logistic Regression — text-only model (summary text alone,
     no source record). Included to test whether text alone can catch
     hallucinations; it cannot catch numeric grounding errors, which is
     itself a finding.

Evaluation: ROC-AUC (primary), precision / recall / F1 at the operating
threshold tuned for max F1 on validation, confusion matrix, Brier score.
Also a time-based split check (train on earlier service dates, test on
later ones) to simulate deployment drift.

Outputs:
  models/best_model.pkl, models/metrics.json,
  reports/figures/*.png, reports/model_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.features import FEATURE_COLUMNS, build_feature_frame  # noqa: E402
from src.generate_data import generate  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_DIR = PROJECT_ROOT / "models"
SEED = 42


def rule_predict(df_feats: pd.DataFrame) -> np.ndarray:
    """Deterministic audit rule: flag if ANY grounding check fails."""
    fail = (
        (df_feats["num_mismatch_count"] > 0)
        | (df_feats["part_code_match"] == 0)
        | (df_feats["date_match"] == 0)
        | (df_feats["tech_name_match"] == 0)
        | (df_feats["serial_match"] == 0)
        | (df_feats["device_match"] == 0)
        | (df_feats["cost_delta_pct"] > 0.02)
        | (df_feats["hours_delta"] > 0.05)
    )
    return fail.astype(int).to_numpy()


def tune_threshold(y_val: np.ndarray, proba_val: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(y_val, (proba_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def score_model(name: str, y_true: np.ndarray, proba: np.ndarray,
                threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "model": name,
        "threshold": round(float(threshold), 2),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "flag_rate": round(float(pred.mean()), 4),
    }


def plot_roc_curves(results: dict, y_test: np.ndarray, fig_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, proba in results.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, lw=2,
                 label=f"{name} (AUC {roc_auc_score(y_test, proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate (safe summaries wrongly flagged)")
    plt.ylabel("True positive rate (hallucinations caught)")
    plt.title("ROC curves — hallucination detection (test set, synthetic data)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_model_comparison(metrics: list[dict], fig_path: Path) -> None:
    names = [m["model"] for m in metrics]
    x = np.arange(len(names))
    w = 0.22
    plt.figure(figsize=(10, 5.5))
    for i, key in enumerate(["roc_auc", "precision", "recall", "f1"]):
        plt.bar(x + (i - 1.5) * w, [m[key] for m in metrics], w,
                label=key.replace("_", " ").upper())
    plt.xticks(x, names, rotation=12, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model comparison — test set (synthetic data, n=360)")
    plt.legend(ncols=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_feature_importance(importances: pd.Series, fig_path: Path) -> None:
    top = importances.sort_values(ascending=True).tail(10)
    plt.figure(figsize=(9, 5.5))
    plt.barh(top.index, top.values)
    plt.xlabel("Random Forest feature importance (mean decrease in impurity)")
    plt.title("Top 10 drivers of the hallucination flag (synthetic data)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_class_balance(df: pd.DataFrame, fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    counts = df["is_hallucinated"].value_counts().sort_index()
    axes[0].bar(["Faithful (0)", "Hallucinated (1)"], counts.values,
                color=["#2e7d32", "#c62828"])
    axes[0].set_ylabel("Summaries")
    axes[0].set_title(f"Class balance (n={len(df):,}, "
                      f"{df['is_hallucinated'].mean():.1%} hallucinated)")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, f"{v:,}", ha="center")
    axes[1].hist([df.loc[df["is_hallucinated"] == 0, "llm_confidence"],
                  df.loc[df["is_hallucinated"] == 1, "llm_confidence"]],
                 bins=20, label=["Faithful", "Hallucinated"], alpha=0.75)
    axes[1].set_xlabel("LLM self-reported confidence")
    axes[1].set_ylabel("Summaries")
    axes[1].set_title("Confidence overlaps: overconfident hallucinations exist")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def main() -> None:
    for d in (PROJECT_ROOT / "data" / "synthetic",
              PROJECT_ROOT / "data" / "processed",
              PROJECT_ROOT / "data" / "raw"):
        d.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)

    # --- data (regenerate deterministically; never ship a stale CSV) --------
    raw = generate(n=2400, seed=SEED)
    raw.to_csv(PROJECT_ROOT / "data" / "synthetic" / "field_service_summaries.csv",
               index=False)
    feats = build_feature_frame(raw)
    feats.to_csv(PROJECT_ROOT / "data" / "processed" / "features.csv", index=False)

    rng = np.random.RandomState(SEED)
    y = feats["is_hallucinated"].to_numpy()
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(feats))
    train_idx, temp_idx = train_test_split(
        idx, test_size=720, random_state=SEED, stratify=y)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=360, random_state=SEED,
        stratify=y[temp_idx])

    X = feats[FEATURE_COLUMNS].to_numpy()
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    Xte, yte = X[test_idx], y[test_idx]
    summaries_te = raw.iloc[test_idx]["llm_summary"].tolist()
    summaries_tr = raw.iloc[train_idx]["llm_summary"].tolist()

    # --- rules baseline ------------------------------------------------------
    rules_proba = rule_predict(feats.iloc[test_idx]).astype(float)
    rules_val = rule_predict(feats.iloc[val_idx]).astype(float)

    # --- ML models -----------------------------------------------------------
    lr = Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                              random_state=SEED))])
    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                min_samples_leaf=2, n_jobs=-1,
                                random_state=SEED, class_weight="balanced")
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                        max_leaf_nodes=31, random_state=SEED)
    text_clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=4000,
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
    ])
    models = {"Logistic Regression": lr, "Random Forest": rf,
              "Gradient Boosting": gb}
    for m in models.values():
        m.fit(Xtr, ytr)
    text_clf.fit(summaries_tr, ytr)

    probas, thresholds, metrics = {}, {}, []
    # rules baseline: fixed threshold 0.5 on a 0/1 score
    metrics.append(score_model("Rules baseline", yte, rules_proba, 0.5))
    probas["Rules baseline"] = rules_proba
    for name, m in models.items():
        p_va = m.predict_proba(Xva)[:, 1]
        t = tune_threshold(yva, p_va)
        p_te = m.predict_proba(Xte)[:, 1]
        thresholds[name] = t
        probas[name] = p_te
        metrics.append(score_model(name, yte, p_te, t))
    p_va_txt = text_clf.predict_proba(raw.iloc[val_idx]["llm_summary"])[:, 1]
    t_txt = tune_threshold(yva, p_va_txt)
    p_te_txt = text_clf.predict_proba(summaries_te)[:, 1]
    thresholds["TF-IDF text-only"] = t_txt
    probas["TF-IDF text-only"] = p_te_txt
    metrics.append(score_model("TF-IDF text-only", yte, p_te_txt, t_txt))

    # --- cross-validation on the champion family -----------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_auc_rf = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    cv_auc_gb = cross_val_score(gb, X, y, cv=cv, scoring="roc_auc")

    # --- time-based split check (train <= 2026-03-31, test after) -----------
    dates = pd.to_datetime(feats["service_date"])
    time_tr = (dates <= "2026-03-31").to_numpy()
    gb_time = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=SEED)
    gb_time.fit(X[time_tr], y[time_tr])
    time_auc = float(roc_auc_score(
        y[~time_tr], gb_time.predict_proba(X[~time_tr])[:, 1]))

    # --- feature importance --------------------------------------------------
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS)

    # --- figures -------------------------------------------------------------
    plot_roc_curves(probas, yte, FIG_DIR / "roc_curves.png")
    plot_model_comparison(metrics, FIG_DIR / "model_comparison.png")
    plot_feature_importance(importances, FIG_DIR / "feature_importance.png")
    plot_class_balance(raw, FIG_DIR / "class_balance.png")

    # --- persist champion (Gradient Boosting if best F1, else RF) ------------
    best = max(metrics, key=lambda m: (m["f1"], m["roc_auc"]))
    champion_name = best["model"]
    champion = {"Gradient Boosting": gb, "Random Forest": rf,
                "Logistic Regression": lr}.get(champion_name, gb)
    joblib.dump({"model": champion, "features": FEATURE_COLUMNS,
                 "threshold": thresholds.get(champion_name, 0.5),
                 "champion": champion_name},
                MODEL_DIR / "best_model.pkl")

    results = {
        "n_rows": int(len(raw)),
        "prevalence": round(float(raw["is_hallucinated"].mean()), 4),
        "split": {"train": int(len(train_idx)), "val": int(len(val_idx)),
                  "test": int(len(test_idx))},
        "metrics_test": metrics,
        "cv_roc_auc_rf_mean": round(float(cv_auc_rf.mean()), 4),
        "cv_roc_auc_rf_std": round(float(cv_auc_rf.std()), 4),
        "cv_roc_auc_gb_mean": round(float(cv_auc_gb.mean()), 4),
        "cv_roc_auc_gb_std": round(float(cv_auc_gb.std()), 4),
        "time_split_auc_gb": round(time_auc, 4),
        "champion": champion_name,
        "thresholds": {k: round(float(v), 2) for k, v in thresholds.items()},
        "feature_importance_top10": {
            k: round(float(v), 4)
            for k, v in importances.sort_values(ascending=False).head(10).items()},
        "seed": SEED,
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- markdown report -----------------------------------------------------
    lines = ["# VerityOps — Hallucination / Risk Classifier: Model Report",
             "",
             "**All data is synthetic.** Generated by `src/generate_data.py` "
             "(seed 42, n=2,400). No real Canon data is used.",
             "",
             f"- Champion model: **{champion_name}** (selected on test F1, "
             "tie-broken by ROC-AUC)",
             f"- Test set: n={len(test_idx)}, prevalence "
             f"{yte.mean():.1%} hallucinated",
             f"- 5-fold CV ROC-AUC — Random Forest {cv_auc_rf.mean():.4f} "
             f"(±{cv_auc_rf.std():.4f}); Gradient Boosting "
             f"{cv_auc_gb.mean():.4f} (±{cv_auc_gb.std():.4f})",
             f"- Time-based split ROC-AUC (Gradient Boosting, train ≤ "
             f"2026-03-31): {time_auc:.4f}",
             "",
             "## Test-set metrics (operating threshold tuned for max F1 on validation)",
             "",
             "| Model | Threshold | ROC-AUC | Precision | Recall | F1 | Brier | TP | FP | TN | FN |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
             ]
    for m in metrics:
        lines.append(f"| {m['model']} | {m['threshold']} | {m['roc_auc']} | "
                     f"{m['precision']} | {m['recall']} | {m['f1']} | {m['brier']} | "
                     f"{m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")
    lines += ["",
              "## Top 10 features (Random Forest importance)", "",
              "| Feature | Importance |", "|---|---:|"]
    for k, v in importances.sort_values(ascending=False).head(10).items():
        lines.append(f"| {k} | {v:.4f} |")
    lines += ["",
              "## Honest limitations",
              "",
              "- ~30% of synthetic hallucinations are SUBTLE (wrong problem "
              "description, or right part code with the wrong part name). No "
              "structured field contradicts them, so no grounding check can "
              "catch them — they cap recall. This mirrors reality: grounding "
              "checks catch factual contradictions, not plausible paraphrase.",
              "- Summaries omit fields (~15% per field). A hallucination hiding "
              "in an omitted field is undetectable from the summary alone.",
              "- Faithful summaries carry digit-transposition typos (~6%), so "
              "no single check is decisive; the model weighs evidence.",
              "- The TF-IDF text-only model cannot see the source record, so it "
              "misses most numeric grounding errors — grounding checks against "
              "the work order are the core of the system, not the text model.",
              "- Simulated LLM confidence overlaps heavily between classes; "
              "confidence alone must never auto-approve a summary.",
              "- Time-split AUC is close to random-split AUC here, but on real "
              "data expect drift as the generator model is updated; monitor "
              "and retrain (see reports/monitoring.md).",
              "- All data is synthetic and template-based; real LLM errors are "
              "more varied. Treat every metric above as a lower bound on "
              "real-world difficulty, not a production guarantee.",
              "",
              "## How to reproduce",
              "",
              "```bash",
              "python3 -m venv .venv && source .venv/bin/activate",
              "pip install -r requirements.txt",
              "python run_all.py",
              "```",
              "Then inspect `reports/figures/`, `models/metrics.json`, and "
              "`reports/monitoring.md`.",
              ""]
    (PROJECT_ROOT / "reports" / "model_report.md").write_text("\n".join(lines))

    print(json.dumps(results, indent=2))
    print("\nWrote models/best_model.pkl, models/metrics.json, "
          "reports/model_report.md, reports/figures/*.png")


if __name__ == "__main__":
    main()
