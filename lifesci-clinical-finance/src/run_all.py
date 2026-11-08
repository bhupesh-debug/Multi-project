"""run_all.py — end-to-end pipeline. Run:  python src/run_all.py

Steps: generate synthetic data -> load/clean -> engineer features ->
train regression (rNPV) + classification (transition success) ->
Monte Carlo budget scenarios -> charts -> save models + metrics JSON.
All data synthetic; all outputs reproducible (seed 42).
"""
import json
import pathlib
import sys

import joblib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data_pipeline import clean_programs, load_burn, load_programs
from evaluate import (plot_burn_variance, plot_feature_importance,
                      plot_model_comparison, plot_monte_carlo, plot_rnpv_distribution)
from feature_engineering import engineer, feature_matrix
from generate_data import generate_monthly_burn, generate_programs
from model import train_classification, train_regression
from monte_carlo import run_monte_carlo

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    # 1. Generate synthetic data (overwrites data/synthetic/ deterministically).
    programs_raw = generate_programs()
    burn = generate_monthly_burn(programs_raw)
    syn = ROOT / "data" / "synthetic"
    syn.mkdir(parents=True, exist_ok=True)
    programs_raw.to_csv(syn / "clinical_programs.csv", index=False)
    burn.to_csv(syn / "monthly_burn.csv", index=False)

    # 2. Load + clean + engineer.
    df = engineer(clean_programs(load_programs()))
    burn = load_burn()

    # 3. Regression: predict risk-adjusted NPV ($M).
    X_reg, reg_cols = feature_matrix(df, "regression")
    reg_results, reg_fitted, reg_best, _ = train_regression(X_reg, df["rnpv_usd_m"])

    # 4. Classification: predict next-phase transition success.
    X_clf, clf_cols = feature_matrix(df, "classification")
    clf_results, clf_fitted, clf_best, _ = train_classification(X_clf, df["transition_success"])

    # 5. Monte Carlo: representative synthetic Phase II oncology program.
    mc = run_monte_carlo(base_cost_usd_m=36.0, base_months=30, approved_budget_usd_m=40.0)

    # 6. Charts.
    charts = {
        "feature_importance": plot_feature_importance(reg_fitted["Random Forest"], reg_cols),
        "model_comparison": plot_model_comparison(reg_results),
        "rnpv_distribution": plot_rnpv_distribution(df),
        "burn_variance": plot_burn_variance(burn),
        "monte_carlo": plot_monte_carlo(mc["cost_p50_usd_m"], mc["cost_p80_usd_m"],
                                        mc["cost_p90_usd_m"], mc["approved_budget_usd_m"]),
    }

    # 7. Save best models + metrics.
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(reg_fitted[reg_best], models_dir / "best_rnpv_model.pkl")
    joblib.dump(clf_fitted[clf_best], models_dir / "best_transition_model.pkl")
    metrics = {
        "data": {"n_programs": int(len(df)), "n_burn_rows": int(len(burn)),
                 "synthetic": True, "seed": 42},
        "regression_rnpv_usd_m": reg_results,
        "regression_best": reg_best,
        "classification_transition_success": clf_results,
        "classification_best": clf_best,
        "monte_carlo": mc,
        "charts": charts,
        "rnpv_summary_usd_m": {
            "median": round(float(df["rnpv_usd_m"].median()), 2),
            "mean": round(float(df["rnpv_usd_m"].mean()), 2),
            "share_positive": round(float((df["rnpv_usd_m"] > 0).mean()), 3),
        },
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("\nOK — pipeline complete. Best regression:", reg_best, "| Best classifier:", clf_best)
    return metrics


if __name__ == "__main__":
    main()
