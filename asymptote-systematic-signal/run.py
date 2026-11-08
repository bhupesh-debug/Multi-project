"""Asymptote Systematic Signal — end-to-end pipeline.

Run from the project root:
    python run.py

Steps: generate synthetic data -> build features -> train models ->
evaluate -> backtest -> scenarios -> charts -> Excel KPI dashboard ->
written model report + saved best model.

All data is synthetic (seed 42). See README.md for limitations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from backtest import compare_strategies, positions_from_proba, run_backtest  # noqa: E402
from charts import (  # noqa: E402
    chart_correlation,
    chart_drawdown,
    chart_equity_curves,
    chart_feature_importance,
    chart_model_comparison,
    chart_monte_carlo,
    chart_prices_regimes,
)
from dashboard import build_dashboard  # noqa: E402
from data_pipeline import generate_synthetic_kpis, generate_synthetic_prices  # noqa: E402
from evaluate import write_model_report  # noqa: E402
from feature_engineering import (  # noqa: E402
    FEATURE_COLUMNS,
    build_model_frame,
    time_split,
)
from model import (  # noqa: E402
    evaluate_binary_models,
    feature_importances,
    rule_metrics,
    save_model,
    train_binary_models,
    train_regime_model,
)
from scenario import (  # noqa: E402
    block_bootstrap_returns,
    deterministic_scenarios,
    monte_carlo_summary,
)

DATA_SYN = ROOT / "data" / "synthetic"
DATA_PROC = ROOT / "data" / "processed"
CHARTS = ROOT / "reports" / "charts"


def main() -> None:
    DATA_SYN.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic data
    print("1/8  Generating synthetic data ...")
    prices = generate_synthetic_prices()
    kpis = generate_synthetic_kpis(prices)
    prices.to_csv(DATA_SYN / "synthetic_prices.csv", index=False)
    kpis.to_csv(DATA_SYN / "synthetic_business_kpis.csv", index=False)

    # 2. Features
    print("2/8  Building features ...")
    frame = build_model_frame(prices)
    frame.to_csv(DATA_PROC / "model_frame.csv", index=False)
    train, test = time_split(frame)
    print(f"     train rows={len(train):,}  test rows={len(test):,}")

    # 3. Train
    print("3/8  Training models ...")
    models = train_binary_models(train[FEATURE_COLUMNS], train["target_up"])
    regime_model = train_regime_model(train[FEATURE_COLUMNS], train["target_regime"])
    regime_acc = accuracy_score(
        test["target_regime"], regime_model.predict(test[FEATURE_COLUMNS])
    )
    print(f"     regime classifier accuracy (test): {regime_acc:.4f}")

    # 4. Evaluate
    print("4/8  Evaluating ...")
    results = evaluate_binary_models(models, test[FEATURE_COLUMNS], test["target_up"])
    rule = rule_metrics(test)
    results = pd.concat([results, pd.DataFrame([rule])], ignore_index=True)
    importances = feature_importances(models["rf"])
    print(results.to_string(index=False))

    # 5. Backtest
    print("5/8  Backtesting ...")
    best_name = results.sort_values("f1", ascending=False).iloc[0]["model"]
    best_model = models[best_name] if best_name in models else models["rf"]
    bt_metrics, curves = compare_strategies(test, best_model, FEATURE_COLUMNS)
    print(bt_metrics.to_string(index=False))
    save_model(best_model, str(ROOT / "models" / "best_model.pkl"))

    # 6. Scenarios
    print("6/8  Scenario analysis ...")
    ml_daily = curves["ml_signal"]
    scenarios = deterministic_scenarios(ml_daily["pnl"])
    mc = monte_carlo_summary(ml_daily["pnl"])
    print(scenarios.to_string(index=False))
    print("     Monte Carlo:", mc)

    # 7. Charts
    print("7/8  Charts ...")
    chart_prices_regimes(prices, CHARTS / "prices_regimes.png")
    chart_feature_importance(importances, CHARTS / "feature_importance.png")
    chart_model_comparison(results, CHARTS / "model_comparison.png")
    chart_equity_curves(curves, CHARTS / "equity_curves.png")
    chart_drawdown(curves, CHARTS / "drawdown.png")
    chart_correlation(frame, FEATURE_COLUMNS, CHARTS / "correlation_heatmap.png")
    paths = block_bootstrap_returns(ml_daily["pnl"].to_numpy())
    terminal = (1 + paths).prod(axis=1)
    chart_monte_carlo(terminal, CHARTS / "monte_carlo.png")

    # 8. Dashboard + report
    print("8/8  Dashboard + report ...")
    build_dashboard(
        ROOT / "reports" / "kpi_dashboard.xlsx",
        kpis,
        bt_metrics,
        curves,
        scenarios,
        mc,
        results,
    )
    write_model_report(
        ROOT / "reports" / "model_report.md",
        results,
        rule,
        bt_metrics,
        scenarios,
        mc,
        importances,
        regime_acc,
        len(train),
        len(test),
    )
    print("Done. See reports/model_report.md, reports/kpi_dashboard.xlsx, reports/charts/")


if __name__ == "__main__":
    main()
