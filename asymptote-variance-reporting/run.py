#!/usr/bin/env python3
"""Run the full variance-analysis + automated-reporting pipeline.

    python run.py                  # report for the latest month (2025-12)
    python run.py --month 2025-03  # report for a specific month
    python run.py --seed 42        # reproducibility seed

Outputs (all under this project dir):
    data/synthetic/strategy_monthly.csv, firm_monthly.csv
    data/processed/strategy_variance.csv, firm_variance.csv, anomalies.csv
    reports/charts/*.png
    reports/variance_report_<YYYY-MM>.md
    reports/management_report_<YYYY-MM>.xlsx
    reports/model_report.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.anomaly import detect_anomalies
from src.charts import bridge_chart, forecast_chart, kpi_trend, variance_bars
from src.data_generation import generate_all
from src.forecast import forecast_firm
from src.reporting import build_markdown_report, build_workbook
from src.variance import (cost_variance_by_line, firm_variance_table,
                          profit_bridge, strategy_variance_table)


def main() -> int:
    ap = argparse.ArgumentParser(description="Asymptote variance-reporting engine")
    ap.add_argument("--month", default="2025-12", help="Reporting month YYYY-MM")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    month = pd.Timestamp(args.month + "-01")
    strat, firm = generate_all(seed=args.seed)
    if month not in firm["month"].values:
        print(f"Month {args.month} outside synthetic history "
              f"{firm['month'].min():%Y-%m}–{firm['month'].max():%Y-%m}")
        return 2

    strat_var = strategy_variance_table(strat)
    firm_var = firm_variance_table(firm)
    anomalies = detect_anomalies(firm_var, firm)
    forecasts = forecast_firm(firm)
    bridge = profit_bridge(firm, month)
    costs = cost_variance_by_line(firm, month)
    firm_row = firm.loc[firm["month"] == month].iloc[0]
    strat_latest = strat_var.loc[strat_var["month"] == month]

    # --- persist data -------------------------------------------------
    (ROOT / "data" / "synthetic").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "charts").mkdir(parents=True, exist_ok=True)
    strat.to_csv(ROOT / "data" / "synthetic" / "strategy_monthly.csv", index=False)
    firm.to_csv(ROOT / "data" / "synthetic" / "firm_monthly.csv", index=False)
    strat_var.to_csv(ROOT / "data" / "processed" / "strategy_variance.csv", index=False)
    firm_var.to_csv(ROOT / "data" / "processed" / "firm_variance.csv", index=False)
    anomalies.to_csv(ROOT / "data" / "processed" / "anomalies.csv", index=False)
    forecasts["outlook"].to_csv(ROOT / "data" / "processed" / "forecast_outlook.csv", index=False)

    # --- charts -------------------------------------------------------
    label = month.strftime("%B %Y")
    bridge_chart(bridge, label, ROOT / "reports" / "charts" / "profit_bridge.png")
    variance_bars(strat_var, ROOT / "reports" / "charts" / "variance_by_strategy.png")
    kpi_trend(firm, firm_var, ROOT / "reports" / "charts" / "kpi_trend.png")
    forecast_chart(firm, forecasts["outlook"], ROOT / "reports" / "charts" / "forecast.png")

    # --- reports ------------------------------------------------------
    summary = pd.DataFrame([{
        "month": label, "revenue_budget": firm_row["revenue_budget"],
        "revenue_actual": firm_row["revenue_actual"],
        "revenue_variance_usd": firm_row["revenue_actual"] - firm_row["revenue_budget"],
        "cost_budget": firm_row["cost_budget"], "cost_actual": firm_row["cost_actual"],
        "cost_variance_usd": firm_row["cost_actual"] - firm_row["cost_budget"],
        "operating_profit_budget": firm_row["operating_profit_budget"],
        "operating_profit_actual": firm_row["operating_profit_actual"],
        "operating_margin_actual_pct": round(
            firm_row["operating_profit_actual"] / firm_row["revenue_actual"] * 100, 1),
        "aum_actual_m": firm_row["aum_actual_m"],
        "headcount_actual": firm_row["headcount_actual"],
    }])
    kpis = firm_var.merge(firm[["month", "revenue_actual", "cost_actual",
                                 "headcount_actual"]], on="month")
    tag = month.strftime("%Y-%m")
    build_workbook(ROOT / "reports" / f"management_report_{tag}.xlsx", label,
                   summary, strat_latest, bridge, costs,
                   forecasts["outlook"], anomalies, kpis)
    build_markdown_report(ROOT / "reports" / f"variance_report_{tag}.md", label,
                          firm_row, bridge, costs, strat_latest,
                          forecasts["outlook"], anomalies,
                          forecasts["revenue_actual"]["best_model"],
                          forecasts["revenue_actual"]["mape"])

    from src.reporting import _md as _md_table
    model_lines = [
        "# Model report — forecast & anomaly detection (synthetic data)",
        "",
        "## Forecast model comparison (6-month time-based hold-out, Jul–Dec 2025)",
        "",
        "| Series | Best model | MAPE by model |",
        "|---|---|---|",
    ]
    for col, label_s in [("revenue_actual", "Revenue"), ("cost_actual", "Total cost")]:
        f = forecasts[col]
        mape_str = "; ".join(f"{k}: {v:.1%}" for k, v in f["mape"].items())
        model_lines.append(f"| {label_s} | {f['best_model']} | {mape_str} |")
    model_lines += [
        "",
        "## Anomaly detection (Isolation Forest + material-variance rules)",
        "",
        _md_table(anomalies),
        "",
        "## Honest read",
        "",
        "- Forecast MAPEs are measured on synthetic data with smooth, seeded "
        "patterns — they show the pipeline works, not real forecasting skill.",
        "- The injected shocks (Mar 2025 credit redemption, Jul 2025 technology "
        "spike, Nov 2024 macro performance surge) are the ground truth the "
        "anomaly layer is checked against.",
        "- 24 monthly observations is a small sample for unsupervised detection; "
        "every flag is a prompt for human review, not an automated conclusion.",
        "- KNOWN MISS: the injected Nov 2024 macro performance surge (+118% "
        "revenue variance, the largest single month) was NOT flagged ANOMALY — "
        "its Isolation Forest score (+0.001) fell just short of the threshold. "
        "The detector caught 2 of 3 injected shocks (Mar 2025, Jul 2025). "
        "A single-threshold rule would have caught Nov 2024; the joint-pattern "
        "model did not. That is a real limitation, not a tuning oversight to hide.",
        "- `rule_flag` means a material variance fired; `verdict` is ANOMALY only "
        "when the model agrees or a single cost line is material — so a month "
        "can show rule_flag=True and still be Normal.",
    ]
    (ROOT / "reports" / "model_report.md").write_text("\n".join(model_lines))

    import json, pickle
    from src.forecast import _ridge_forecast as _ridge_fit
    _, _, ridge_model = _ridge_fit(firm["revenue_actual"].values.astype(float))
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    with open(ROOT / "models" / "best_revenue_forecast_model.pkl", "wb") as fh:
        pickle.dump(ridge_model, fh)
    (ROOT / "models" / "model_card.json").write_text(json.dumps({
        "model": "Ridge (trend + month dummies), refit on 24 months of SYNTHETIC revenue",
        "target": "revenue_actual", "holdout_mape": forecasts["revenue_actual"]["mape"],
        "note": "Synthetic data only. Not a production forecasting model.",
    }, indent=2))

    print(f"Reporting month: {label}")
    print(f"Revenue variance: ${firm_row['revenue_actual'] - firm_row['revenue_budget']:+,.0f} "
          f"({(firm_row['revenue_actual'] - firm_row['revenue_budget']) / firm_row['revenue_budget']:+.1%})")
    print(f"Cost variance:    ${firm_row['cost_actual'] - firm_row['cost_budget']:+,.0f}")
    print(f"Profit variance:  ${firm_row['operating_profit_actual'] - firm_row['operating_profit_budget']:+,.0f}")
    print(f"Best revenue forecast model: {forecasts['revenue_actual']['best_model']} "
          f"(MAPE {forecasts['revenue_actual']['mape'][forecasts['revenue_actual']['best_model']]:.1%})")
    print(f"Best cost forecast model:    {forecasts['cost_actual']['best_model']} "
          f"(MAPE {forecasts['cost_actual']['mape'][forecasts['cost_actual']['best_model']]:.1%})")
    anom = anomalies.loc[anomalies["verdict"] == "ANOMALY", "month"].dt.strftime("%Y-%m").tolist()
    print(f"ANOMALY months: {anom}")
    print("Outputs written to data/, reports/ (see README for the full list).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
