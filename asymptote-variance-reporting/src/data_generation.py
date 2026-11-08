"""Synthetic data generation for the Asymptote variance-reporting project.

ALL DATA IS SYNTHETIC. Nothing here is real Asymptote Capital data, real
fund performance, or real market data. Strategy names are prefixed SYN_.
The generator is seeded (default 42) so every run reproduces the same data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42

STRATEGIES = ["SYN_EQ", "SYN_MACRO", "SYN_CREDIT", "SYN_FX"]
STRATEGY_LABELS = {
    "SYN_EQ": "Equity Long/Short (synthetic)",
    "SYN_MACRO": "Global Macro (synthetic)",
    "SYN_CREDIT": "Credit Relative Value (synthetic)",
    "SYN_FX": "FX / Rates (synthetic)",
}
# Annualised management-fee rate (bps) per strategy — budget assumption.
FEE_RATE_BPS = {"SYN_EQ": 150, "SYN_MACRO": 175, "SYN_CREDIT": 125, "SYN_FX": 100}
# Starting AUM ($m) at Jan 2024, budget path grows ~0.8%/month.
START_AUM_M = {"SYN_EQ": 420.0, "SYN_MACRO": 310.0, "SYN_CREDIT": 260.0, "SYN_FX": 180.0}
COST_LINES = ["Compensation", "Technology", "Market data", "Professional services", "Other opex"]


def generate_months(start: str = "2024-01", periods: int = 24) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="MS")


def generate_strategy_monthly(seed: int = SEED) -> pd.DataFrame:
    """One row per strategy-month: budget vs forecast vs actual AUM, fees, P&L.

    Injected, documented shocks (so anomaly detection has ground truth):
    * 2025-03 SYN_CREDIT: large redemption, AUM -18% vs budget.
    * 2025-07 firm Technology cost spike (see generate_firm_monthly).
    * 2024-11 SYN_MACRO: performance-fee surge (strong synthetic returns).
    """
    rng = np.random.default_rng(seed)
    months = generate_months()
    rows = []
    for strat in STRATEGIES:
        aum_budget = START_AUM_M[strat]
        for i, month in enumerate(months):
            # Budget: smooth 0.8%/mo growth, reset each January to the plan.
            year = month.year
            months_into_year = month.month - 1
            base = START_AUM_M[strat] * (1.008 ** (12 * (year - 2024) + months_into_year))
            budget_aum = base
            # Forecast: quarterly rolling reforecast — budget + drift learned so far.
            forecast_aum = budget_aum * (1 + 0.01 * np.sin(i / 3.0) + rng.normal(0, 0.01))
            # Actual: budget path + noise + shocks.
            actual_aum = budget_aum * (1 + rng.normal(0, 0.02))
            if strat == "SYN_CREDIT" and month == pd.Timestamp("2025-03-01"):
                actual_aum *= 0.82  # injected redemption shock
            if strat == "SYN_MACRO" and month == pd.Timestamp("2024-11-01"):
                actual_aum *= 1.06
            rate_bps_budget = FEE_RATE_BPS[strat]
            rate_bps_actual = rate_bps_budget + rng.normal(0, 3)
            mgmt_fee_budget = budget_aum * 1e6 * (rate_bps_budget / 10_000) / 12
            mgmt_fee_actual = actual_aum * 1e6 * (rate_bps_actual / 10_000) / 12
            mgmt_fee_forecast = forecast_aum * 1e6 * (rate_bps_budget / 10_000) / 12
            gross_ret = rng.normal(0.005, 0.015)
            if strat == "SYN_MACRO" and month == pd.Timestamp("2024-11-01"):
                gross_ret = 0.075  # injected performance surge
            perf_fee_actual = max(0.0, actual_aum * 1e6 * gross_ret * 0.20) if gross_ret > 0.01 else 0.0
            # Budget assumes the expected performance fee: with gross returns
            # ~ N(0.5%, 1.5%), E[0.20 * ret | ret > 1%] * P(ret > 1%) ≈ 0.15%
            # of AUM per month.
            perf_fee_budget = actual_aum * 1e6 * 0.0015
            perf_fee_forecast = perf_fee_budget * 1.1
            strat_cost_actual = actual_aum * 1e6 * rng.normal(0.0012, 0.0002)
            strat_cost_budget = budget_aum * 1e6 * 0.0012
            rows.append({
                "month": month, "strategy": strat,
                "strategy_label": STRATEGY_LABELS[strat],
                "aum_budget_m": round(budget_aum, 2),
                "aum_forecast_m": round(forecast_aum, 2),
                "aum_actual_m": round(actual_aum, 2),
                "fee_rate_budget_bps": rate_bps_budget,
                "fee_rate_actual_bps": round(rate_bps_actual, 1),
                "mgmt_fee_budget": round(mgmt_fee_budget, 0),
                "mgmt_fee_forecast": round(mgmt_fee_forecast, 0),
                "mgmt_fee_actual": round(mgmt_fee_actual, 0),
                "perf_fee_budget": round(perf_fee_budget, 0),
                "perf_fee_forecast": round(perf_fee_forecast, 0),
                "perf_fee_actual": round(perf_fee_actual, 0),
                "gross_return": round(gross_ret, 4),
                "strategy_cost_budget": round(strat_cost_budget, 0),
                "strategy_cost_actual": round(strat_cost_actual, 0),
            })
    return pd.DataFrame(rows)


def generate_firm_monthly(strategy_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Firm-level monthly P&L: revenue from strategies + firm opex lines."""
    rng = np.random.default_rng(seed + 1)
    months = generate_months()
    agg = strategy_df.groupby("month").agg(
        mgmt_fee_budget=("mgmt_fee_budget", "sum"),
        mgmt_fee_forecast=("mgmt_fee_forecast", "sum"),
        mgmt_fee_actual=("mgmt_fee_actual", "sum"),
        perf_fee_budget=("perf_fee_budget", "sum"),
        perf_fee_forecast=("perf_fee_forecast", "sum"),
        perf_fee_actual=("perf_fee_actual", "sum"),
        strategy_cost_budget=("strategy_cost_budget", "sum"),
        strategy_cost_actual=("strategy_cost_actual", "sum"),
        aum_budget_m=("aum_budget_m", "sum"),
        aum_actual_m=("aum_actual_m", "sum"),
    ).reset_index()
    rows = []
    headcount_budget = 32
    for _, r in agg.iterrows():
        month = r["month"]
        # Budget opex: grows ~0.5%/mo; actual = budget + noise + one spike.
        opex_budget = {}
        opex_actual = {}
        base = {"Compensation": 1_150_000, "Technology": 180_000, "Market data": 95_000,
                "Professional services": 70_000, "Other opex": 55_000}
        idx = list(months).index(month)
        for line in COST_LINES:
            b = base[line] * (1.005 ** idx)
            a = b * (1 + rng.normal(0, 0.03))
            if line == "Technology" and month == pd.Timestamp("2025-07-01"):
                a = b * 1.45  # injected vendor / infra cost spike
            if line == "Compensation" and month == pd.Timestamp("2025-01-01"):
                a = b * 1.08  # annual bonus accrual true-up
            opex_budget[line] = round(b, 0)
            opex_actual[line] = round(a, 0)
        headcount_actual = headcount_budget + (1 if month >= pd.Timestamp("2025-01-01") else 0)
        rev_budget = r["mgmt_fee_budget"] + r["perf_fee_budget"]
        rev_actual = r["mgmt_fee_actual"] + r["perf_fee_actual"]
        cost_budget = r["strategy_cost_budget"] + sum(opex_budget.values())
        cost_actual = r["strategy_cost_actual"] + sum(opex_actual.values())
        row = {"month": month, "aum_budget_m": round(r["aum_budget_m"], 1),
               "aum_actual_m": round(r["aum_actual_m"], 1),
               "revenue_budget": round(rev_budget, 0), "revenue_actual": round(rev_actual, 0),
               "revenue_forecast": round(r["mgmt_fee_forecast"] + r["perf_fee_forecast"], 0),
               "cost_budget": round(cost_budget, 0), "cost_actual": round(cost_actual, 0),
               "operating_profit_budget": round(rev_budget - cost_budget, 0),
               "operating_profit_actual": round(rev_actual - cost_actual, 0),
               "headcount_budget": headcount_budget, "headcount_actual": headcount_actual}
        for line in COST_LINES:
            row[f"cost_{line.lower().replace(' ', '_')}_budget"] = opex_budget[line]
            row[f"cost_{line.lower().replace(' ', '_')}_actual"] = opex_actual[line]
        rows.append(row)
    return pd.DataFrame(rows)


def generate_all(seed: int = SEED):
    strat = generate_strategy_monthly(seed)
    firm = generate_firm_monthly(strat, seed)
    return strat, firm


if __name__ == "__main__":
    s, f = generate_all()
    print(s.head(3).to_string())
    print(f.head(3).to_string())
    print(f"strategy rows={len(s)} firm rows={len(f)}")
