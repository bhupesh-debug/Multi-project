"""Variance analysis: budget-vs-actual and forecast-vs-actual.

Core FP&A logic — every number ties out. The revenue bridge decomposes
exactly: AUM effect + fee-rate effect + performance-fee effect == total
revenue variance (up to rounding). The operating-profit bridge likewise
sums exactly from budget profit to actual profit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MATERIAL_PCT = 0.10   # |variance %| >= 10%
MATERIAL_USD = 50_000  # and |variance $| >= $50k  -> MATERIAL flag


def variance(actual: float, baseline: float) -> dict:
    """Variance vs a baseline (budget or forecast). Positive $ = favourable
    for revenue/profit; for costs the caller interprets the sign."""
    var_usd = actual - baseline
    var_pct = var_usd / baseline if baseline else np.nan
    material = abs(var_pct) >= MATERIAL_PCT and abs(var_usd) >= MATERIAL_USD
    return {"variance_usd": var_usd, "variance_pct": var_pct,
            "material": bool(material),
            "direction": "Favourable" if var_usd > 0 else ("Adverse" if var_usd < 0 else "Flat")}


def decompose_revenue_variance(row: pd.Series) -> dict:
    """Split mgmt-fee + perf-fee revenue variance into three exact effects.

    AUM effect  = (AUM_a - AUM_b) * rate_b / 12
    Rate effect = (mgmt_a - mgmt_b) - AUM effect   (residual, so the bridge
                  ties out exactly despite rounding in the generated data)
    Perf effect = perf_fee_a - perf_fee_b
    Sum == (mgmt_a + perf_a) - (mgmt_b + perf_b)  (exact by construction).
    """
    aum_effect = (row["aum_actual_m"] - row["aum_budget_m"]) * 1e6 * (row["fee_rate_budget_bps"] / 10_000) / 12
    rate_effect = (row["mgmt_fee_actual"] - row["mgmt_fee_budget"]) - aum_effect
    perf_effect = row["perf_fee_actual"] - row["perf_fee_budget"]
    total = (row["mgmt_fee_actual"] + row["perf_fee_actual"]) - (row["mgmt_fee_budget"] + row["perf_fee_budget"])
    return {"aum_effect": aum_effect, "rate_effect": rate_effect,
            "perf_fee_effect": perf_effect, "total_revenue_variance": total,
            "bridge_check": aum_effect + rate_effect + perf_effect - total}


def strategy_variance_table(strat: pd.DataFrame) -> pd.DataFrame:
    """Per strategy-month: revenue variance + its three effects + flags."""
    out = []
    for _, r in strat.iterrows():
        dec = decompose_revenue_variance(r)
        rev = variance(r["mgmt_fee_actual"] + r["perf_fee_actual"],
                       r["mgmt_fee_budget"] + r["perf_fee_budget"])
        fcast = variance(r["mgmt_fee_actual"] + r["perf_fee_actual"],
                         r["mgmt_fee_forecast"] + r["perf_fee_forecast"])
        out.append({"month": r["month"], "strategy": r["strategy"],
                    "revenue_budget": r["mgmt_fee_budget"] + r["perf_fee_budget"],
                    "revenue_actual": r["mgmt_fee_actual"] + r["perf_fee_actual"],
                    "revenue_variance_usd": rev["variance_usd"],
                    "revenue_variance_pct": rev["variance_pct"],
                    "revenue_material": rev["material"],
                    "aum_effect": dec["aum_effect"], "rate_effect": dec["rate_effect"],
                    "perf_fee_effect": dec["perf_fee_effect"],
                    "forecast_variance_usd": fcast["variance_usd"],
                    "forecast_variance_pct": fcast["variance_pct"],
                    "aum_variance_pct": (r["aum_actual_m"] - r["aum_budget_m"]) / r["aum_budget_m"]})
    return pd.DataFrame(out)


def firm_variance_table(firm: pd.DataFrame) -> pd.DataFrame:
    """Firm-level monthly variance: revenue, cost, operating profit, KPIs."""
    rows = []
    for _, r in firm.iterrows():
        rev = variance(r["revenue_actual"], r["revenue_budget"])
        cost = variance(r["cost_actual"], r["cost_budget"])  # +$ = adverse for costs
        profit = variance(r["operating_profit_actual"], r["operating_profit_budget"])
        rows.append({"month": r["month"],
                     "revenue_variance_usd": rev["variance_usd"],
                     "revenue_variance_pct": rev["variance_pct"],
                     "revenue_material": rev["material"],
                     "cost_variance_usd": cost["variance_usd"],
                     "cost_variance_pct": cost["variance_pct"],
                     "cost_material": cost["material"],
                     "profit_variance_usd": profit["variance_usd"],
                     "profit_variance_pct": profit["variance_pct"],
                     "operating_margin_budget": r["operating_profit_budget"] / r["revenue_budget"],
                     "operating_margin_actual": r["operating_profit_actual"] / r["revenue_actual"],
                     "revenue_per_head_actual": r["revenue_actual"] / r["headcount_actual"],
                     "cost_per_head_actual": r["cost_actual"] / r["headcount_actual"],
                     "aum_actual_m": r["aum_actual_m"]})
    return pd.DataFrame(rows)


def profit_bridge(firm: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Waterfall bridge for one month: budget profit -> actual profit.

    Steps sum exactly: budget profit + revenue variance - cost variance
    == actual profit (cost variance enters with a minus sign because a
    cost overrun reduces profit).
    """
    r = firm.loc[firm["month"] == month].iloc[0]
    rev_var = r["revenue_actual"] - r["revenue_budget"]
    cost_var = r["cost_actual"] - r["cost_budget"]
    steps = [
        ("Budget operating profit", r["operating_profit_budget"]),
        ("Revenue variance", rev_var),
        ("Cost variance", -cost_var),
        ("Actual operating profit", r["operating_profit_actual"]),
    ]
    df = pd.DataFrame(steps, columns=["step", "amount_usd"])
    df["cumulative_usd"] = [steps[0][1], steps[0][1] + rev_var,
                            steps[0][1] + rev_var - cost_var, steps[3][1]]
    return df


def cost_variance_by_line(firm: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Opex variance broken down by cost line for one month."""
    from src.data_generation import COST_LINES
    r = firm.loc[firm["month"] == month].iloc[0]
    rows = []
    for line in COST_LINES:
        key = line.lower().replace(" ", "_")
        b, a = r[f"cost_{key}_budget"], r[f"cost_{key}_actual"]
        v = variance(a, b)
        rows.append({"cost_line": line, "budget": b, "actual": a,
                     "variance_usd": v["variance_usd"], "variance_pct": v["variance_pct"],
                     "material": v["material"],
                     "direction": "Adverse (overrun)" if v["variance_usd"] > 0 else "Favourable (saving)"})
    return pd.DataFrame(rows).sort_values("variance_usd", ascending=False)
