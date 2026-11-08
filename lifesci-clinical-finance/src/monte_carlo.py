"""monte_carlo.py — budget / timeline scenario engine (SYNTHETIC).

Simulates cost-to-complete and duration for one representative program
using lognormal cost and normal timeline draws. 10,000 simulations,
seeded for reproducibility. Outputs P50 / P80 / P90 and the probability
of breaching the approved budget — the number a Finance lead actually
presents to a program committee.
"""
import numpy as np


def run_monte_carlo(base_cost_usd_m: float, base_months: float,
                    approved_budget_usd_m: float, n_sims: int = 10_000,
                    seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    # Illustrative volatility assumptions (SYNTHETIC — stated, not measured).
    costs = rng.lognormal(mean=np.log(base_cost_usd_m), sigma=0.25, size=n_sims)
    months = np.clip(rng.normal(loc=base_months, scale=base_months * 0.15, size=n_sims),
                     a_min=1, a_max=None)
    return {
        "n_sims": n_sims,
        "base_cost_usd_m": base_cost_usd_m,
        "base_months": base_months,
        "approved_budget_usd_m": approved_budget_usd_m,
        "cost_p50_usd_m": round(float(np.percentile(costs, 50)), 2),
        "cost_p80_usd_m": round(float(np.percentile(costs, 80)), 2),
        "cost_p90_usd_m": round(float(np.percentile(costs, 90)), 2),
        "cost_mean_usd_m": round(float(costs.mean()), 2),
        "months_p50": round(float(np.percentile(months, 50)), 1),
        "months_p90": round(float(np.percentile(months, 90)), 1),
        "prob_budget_breach": round(float((costs > approved_budget_usd_m).mean()), 4),
        "prob_overrun_20pct": round(float((costs > base_cost_usd_m * 1.2).mean()), 4),
    }


if __name__ == "__main__":
    # Representative synthetic Phase II oncology program.
    print(run_monte_carlo(base_cost_usd_m=36.0, base_months=30,
                          approved_budget_usd_m=40.0))
