"""Scenario analysis and Monte Carlo stress testing.

Two layers, both on the *backtested* (synthetic) strategy returns:
1. Deterministic scenarios — bull / base / bear: shift the historical
   daily mean return and volatility and report the 1-year outlook.
2. Monte Carlo — resample daily strategy returns (block bootstrap, 20-day
   blocks to preserve autocorrelation) into 2,000 one-year paths; report
   terminal-wealth percentiles, probability of a >10% drawdown, and
   5% expected shortfall.

Everything here is conditional on synthetic data. The point is the
machinery (scenario engine, percentiles, tail metrics), not the numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_PATHS = 2000
HORIZON_DAYS = 252
SEED = 42


def block_bootstrap_returns(
    returns: np.ndarray, n_paths: int = N_PATHS, horizon: int = HORIZON_DAYS,
    block: int = 20, seed: int = SEED,
) -> np.ndarray:
    """Resample daily returns in blocks; returns (n_paths, horizon)."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, n - block + 1, size=(n_paths, n_blocks))
    paths = np.empty((n_paths, horizon))
    for i in range(n_paths):
        sampled = np.concatenate(
            [returns[s:s + block] for s in starts[i]]
        )[:horizon]
        paths[i] = sampled
    return paths


def monte_carlo_summary(
    daily_returns: pd.Series,
    n_paths: int = N_PATHS,
    horizon: int = HORIZON_DAYS,
    seed: int = SEED,
) -> dict[str, float]:
    """Terminal-wealth distribution of $1 invested for one year."""
    paths = block_bootstrap_returns(
        daily_returns.to_numpy(), n_paths=n_paths, horizon=horizon, seed=seed
    )
    terminal = np.prod(1 + paths, axis=1)
    running_min = np.minimum.accumulate(np.cumprod(1 + paths, axis=1), axis=1).min(axis=1)
    max_dd = running_min - 1  # worst intra-path drawdown from $1 start
    tail = terminal[terminal <= np.percentile(terminal, 5)]
    return {
        "paths": int(n_paths),
        "horizon_days": int(horizon),
        "terminal_p5": round(float(np.percentile(terminal, 5)), 3),
        "terminal_p50": round(float(np.percentile(terminal, 50)), 3),
        "terminal_p95": round(float(np.percentile(terminal, 95)), 3),
        "prob_loss_pct": round(100 * float((terminal < 1).mean()), 1),
        "prob_dd_gt_10pct": round(100 * float((max_dd < -0.10).mean()), 1),
        "expected_shortfall_5pct": round(float(tail.mean() - 1), 3),
        "median_max_drawdown_pct": round(100 * float(np.median(max_dd)), 1),
    }


def deterministic_scenarios(daily_returns: pd.Series) -> pd.DataFrame:
    """Bull/base/bear 1-year outlooks from shifted mean/vol assumptions."""
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    scenarios = {
        "bear": {"mu_shift": -0.0006, "vol_mult": 1.5},
        "base": {"mu_shift": 0.0, "vol_mult": 1.0},
        "bull": {"mu_shift": 0.0006, "vol_mult": 0.8},
    }
    rows = []
    for name, adj in scenarios.items():
        mu_s = mu + adj["mu_shift"]
        vol_s = sigma * adj["vol_mult"]
        exp_1y = (1 + mu_s) ** 252 - 1
        var_5 = mu_s * 252 - 1.645 * vol_s * np.sqrt(252)  # approx 1y 5% VaR
        rows.append(
            {
                "scenario": name,
                "daily_mean_bps": round(mu_s * 10_000, 1),
                "daily_vol_pct": round(vol_s * 100, 2),
                "expected_1y_return_pct": round(100 * exp_1y, 1),
                "approx_1y_var5_pct": round(100 * var_5, 1),
            }
        )
    return pd.DataFrame(rows)


def stress_tests(daily_returns: pd.Series) -> pd.DataFrame:
    """Single-day shocks applied to a $1m notional portfolio."""
    shocks = {
        "equity_-10pct_day": -0.10,
        "rates_+100bp_shock": -0.03,
        "vol_spike_2x": -0.02,
        "fx_gap_-5pct": -0.05,
    }
    return pd.DataFrame(
        [
            {"shock": name, "pnl_usd_on_1m": round(shock * 1_000_000)}
            for name, shock in shocks.items()
        ]
    )


if __name__ == "__main__":
    from backtest import positions_from_proba, run_backtest
    from data_pipeline import generate_synthetic_prices
    from feature_engineering import FEATURE_COLUMNS, build_model_frame, time_split
    from model import train_binary_models

    frame = build_model_frame(generate_synthetic_prices())
    train, test = time_split(frame)
    models = train_binary_models(train[FEATURE_COLUMNS], train["target_up"])
    proba = models["rf"].predict_proba(test[FEATURE_COLUMNS])[:, 1]
    daily = run_backtest(test, positions_from_proba(proba))
    print(deterministic_scenarios(daily["pnl"]).to_string(index=False))
    print(monte_carlo_summary(daily["pnl"]))
    print(stress_tests(daily["pnl"]).to_string(index=False))
