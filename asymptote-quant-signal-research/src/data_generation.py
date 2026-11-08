"""Synthetic market data generation.

ALL DATA IS SYNTHETIC. No real market data is used.
Seed: 2026 (deliberately different from sibling projects, which used 42).
Tickers are prefixed SYN_ so nothing can be mistaken for a real security.

Design (ground truth, known to the generator, NOT to the validator):
- 12 synthetic assets across 4 synthetic asset classes.
- Daily log returns driven by a common market factor + idiosyncratic noise.
- TWO weak true effects are embedded, everything else is noise:
    * 60-day momentum:  r_t += 0.035 * sigma * z(past-60d return)
    * 5-day short-term reversal: r_t -= 0.025 * sigma * z(past-5d return)
  Coefficients are small on purpose: a competent validator should find
  ICs of only ~0.02-0.05 for the true signals and ~0 for the rest.
  Any method that "discovers" a dozen strong signals on this data is
  overfitting / data-snooping — that is exactly what the validation
  pipeline is built to catch.
- Two injected regime shifts (2022-06 bear stress, 2024-03 vol spike)
  test whether signals survive out-of-sample, not just in-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 2026
N_DAYS = 1512  # ~6 years of trading days, 2020-01-02 .. 2025-12-31
START = "2020-01-02"
ASSETS = [
    ("SYN_EQ1", "Equity-like A", 1.00, 0.016),
    ("SYN_EQ2", "Equity-like B", 0.95, 0.017),
    ("SYN_EQ3", "Equity-like C", 0.80, 0.018),
    ("SYN_BD1", "Bond-like A", 0.25, 0.006),
    ("SYN_BD2", "Bond-like B", 0.20, 0.005),
    ("SYN_BD3", "Bond-like C", 0.30, 0.007),
    ("SYN_CMD1", "Commodity-like A", 0.45, 0.020),
    ("SYN_CMD2", "Commodity-like B", 0.40, 0.022),
    ("SYN_CMD3", "Commodity-like C", 0.50, 0.019),
    ("SYN_FX1", "FX/carry-like A", 0.15, 0.009),
    ("SYN_FX2", "FX/carry-like B", 0.10, 0.008),
    ("SYN_FX3", "FX/carry-like C", 0.20, 0.010),
]
MOM_COEF = 0.035  # true 60d momentum loading (in sigma units) — deliberately weak
REV_COEF = 0.025   # true 5d reversal loading (in sigma units) — deliberately weak


def generate_prices(seed: int = SEED, n_days: int = N_DAYS) -> pd.DataFrame:
    """Generate synthetic daily close prices. Returns DataFrame (dates x assets)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START, periods=n_days)
    tickers = [a[0] for a in ASSETS]
    betas = np.array([a[2] for a in ASSETS])
    sigmas = np.array([a[3] for a in ASSETS])

    # Market factor: mild AR(1), plus two stress episodes
    market = rng.normal(0, 0.011, n_days)
    for t in range(1, n_days):
        market[t] += 0.08 * market[t - 1]
    # stress: Jun 2022 bear (days ~620-680), Mar 2024 vol spike (~1060-1080)
    market[620:680] -= 0.004
    market[1060:1080] *= 2.2

    rets = np.zeros((n_days, len(tickers)))
    logret = np.zeros((n_days, len(tickers)))
    for t in range(n_days):
        idio = rng.normal(0, 1, len(tickers)) * sigmas
        r = betas * market[t] + idio
        if t >= 61:
            past60 = logret[t - 60:t].sum(axis=0)
            z60 = (past60 - past60.mean()) / (past60.std() + 1e-9)
            past5 = logret[t - 5:t].sum(axis=0)
            z5 = (past5 - past5.mean()) / (past5.std() + 1e-9)
            r = r + MOM_COEF * sigmas * np.clip(z60, -3, 3) \
                  - REV_COEF * sigmas * np.clip(z5, -3, 3)
        logret[t] = r
        rets[t] = r
    prices = 100.0 * np.exp(np.cumsum(logret, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


if __name__ == "__main__":
    df = generate_prices()
    print(df.shape)
    print(df.head())
    print(df.tail())
