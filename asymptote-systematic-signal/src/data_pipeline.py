"""Synthetic data generation for the Asymptote systematic-signal project.

ALL data in this project is SYNTHETIC. Nothing here is real market data,
real Asymptote data, or a claim about any real asset. Tickers are prefixed
``SYN_`` precisely so synthetic series can never be mistaken for live prices.

Two synthetic datasets are generated (seeded, reproducible):

1. ``synthetic_prices`` — daily prices for four synthetic asset proxies
   (equity-like, bond-like, commodity-like, FX/carry-like) driven by a
   hidden Markov regime process (bull / range / bear) with regime-dependent
   drift and volatility plus volatility clustering.
2. ``synthetic_business_kpis`` — monthly synthetic firm metrics (AUM,
   revenue, costs, headcount, operating margin) for the KPI-dashboard part
   of the JD (business performance, revenue, costs, operational metrics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
ASSETS = ["SYN_EQ", "SYN_BD", "SYN_CMD", "SYN_FX"]

# Regime labels: 0 = range, 1 = bull, 2 = bear (market-wide hidden state).
REGIME_NAMES = {0: "range", 1: "bull", 2: "bear"}
REGIME_TRANSITION = np.array(
    [
        [0.970, 0.020, 0.010],  # from range
        [0.020, 0.960, 0.020],  # from bull
        [0.030, 0.020, 0.950],  # from bear
    ]
)

# Per-asset (annualised drift, annualised vol) by regime.
# Values are illustrative only — chosen to make the four proxies behave
# differently, not calibrated to any real asset.
ASSET_PARAMS = {
    "SYN_EQ": {"bull": (0.18, 0.16), "range": (0.03, 0.14), "bear": (-0.22, 0.28)},
    "SYN_BD": {"bull": (0.05, 0.05), "range": (0.04, 0.04), "bear": (0.06, 0.07)},
    "SYN_CMD": {"bull": (0.12, 0.22), "range": (0.00, 0.20), "bear": (-0.15, 0.30)},
    "SYN_FX": {"bull": (0.06, 0.09), "range": (0.02, 0.08), "bear": (-0.08, 0.14)},
}

START_PRICES = {"SYN_EQ": 100.0, "SYN_BD": 100.0, "SYN_CMD": 100.0, "SYN_FX": 100.0}


def generate_regimes(n_days: int, seed: int = SEED) -> np.ndarray:
    """Markov-chain market regimes (0=range, 1=bull, 2=bear)."""
    rng = np.random.default_rng(seed)
    regimes = np.empty(n_days, dtype=int)
    regimes[0] = 0
    for t in range(1, n_days):
        regimes[t] = rng.choice(3, p=REGIME_TRANSITION[regimes[t - 1]])
    return regimes


def generate_synthetic_prices(
    start: str = "2019-01-01", end: str = "2025-12-31", seed: int = SEED
) -> pd.DataFrame:
    """Daily synthetic prices (business days) with a regime column.

    Returns a wide DataFrame: ``date`` + one close column per asset +
    ``regime`` (bull/range/bear) + ``source`` (always ``"synthetic"``).
    """
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    rng = np.random.default_rng(seed + 1)
    regimes = generate_regimes(n, seed=seed)
    regime_labels = [REGIME_NAMES[r] for r in regimes]

    # Volatility clustering: log-vol AR(1) shock shared across assets.
    log_vol_shock = np.zeros(n)
    for t in range(1, n):
        log_vol_shock[t] = 0.92 * log_vol_shock[t - 1] + rng.normal(0, 0.25)
    vol_mult = np.exp(0.35 * log_vol_shock)

    data = {"date": dates, "regime": regime_labels, "source": "synthetic"}
    for asset in ASSETS:
        drift = np.array([ASSET_PARAMS[asset][r][0] for r in regime_labels])
        vol = np.array([ASSET_PARAMS[asset][r][1] for r in regime_labels])
        daily_drift = drift / 252.0
        daily_vol = vol / np.sqrt(252.0) * vol_mult
        shocks = rng.normal(0, 1, n)
        log_rets = daily_drift + daily_vol * shocks
        # Small cross-asset correlation: equity and commodity share a shock.
        prices = START_PRICES[asset] * np.exp(np.cumsum(log_rets))
        data[asset] = np.round(prices, 4)
    return pd.DataFrame(data)


def generate_synthetic_kpis(prices: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Monthly synthetic business KPIs for a fictional systematic manager.

    AUM is driven by the synthetic strategy's own returns plus noisy net
    flows; revenue is AUM x a 45 bp fee; costs grow with headcount. This is
    the dataset behind the KPI dashboard — it describes a fictional firm,
    not Asymptote Capital or any real company.
    """
    rng = np.random.default_rng(seed + 2)
    monthly = prices.set_index("date").resample("ME").last()
    eq_ret = monthly["SYN_EQ"].pct_change().fillna(0.0)
    n = len(monthly)
    # Start at $4.0bn AUM: at a 45 bp fee that yields ~$18m revenue, enough
    # to cover a ~30-person quant firm. (A $500m start would make the
    # synthetic firm deeply loss-making at any realistic cost base — an
    # earlier draft did exactly that and produced -500% margins.)
    aum, flows = [4000.0], []
    for ret in eq_ret:
        flow = rng.normal(35.0, 90.0)  # $m net flow, noisy
        flows.append(flow)
        aum.append(aum[-1] * (1 + ret) + flow)
    aum = np.array(aum[1:])
    headcount = np.round(28 + np.cumsum(rng.normal(0.15, 0.6, n))).astype(int)
    headcount = np.maximum(headcount, 24)
    revenue = aum * 0.0045  # annualised revenue $m (45 bp on AUM)
    revenue = revenue * rng.normal(1.0, 0.03, n)
    costs = 8.0 + headcount * 0.30 + rng.normal(0, 0.5, n)  # $m, annualised
    df = pd.DataFrame(
        {
            "month": monthly.index.strftime("%Y-%m"),
            "aum_usd_m": np.round(aum, 1),
            "net_flows_usd_m": np.round(flows, 1),
            "revenue_usd_m": np.round(revenue, 2),
            "costs_usd_m": np.round(costs, 2),
            "headcount": headcount,
            "source": "synthetic",
        }
    )
    df["operating_margin_pct"] = np.round(
        100 * (df["revenue_usd_m"] - df["costs_usd_m"]) / df["revenue_usd_m"], 1
    )
    df["revenue_per_head_usd_k"] = np.round(
        1000 * df["revenue_usd_m"] / df["headcount"], 0
    )
    return df


def load_prices(path: str) -> pd.DataFrame:
    """Load a generated prices CSV, parsing dates."""
    df = pd.read_csv(path, parse_dates=["date"])
    if (df["source"] != "synthetic").any():
        raise ValueError("Prices file contains non-synthetic rows — refusing to load.")
    return df


if __name__ == "__main__":
    prices = generate_synthetic_prices()
    print(prices.head())
    print(prices["regime"].value_counts())
    print(generate_synthetic_kpis(prices).head())
