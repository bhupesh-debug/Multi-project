"""Candidate signal library — 17 signals across 5 families.

Each signal returns a DataFrame of scores (dates x SYN_ assets), aligned
with prices: signal at close t is tradeable from t+1 (no look-ahead).
Scores are cross-sectionally z-scored each day so families are comparable.

Families:
  MOM  momentum (5/20/60/120d, incl. 12-1 style 252-21)
  REV  mean reversion (5d, 20d, distance from MA)
  VOL  volatility / risk (20d vol, vol ratio, vol-of-vol proxy)
  TREND moving-average / trend (MA ratio, MACD-like, breakout)
  XSEC cross-sectional / misc (RSI-like, skewness, turnover proxy)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    m = df.mean(axis=1)
    s = df.std(axis=1)
    return df.sub(m, axis=0).div(s + 1e-9, axis=0)


def _rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    chg = prices.diff()
    up = chg.clip(lower=0).rolling(window).mean()
    dn = (-chg.clip(upper=0)).rolling(window).mean()
    rs = up / (dn + 1e-9)
    return 100 - 100 / (1 + rs)


def build_signals(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rets = np.log(prices / prices.shift(1))
    sig: dict[str, pd.DataFrame] = {}
    # --- Momentum ---
    for w in (5, 20, 60, 120):
        sig[f"MOM_{w}d"] = _zscore(prices.pct_change(w))
    sig["MOM_12m1"] = _zscore(prices.pct_change(252) - prices.pct_change(21))
    # --- Mean reversion ---
    sig["REV_5d"] = _zscore(-prices.pct_change(5))
    sig["REV_20d"] = _zscore(-prices.pct_change(20))
    ma60 = prices.rolling(60).mean()
    sig["REV_MA60"] = _zscore(-(prices / ma60 - 1))
    # --- Volatility ---
    vol20 = rets.rolling(20).std()
    vol60 = rets.rolling(60).std()
    sig["VOL_20d"] = _zscore(-vol20)          # low-vol tilt
    sig["VOL_ratio"] = _zscore(vol20 / (vol60 + 1e-9))
    sig["VOL_absret"] = _zscore(-rets.abs().rolling(20).mean())
    # --- Trend ---
    ma20 = prices.rolling(20).mean()
    sig["TREND_MA20"] = _zscore(prices / ma20 - 1)
    sig["TREND_MACD"] = _zscore(
        prices.ewm(span=12, adjust=False).mean()
        - prices.ewm(span=26, adjust=False).mean())
    hi252 = prices.rolling(252).max()
    sig["TREND_breakout"] = _zscore(prices / hi252 - 1)
    # --- Cross-sectional / misc ---
    sig["XSEC_RSI14"] = _zscore(-_rsi(prices, 14))   # oversold = long
    sig["XSEC_skew20"] = _zscore(rets.rolling(20).skew())
    sig["XSEC_ret1d"] = _zscore(-rets)                # 1-day reversal
    return {k: v.fillna(0.0) for k, v in sig.items()}


def forward_returns(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Forward h-day log return: fwd[t] = log(P[t+h] / P[t]) = sum of r[t+1..t+h].

    Signal at close t uses data up to t only, so this forward return is
    fully out-of-sample relative to the signal (no look-ahead).
    """
    return np.log(prices.shift(-horizon) / prices)


def forward_returns_1d(prices: pd.DataFrame) -> pd.DataFrame:
    rets = np.log(prices / prices.shift(1))
    return rets.shift(-1)
