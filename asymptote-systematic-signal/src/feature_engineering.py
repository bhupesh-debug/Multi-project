"""Feature engineering: price series -> model frame.

One row = (date, asset). Features are trailing-only (no look-ahead):
momentum, volatility, trend, RSI, drawdown and cross-asset context.
Targets are forward 20-day returns of the same asset:

* ``target_up`` (binary, primary): 1 if the next-20-day return > 0.
* ``target_regime`` (3-class, secondary): bull / range / bear from the
  next-20-day return (> +2% / within +/-2% / < -2%).
* ``target_ret``: the raw forward 20-day return (for diagnostics).

Rows whose forward window runs past the end of the sample are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline import ASSETS

FORWARD_DAYS = 20
FEATURE_COLUMNS: list[str] = [
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "vol_20d",
    "vol_ratio",
    "ma_ratio",
    "rsi_14",
    "drawdown_60d",
    "trend_slope",
    "dist_252d_high",
    "breadth",
    "eq_minus_bd",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _trend_slope(log_price: pd.Series, window: int = 20) -> pd.Series:
    """OLS slope of log-price over the trailing window, scaled by vol."""
    x = np.arange(window, dtype=float)
    x = x - x.mean()

    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        return float(np.dot(x, y - y.mean()) / np.dot(x, x))

    return log_price.rolling(window).apply(_slope, raw=True)


def build_model_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the stacked (date, asset) modelling dataset."""
    df = prices.set_index("date").sort_index()
    log_px = np.log(df[ASSETS])

    # Market context: share of assets above their 60-day moving average.
    above_ma = pd.DataFrame(
        {a: (df[a] > df[a].rolling(60).mean()).astype(float) for a in ASSETS}
    )
    breadth = above_ma.mean(axis=1)

    frames = []
    for asset in ASSETS:
        close = df[asset]
        feats = pd.DataFrame(index=df.index)
        feats["ret_5d"] = close.pct_change(5)
        feats["ret_20d"] = close.pct_change(20)
        feats["ret_60d"] = close.pct_change(60)
        feats["vol_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)
        feats["vol_ratio"] = feats["vol_20d"] / (
            close.pct_change().rolling(60).std() * np.sqrt(252)
        )
        feats["ma_ratio"] = close / close.rolling(60).mean() - 1
        feats["rsi_14"] = _rsi(close)
        feats["drawdown_60d"] = close / close.rolling(60).max() - 1
        feats["trend_slope"] = _trend_slope(log_px[asset])
        feats["dist_252d_high"] = close / close.rolling(252).max() - 1
        feats["breadth"] = breadth
        feats["eq_minus_bd"] = df["SYN_EQ"].pct_change(20) - df["SYN_BD"].pct_change(20)
        feats["regime"] = df["regime"]
        feats["asset"] = asset
        feats["close"] = close
        # Forward targets (no look-ahead in features; targets use future prices).
        fwd = close.shift(-FORWARD_DAYS) / close - 1
        feats["target_ret"] = fwd
        feats["target_up"] = (fwd > 0).astype(int)
        feats["target_regime"] = pd.cut(
            fwd, bins=[-np.inf, -0.02, 0.02, np.inf], labels=["bear", "range", "bull"]
        )
        frames.append(feats)
    out = pd.concat(frames).reset_index(names="date")
    out = out.dropna(subset=FEATURE_COLUMNS + ["target_ret"]).reset_index(drop=True)
    return out


def time_split(
    frame: pd.DataFrame, train_end: str = "2023-12-31"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — no shuffling, no look-ahead across the cut."""
    train = frame[frame["date"] <= train_end].reset_index(drop=True)
    test = frame[frame["date"] > train_end].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Time split produced an empty side — check date range.")
    return train, test


if __name__ == "__main__":
    from data_pipeline import generate_synthetic_prices

    frame = build_model_frame(generate_synthetic_prices())
    print(frame.shape)
    print(frame[["date", "asset", "ret_20d", "rsi_14", "target_up"]].head())
    print("Class balance (target_up):")
    print(frame["target_up"].value_counts(normalize=True).round(3))
    print("Class balance (target_regime):")
    print(frame["target_regime"].value_counts(normalize=True).round(3))
