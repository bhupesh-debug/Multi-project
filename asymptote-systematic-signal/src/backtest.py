"""Backtesting: model signal vs rule baseline vs buy-and-hold.

Design (deliberately simple and auditable — this is what a hiring manager
at a systematic firm can actually check):
* One position per (date, asset), rebalanced daily from the model's
  predicted probability of a positive 20-day return.
* Position: +1 if P(up) >= 0.55, -1 if P(up) <= 0.45, else 0 (flat).
* Equal weight across the four synthetic assets.
* Transaction cost: 5 bps per unit change in position.
* Benchmarks: buy-and-hold (always +1) and the momentum rule baseline.

Performance is computed on SYNTHETIC data only. A good backtest here
proves the pipeline works — it is not evidence of a real trading edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model import rule_signal

COST_BPS = 5  # transaction cost per unit position change


def positions_from_proba(proba: np.ndarray, hi: float = 0.55, lo: float = 0.45) -> np.ndarray:
    return np.where(proba >= hi, 1, np.where(proba <= lo, -1, 0))


def _daily_asset_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Next-day return per (date, asset), aligned to the signal date."""
    out = frame[["date", "asset"]].copy()
    out["next_ret"] = frame.groupby("asset")["close"].pct_change().shift(-1)
    return out


def run_backtest(
    frame: pd.DataFrame,
    positions: np.ndarray,
    cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    """Return a daily portfolio series: date, portfolio return, equity."""
    work = frame[["date", "asset"]].copy()
    work["position"] = positions
    work["next_ret"] = frame.groupby("asset")["close"].pct_change().shift(-1)
    work["prev_position"] = work.groupby("asset")["position"].shift(1).fillna(0)
    work["turnover"] = (work["position"] - work["prev_position"]).abs()
    work["cost"] = work["turnover"] * cost_bps / 10_000
    work["pnl"] = work["position"] * work["next_ret"] - work["cost"]
    daily = work.groupby("date")["pnl"].mean().reset_index()
    daily = daily.dropna(subset=["pnl"]).reset_index(drop=True)
    daily["equity"] = (1 + daily["pnl"]).cumprod()
    return daily


def backtest_metrics(daily: pd.DataFrame) -> dict[str, float]:
    """Standard KPI set for a daily strategy series."""
    rets = daily["pnl"]
    equity = daily["equity"]
    n_days = len(daily)
    years = n_days / 252
    total_ret = equity.iloc[-1] - 1
    cagr = (equity.iloc[-1]) ** (1 / years) - 1 if years > 0 else float("nan")
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else float("nan")
    downside = rets[rets < 0]
    sortino = (
        rets.mean() / downside.std() * np.sqrt(252) if len(downside) > 1 else float("nan")
    )
    peak = equity.cummax()
    drawdown = equity / peak - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    win_rate = (rets > 0).mean()
    return {
        "days": int(n_days),
        "total_return_pct": round(100 * total_ret, 2),
        "cagr_pct": round(100 * cagr, 2),
        "ann_vol_pct": round(100 * vol, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown_pct": round(100 * max_dd, 2),
        "calmar": round(calmar, 2),
        "win_rate_pct": round(100 * win_rate, 1),
    }


def compare_strategies(
    frame_test: pd.DataFrame, model, feature_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Backtest ML signal, rule baseline and buy-and-hold on the test set."""
    proba = model.predict_proba(frame_test[feature_columns])[:, 1]
    strategies = {
        "ml_signal": positions_from_proba(proba),
        "rule_momentum": rule_signal(frame_test),
        "buy_and_hold": np.ones(len(frame_test), dtype=int),
    }
    curves, metrics = {}, {}
    for name, pos in strategies.items():
        daily = run_backtest(frame_test, pos)
        curves[name] = daily
        metrics[name] = backtest_metrics(daily)
    metrics_df = (
        pd.DataFrame(metrics)
        .T.reset_index(names="strategy")
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )
    return metrics_df, curves


if __name__ == "__main__":
    from data_pipeline import generate_synthetic_prices
    from feature_engineering import FEATURE_COLUMNS, build_model_frame, time_split
    from model import train_binary_models

    frame = build_model_frame(generate_synthetic_prices())
    train, test = time_split(frame)
    models = train_binary_models(train[FEATURE_COLUMNS], train["target_up"])
    metrics_df, _ = compare_strategies(test, models["rf"], FEATURE_COLUMNS)
    print(metrics_df.to_string(index=False))
