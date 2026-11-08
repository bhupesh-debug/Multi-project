"""Forecasting: next-quarter revenue & cost from synthetic monthly history.

Models compared on a time-based hold-out (train first 18 months, test last
6): seasonal naive, Holt linear-trend exponential smoothing, and a Ridge
regression on trend + month dummies. Best MAPE wins. Prediction intervals
come from hold-out residual standard deviation. All models are statistical
/ ML baselines — no claim of real predictive power on synthetic data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error


def _holt_linear(y: np.ndarray, alpha: float = 0.4, beta: float = 0.3, h: int = 3):
    level, trend = y[0], y[1] - y[0]
    fitted = [level]
    for t in range(1, len(y)):
        prev_level = level
        level = alpha * y[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        fitted.append(level + trend)
    fcst = [level + (i + 1) * trend for i in range(h)]
    return np.array(fitted), np.array(fcst)


def _ridge_forecast(y: np.ndarray, h: int = 3):
    n = len(y)
    t = np.arange(n).reshape(-1, 1)
    month = np.arange(n) % 12
    X = np.hstack([t, pd.get_dummies(month, drop_first=True).values.astype(float)])
    model = Ridge(alpha=10.0)
    model.fit(X, y)
    fitted = model.predict(X)
    t_f = np.arange(n, n + h).reshape(-1, 1)
    month_f = np.arange(n, n + h) % 12
    # Future month dummies must match training columns (months 1..11).
    X_f = np.hstack([t_f, np.array([[1.0 if m == k else 0.0 for k in range(1, 12)]
                                    for m in month_f])])
    return fitted, model.predict(X_f), model


def forecast_series(y: pd.Series, h: int = 3, test_months: int = 6) -> dict:
    """Compare 3 models on the last `test_months` as hold-out; refit the
    winner on the full series and forecast `h` months ahead."""
    values = y.values.astype(float)
    train, test = values[:-test_months], values[-test_months:]
    # Seasonal naive: same month last year. Test month j (global index
    # 18+j) is predicted by train month 6+j (12 months earlier).
    naive_pred = np.array([train[6 + j] for j in range(test_months)])
    _, holt_test = _holt_linear(train, h=test_months)
    ridge_fitted, ridge_test, _ = _ridge_forecast(train, h=test_months)
    preds = {"Seasonal naive": naive_pred, "Holt linear trend": holt_test,
             "Ridge (trend + seasonality)": ridge_test}
    mape = {k: mean_absolute_percentage_error(test, np.maximum(v, 1.0))
            for k, v in preds.items()}
    best = min(mape, key=mape.get)
    # Refit winner on full series for the forward forecast.
    if best == "Seasonal naive":
        fcst = np.array([values[-12 + j] for j in range(h)])
        resid = test - naive_pred
    elif best == "Holt linear trend":
        _, fcst = _holt_linear(values, h=h)
        resid = test - holt_test
    else:
        _, fcst, _ = _ridge_forecast(values, h=h)
        resid = test - ridge_test
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else float(np.std(values) * 0.05)
    return {"best_model": best, "mape": mape, "holdout_actual": test,
            "holdout_pred_best": preds[best], "forecast": fcst,
            "sigma": sigma, "lower": fcst - 1.96 * sigma, "upper": fcst + 1.96 * sigma}


def forecast_firm(firm: pd.DataFrame, h: int = 3) -> dict:
    """Forecast revenue and total cost; return per-series results + a
    3-month operating-profit outlook table."""
    out = {}
    for col in ["revenue_actual", "cost_actual"]:
        out[col] = forecast_series(firm[col], h=h)
    months = pd.date_range(firm["month"].max() + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    rev, cost = out["revenue_actual"], out["cost_actual"]
    outlook = pd.DataFrame({
        "month": months,
        "revenue_forecast": np.round(rev["forecast"], 0),
        "revenue_lower_95": np.round(rev["lower"], 0),
        "revenue_upper_95": np.round(rev["upper"], 0),
        "cost_forecast": np.round(cost["forecast"], 0),
        "cost_lower_95": np.round(cost["lower"], 0),
        "cost_upper_95": np.round(cost["upper"], 0),
    })
    outlook["operating_profit_forecast"] = outlook["revenue_forecast"] - outlook["cost_forecast"]
    out["outlook"] = outlook
    return out
