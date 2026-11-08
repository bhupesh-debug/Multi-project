"""Charts for the management report (matplotlib, PNG output)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 11, "axes.labelsize": 9})


def bridge_chart(bridge: pd.DataFrame, month_label: str, path: Path) -> Path:
    """Waterfall: budget profit -> revenue var -> cost var -> actual profit."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["Budget\nprofit", "Revenue\nvariance", "Cost\nvariance", "Actual\nprofit"]
    cum = bridge["cumulative_usd"].values / 1e6
    amounts = bridge["amount_usd"].values / 1e6
    colors = ["#1f4e79", "#2e9e6b" if amounts[1] >= 0 else "#c0392b",
              "#2e9e6b" if amounts[2] >= 0 else "#c0392b", "#1f4e79"]
    x = np.arange(4)
    bottoms = [0, cum[0], cum[1], 0]
    heights = [cum[0], amounts[1], amounts[2], cum[3]]
    ax.bar(x, heights, bottom=bottoms, color=colors, width=0.6)
    for i in range(3):
        ax.plot([x[i] + 0.3, x[i + 1] - 0.3], [cum[i], cum[i]], "k--", lw=0.8)
    for i, (h, b) in enumerate(zip(heights, bottoms)):
        ax.text(x[i], b + h + (0.05 if h >= 0 else -0.12), f"${h:+.2f}m" if i in (1, 2) else f"${cum[i]:.2f}m",
                ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Operating profit ($m)")
    ax.set_ylim(0, max(cum) * 1.25)
    ax.set_title(f"Operating-profit bridge — {month_label} (synthetic data)")
    ax.axhline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def variance_bars(strat_var: pd.DataFrame, path: Path) -> Path:
    """Revenue variance % by strategy for the latest month."""
    latest = strat_var.loc[strat_var["month"] == strat_var["month"].max()]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2e9e6b" if v >= 0 else "#c0392b" for v in latest["revenue_variance_pct"]]
    ax.barh(latest["strategy"], latest["revenue_variance_pct"] * 100, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(10, color="#c0392b", ls="--", lw=0.8, label="Materiality ±10%")
    ax.axvline(-10, color="#c0392b", ls="--", lw=0.8)
    ax.set_xlabel("Revenue variance vs budget (%)")
    ax.set_title(f"Revenue variance by strategy — {latest['month'].iloc[0].strftime('%b %Y')} (synthetic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def kpi_trend(firm: pd.DataFrame, firm_var: pd.DataFrame, path: Path) -> Path:
    """Operating margin (budget vs actual) and AUM over 24 months."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax1.plot(firm["month"], firm_var["operating_margin_budget"] * 100, label="Budget margin", marker="o", ms=3)
    ax1.plot(firm["month"], firm_var["operating_margin_actual"] * 100, label="Actual margin", marker="s", ms=3)
    ax1.set_ylabel("Operating margin (%)")
    ax1.legend()
    ax1.set_title("Operating margin & AUM — 24 months (synthetic data)")
    ax2.plot(firm["month"], firm["aum_actual_m"], label="Actual AUM ($m)", marker="o", ms=3)
    ax2.plot(firm["month"], firm["aum_budget_m"], label="Budget AUM ($m)", ls="--")
    ax2.set_ylabel("AUM ($m)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def forecast_chart(firm: pd.DataFrame, outlook: pd.DataFrame, path: Path) -> Path:
    """Revenue history + 3-month forecast with 95% interval."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(firm["month"], firm["revenue_actual"] / 1e6, label="Actual revenue", marker="o", ms=3)
    ax.plot(firm["month"], firm["revenue_budget"] / 1e6, label="Budget", ls="--")
    ax.plot(outlook["month"], outlook["revenue_forecast"] / 1e6, label="Forecast", marker="D", ms=4, color="#c0392b")
    ax.fill_between(outlook["month"], outlook["revenue_lower_95"] / 1e6,
                    outlook["revenue_upper_95"] / 1e6, alpha=0.25, color="#c0392b", label="95% interval")
    ax.set_ylabel("Revenue ($m)")
    ax.set_title("Revenue: history, budget & 3-month forecast (synthetic data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
