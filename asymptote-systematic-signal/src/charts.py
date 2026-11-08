"""Charts for the report (Step 5: Visual Intelligence).

Every chart is built from the synthetic pipeline outputs and saved as PNG
under ``reports/charts/``. Each chart answers a business question, not
just a technical one — see the captions in ``reports/model_report.md``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def chart_feature_importance(importances: pd.DataFrame, out: Path) -> Path:
    top = importances.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], color="#1f4e79")
    ax.set_xlabel("Random-forest importance")
    ax.set_title("Top 10 features — what drives the 20-day signal?")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_model_comparison(results: pd.DataFrame, out: Path) -> Path:
    plot = results.set_index("model")[["accuracy", "roc_auc", "f1"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    plot.plot(kind="bar", ax=ax, color=["#1f4e79", "#2e75b6", "#ed7d31"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Model comparison — test set (2024–2025, chronological split)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_equity_curves(curves: dict[str, pd.DataFrame], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, daily in curves.items():
        ax.plot(daily["date"], daily["equity"], label=name)
    ax.set_ylabel("Growth of $1 (synthetic backtest)")
    ax.set_title("Equity curves — ML signal vs momentum rule vs buy-and-hold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_drawdown(curves: dict[str, pd.DataFrame], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, daily in curves.items():
        dd = daily["equity"] / daily["equity"].cummax() - 1
        ax.plot(daily["date"], 100 * dd, label=name)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown — how much does each strategy give back?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_correlation(frame: pd.DataFrame, features: list[str], out: Path) -> Path:
    corr = frame[features].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(features, fontsize=8)
    ax.set_title("Feature correlation — what overlaps?")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_prices_regimes(prices: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(prices["date"], prices["SYN_EQ"], color="#1f4e79", lw=0.8)
    axes[0].set_ylabel("SYN_EQ price (synthetic)")
    axes[0].set_title("Synthetic equity proxy and market regimes (2019–2025)")
    regime_codes = prices["regime"].map({"range": 0, "bull": 1, "bear": 2})
    axes[1].fill_between(
        prices["date"], 0, 1, where=(regime_codes == 1), color="#70ad47", alpha=0.5, label="bull"
    )
    axes[1].fill_between(
        prices["date"], 0, 1, where=(regime_codes == 2), color="#c00000", alpha=0.5, label="bear"
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([])
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_monte_carlo(terminal: np.ndarray, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(terminal, bins=60, color="#2e75b6", alpha=0.85)
    ax.axvline(1.0, color="#c00000", lw=1.5, label="Break-even ($1)")
    ax.axvline(np.percentile(terminal, 5), color="#ed7d31", lw=1.5, linestyle="--", label="5th percentile")
    ax.set_xlabel("Terminal wealth of $1 after 1 year (2,000 paths)")
    ax.set_ylabel("Paths")
    ax.set_title("Monte Carlo — 1-year terminal wealth distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
