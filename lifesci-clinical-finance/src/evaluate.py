"""evaluate.py — charts + metrics tables for the model report.

Writes PNG charts to reports/figures/ (all labelled SYNTHETIC) and
returns the metrics dicts used by run_all.py.
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150})
SYN = "SYNTHETIC DATA — illustrative only, not real clinical data"


def _save(fig, name: str):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    return str(FIG / name)


def plot_feature_importance(model, feature_names, top_n: int = 10):
    # Works for tree models; for pipelines, unwrap the final estimator.
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if not hasattr(est, "feature_importances_"):
        return None
    imp = pd.Series(est.feature_importances_, index=feature_names).sort_values().tail(top_n)
    fig, ax = plt.subplots(figsize=(9, 5))
    imp.plot(kind="barh", ax=ax, color="#1f4e79")
    ax.set_title(f"Top {top_n} features — rNPV model\n{SYN}", fontsize=10)
    ax.set_xlabel("Random Forest feature importance")
    return _save(fig, "feature_importance.png")


def plot_model_comparison(reg_results: dict):
    names = list(reg_results)
    r2 = [reg_results[n]["r2_test"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, r2, color=["#7f7f7f", "#5b9bd5", "#1f4e79", "#2e75b6"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R² (held-out test set)")
    ax.set_title(f"rNPV regression — model comparison\n{SYN}", fontsize=10)
    for i, v in enumerate(r2):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=12, ha="right")
    return _save(fig, "model_comparison.png")


def plot_rnpv_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df["rnpv_usd_m"].hist(ax=axes[0], bins=30, color="#1f4e79")
    axes[0].set_title("rNPV distribution ($M)")
    axes[0].set_xlabel("rNPV ($M)")
    df.groupby("phase")["rnpv_usd_m"].median().plot(kind="bar", ax=axes[1], color="#2e75b6")
    axes[1].set_title("Median rNPV by phase ($M)")
    plt.xticks(rotation=12, ha="right")
    fig.suptitle(SYN, fontsize=9)
    return _save(fig, "rnpv_distribution.png")


def plot_burn_variance(burn: pd.DataFrame):
    monthly = burn.groupby("month").agg(
        planned=("planned_spend_usd_m", "mean"),
        actual=("actual_spend_usd_m", "mean"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(monthly.index, monthly["planned"], label="Planned (mean)", color="#1f4e79")
    ax.plot(monthly.index, monthly["actual"], label="Actual — synthetic (mean)", color="#c00000")
    ax.set_xlabel("Month"); ax.set_ylabel("Spend ($M / month)")
    ax.set_title(f"Budget variance — planned vs actual spend\n{SYN}", fontsize=10)
    ax.legend()
    return _save(fig, "burn_variance.png")


def plot_monte_carlo(cost_p50, cost_p80, cost_p90, approved):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(["P50", "P80", "P90", "Approved budget"],
           [cost_p50, cost_p80, cost_p90, approved],
           color=["#5b9bd5", "#2e75b6", "#1f4e79", "#c00000"])
    ax.set_ylabel("Cost to complete ($M)")
    ax.set_title(f"Monte Carlo — cost-to-complete percentiles vs approved budget\n{SYN}", fontsize=10)
    return _save(fig, "monte_carlo.png")
