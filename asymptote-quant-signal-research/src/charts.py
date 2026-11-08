"""Charts — all from measured run output, saved as PNGs under charts/."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def chart_ic_bars(stats: pd.DataFrame, path: str):
    d = stats.sort_values("ic_mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2e7d32" if p < 0.05 else "#9e9e9e" for p in d["p_bh"]]
    ax.barh(d["signal"], d["ic_mean"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Mean daily cross-sectional IC (5-day forward return)")
    ax.set_title("Signal ICs — green = survives Benjamini-Hochberg FDR 5%")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_volcano(stats: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(stats["ic_mean"], -np.log10(stats["p_raw"].clip(lower=1e-12)),
               s=70, color="#1565c0")
    for _, r in stats.iterrows():
        ax.annotate(r["signal"], (r["ic_mean"], -np.log10(max(r["p_raw"], 1e-12))),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(-np.log10(0.05), color="r", ls="--", lw=1, label="p = 0.05 (raw)")
    ax.set_xlabel("Mean IC"); ax.set_ylabel("-log10(p-value, Newey-West)")
    ax.set_title("Volcano plot — IC vs statistical significance")
    ax.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_walkforward(wf: dict[str, pd.DataFrame], path: str, top: list[str]):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in top:
        d = wf[name]
        ax.plot(d["fold"], d["ic_test"], marker="o", label=f"{name} test IC")
        ax.plot(d["fold"], d["ic_train"], marker="x", ls="--", label=f"{name} train IC")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Walk-forward fold"); ax.set_ylabel("Mean IC")
    ax.set_title("Walk-forward IC — train vs test (decay = overfitting signal)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_ls_cumulative(ls_rets: dict[str, pd.Series], path: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in ls_rets.items():
        ax.plot((1 + r.fillna(0)).cumprod(), label=name)
    ax.set_ylabel("Cumulative wealth ($1 start, net of 5bps costs)")
    ax.set_title("Long-short (top/bottom tercile) cumulative wealth — full sample")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_corr_heatmap(corr: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Signal-score correlation — clusters = redundant signals")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def chart_ic_decay(ic_series: pd.Series, path: str, name: str):
    # 126-day rolling mean IC
    roll = ic_series.rolling(126).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(roll.index, roll.values, color="#1565c0")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("126-day rolling mean IC")
    ax.set_title(f"Signal decay — {name} rolling IC (does the edge persist?)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
