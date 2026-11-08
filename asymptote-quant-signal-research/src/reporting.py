"""Research memo + model report writers (Markdown, from measured results)."""
from __future__ import annotations

import pandas as pd


def write_memo(path: str, stats: pd.DataFrame, wf_summary: pd.DataFrame,
               survivors: list[str], n_signals: int, seed: int):
    top = stats.sort_values("ic_mean", ascending=False).head(5)
    lines = [
        "# Research Memo — Signal Discovery & Validation (Synthetic Data)",
        "",
        "**Author:** Bhupesh Chandra Dimri | **Firm context:** Quantitative Researcher Intern, "
        "Asymptote Capital (New York, NY) | **Date:** November 2026",
        "",
        "> ⚠️ **ALL DATA IS SYNTHETIC** (seed %d). No real market data, no real "
        "Asymptote data. This memo documents a *research process*, not a trading "
        "recommendation. Nothing here should be traded on." % seed,
        "",
        "## 1. Question",
        "Which of %d candidate quantitative signals show statistically robust, "
        "out-of-sample predictive power for 5-day forward returns — after "
        "controlling for multiple testing and data snooping — and which are "
        "just noise that looks good in-sample?" % n_signals,
        "",
        "## 2. Method (in one paragraph)",
        "Daily synthetic prices for 12 SYN_ assets (2020–2025, 1,512 days). "
        "17 signals in 5 families (momentum, mean reversion, volatility, trend, "
        "cross-sectional). Each signal scored daily, cross-sectionally z-scored. "
        "Validation: daily Spearman IC vs 5-day forward returns, Newey-West "
        "t-statistics (lag 5), Bonferroni / Holm / Benjamini-Hochberg corrections, "
        "deflated Sharpe (Bailey & Lopez de Prado), signal-correlation clustering, "
        "and 3-fold expanding-window walk-forward backtest with purge (5d) + "
        "embargo (5d). Long-short tercile portfolios net of 5 bps costs.",
        "",
        "## 3. Findings",
        "",
        "| Signal | Mean IC | NW t | p (raw) | p (BH) | Passes FDR 5%? |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in stats.sort_values("ic_mean", ascending=False).iterrows():
        lines.append(f"| {r['signal']} | {r['ic_mean']:.4f} | {r['ic_t']:.2f} | "
                     f"{r['p_raw']:.4g} | {r['p_bh']:.4g} | "
                     f"{'YES' if r['p_bh'] < 0.05 else 'no'} |")
    lines += [
        "",
        f"**Survivors after BH-FDR 5%: {', '.join(survivors) if survivors else 'NONE'}.**",
        "",
        "Top 5 by IC:",
    ]
    for _, r in top.iterrows():
        lines.append(f"- **{r['signal']}**: IC {r['ic_mean']:.4f}, NW t {r['ic_t']:.2f}, "
                     f"BH p {r['p_bh']:.4g}, long-short Sharpe {r['ls_sharpe']:.2f} "
                     f"(net of costs), deflated Sharpe {r['dsr']:.2f}.")
    lines += [
        "",
        "## 4. Walk-forward (out-of-sample) — the honest test",
        "",
        wf_summary.to_markdown(index=False),
        "",
        "Read the decay column first: a signal whose train IC collapses "
        "out-of-sample was fit to noise, however impressive its in-sample p-value.",
        "",
        "## 5. What I would NOT do",
        "- Do not trade the full 17-signal library, and do not trade any signal "
        "that fails FDR correction — most of the library is noise by construction.",
        "- Do not re-run the search with new signal variants until a survivor "
        "fails; each new variant is another trial and inflates the false-discovery rate.",
        "- Do not use the in-sample Sharpe as a sizing input; use the walk-forward "
        "test Sharpe, and even that on synthetic data.",
        "",
        "## 6. Next steps",
        "1. Re-test survivors on a fresh synthetic seed (out-of-sample *data*, "
        "not just out-of-sample *time*) — a true signal should survive a new seed.",
        "2. Add realistic frictions: borrow costs, capacity, market-impact proxy.",
        "3. Combine survivors only if their correlation is low (< 0.5); "
        "otherwise the 'combination' is the same bet twice.",
        "4. Pre-register the next signal family *before* looking at results.",
        "",
        "## 7. Limitations (stated openly)",
        "- Synthetic data with only two embedded true effects; real markets have "
        "regimes, jumps, and microstructure this generator lacks.",
        "- 12 assets is a narrow cross-section; IC estimates are noisy.",
        "- Cost model is a simple turnover × 5 bps proxy, not a fill simulator.",
        "- Walk-forward uses 3 folds only; a single split would be weaker still.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
