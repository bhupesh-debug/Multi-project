"""End-to-end pipeline: generate -> signals -> validate -> walk-forward -> charts -> memo.

Usage:  python run.py            (regenerates everything under data/, charts/, reports/, models/)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.backtest import walk_forward
from src.charts import (chart_corr_heatmap, chart_ic_bars, chart_ic_decay,
                        chart_ls_cumulative, chart_volcano, chart_walkforward)
from src.data_generation import SEED, generate_prices
from src.reporting import write_memo
from src.signals import build_signals, forward_returns, forward_returns_1d
from src.validation import (benjamini_hochberg, daily_ic, deflated_sharpe,
                            holm_bonferroni, long_short_returns, newey_west_t,
                            sharpe_stats, signal_correlation)

ROOT = Path(__file__).parent


def main():
    for d in ("data/synthetic", "data/processed", "charts", "reports", "models"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    prices = generate_prices(seed=SEED)
    prices.to_csv(ROOT / "data/synthetic/syn_prices.csv")
    signals = build_signals(prices)
    fwd5 = forward_returns(prices, 5)
    fwd1d = forward_returns_1d(prices)

    # --- Full-sample validation for every signal ---
    rows, ls_rets, wf_all, ic_series = [], {}, {}, {}
    for name, sig in signals.items():
        ic = daily_ic(sig, fwd5)
        ic_series[name] = ic
        t, p = newey_west_t(ic, lag=5)
        ls = long_short_returns(sig, fwd1d, cost_bps=5.0)
        ls_rets[name] = ls
        st = sharpe_stats(ls)
        rows.append({
            "signal": name, "ic_mean": ic.mean(), "ic_std": ic.std(),
            "ic_t": t, "p_raw": p, "n_ic_days": len(ic),
            "ls_sharpe": st["sharpe"], "ls_mean_bps": st["mean_bps"],
            "ls_vol_ann": st["vol_ann"],
        })
    stats = pd.DataFrame(rows)
    stats["p_bonf"] = stats["p_raw"].clip(lower=1e-300) * len(stats)
    stats["p_bonf"] = stats["p_bonf"].clip(upper=1.0)
    stats["p_holm"] = holm_bonferroni(stats["p_raw"].values)
    stats["p_bh"] = benjamini_hochberg(stats["p_raw"].values)
    stats["dsr"] = stats.apply(
        lambda r: deflated_sharpe(r["ls_sharpe"], int(r["n_ic_days"]), len(stats)), axis=1)
    stats = stats.sort_values("ic_mean", ascending=False)
    stats.to_csv(ROOT / "data/processed/signal_validation.csv", index=False)

    survivors = stats.loc[stats["p_bh"] < 0.05, "signal"].tolist()
    print(f"Signals tested: {len(stats)} | BH-FDR 5% survivors: {survivors}")

    # --- Walk-forward for the top 5 by |IC| + the two known-family leaders ---
    wf_names = list(dict.fromkeys(
        stats.reindex(stats["ic_mean"].abs().sort_values(ascending=False).index)
        ["signal"].head(5).tolist()))
    wf_rows = []
    for name in wf_names:
        wf = walk_forward(signals[name], fwd5, fwd1d, n_folds=3)
        wf["signal"] = name
        wf_all[name] = wf
        wf_rows.append(wf)
        print(f"\n{name} walk-forward:\n{wf.to_string(index=False)}")
    wf_df = pd.concat(wf_rows, ignore_index=True)
    wf_df.to_csv(ROOT / "data/processed/walk_forward.csv", index=False)

    # --- Correlation / deduplication ---
    corr = signal_correlation(signals)
    corr.to_csv(ROOT / "data/processed/signal_correlation.csv")

    # --- Charts ---
    chart_ic_bars(stats, str(ROOT / "charts/ic_bars.png"))
    chart_volcano(stats, str(ROOT / "charts/volcano.png"))
    chart_walkforward(wf_all, str(ROOT / "charts/walkforward.png"), wf_names[:3])
    chart_ls_cumulative({n: ls_rets[n] for n in wf_names[:3]},
                        str(ROOT / "charts/ls_cumulative.png"))
    chart_corr_heatmap(corr, str(ROOT / "charts/signal_correlation.png"))
    top_name = stats.iloc[0]["signal"]
    chart_ic_decay(ic_series[top_name], str(ROOT / "charts/ic_decay_top.png"), top_name)

    # --- Memo + model report ---
    wf_summary = wf_df[["signal", "fold", "ic_train", "ic_test", "ic_decay",
                         "ls_sharpe_test"]].round(4)
    write_memo(str(ROOT / "reports/research_memo.md"), stats, wf_summary,
               survivors, len(stats), SEED)

    with open(ROOT / "models/signal_stats.pkl", "wb") as f:
        pickle.dump({"stats": stats, "walk_forward": wf_df, "seed": SEED}, f)

    report = [
        "# Model Report — Quant Signal Research (Synthetic Data)",
        "",
        f"Seed: {SEED} | Assets: 12 SYN_ | Days: {len(prices)} | Signals tested: {len(stats)}",
        "",
        "## Full-sample validation (sorted by IC)",
        "",
        stats.round(4).to_markdown(index=False),
        "",
        f"## BH-FDR 5% survivors: {survivors if survivors else 'NONE'}",
        "",
        "## Walk-forward (3 folds, purged 5d + embargo 5d)",
        "",
        wf_df.round(4).to_markdown(index=False),
        "",
        "## Honest limits",
        "- All data synthetic (seed 2026); two weak true effects embedded by design.",
        "- 12-asset cross-section; IC estimates noisy; 3 walk-forward folds only.",
        "- Cost model is turnover x 5 bps, not a fill simulator.",
        "- No production deployment, no live trading, no real-market claims.",
        "",
    ]
    (ROOT / "reports/model_report.md").write_text("\n".join(report))
    print("\nDone. Outputs: data/processed/, charts/ (6 PNGs), reports/research_memo.md, reports/model_report.md")
    print(stats[["signal", "ic_mean", "ic_t", "p_raw", "p_bh", "ls_sharpe", "dsr"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
