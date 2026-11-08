"""Statistical validation: IC, Newey-West t-stats, multiple-testing controls,
deflated Sharpe (Bailey & Lopez de Prado), signal correlation / deduplication.

Everything here is computed on SYNTHETIC data. A "significant" result means
significant *on this synthetic sample* — it is evidence about the pipeline,
not about a real market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def daily_ic(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Spearman IC (signal at t vs forward return)."""
    ics = []
    for idx in signal.index:
        s, f = signal.loc[idx], fwd.loc[idx]
        ok = s.notna() & f.notna()
        if ok.sum() >= 5 and s[ok].std() > 0 and f[ok].std() > 0:
            ics.append(stats.spearmanr(s[ok], f[ok]).statistic)
        else:
            ics.append(np.nan)
    return pd.Series(ics, index=signal.index).dropna()


def newey_west_t(x: pd.Series, lag: int = 5) -> tuple[float, float]:
    """Mean / HAC standard error (Bartlett kernel, lag). Returns (t, p two-sided)."""
    x = x.dropna().values.astype(float)
    n = len(x)
    if n < 30:
        return 0.0, 1.0
    mean = x.mean()
    var = x.var(ddof=1) / n
    for h in range(1, lag + 1):
        w = 1 - h / (lag + 1)
        ac = np.cov(x[:-h], x[h:], ddof=1)[0, 1] if n - h > 2 else 0.0
        var += 2 * w * ac / n
    se = np.sqrt(max(var, 1e-18))
    t = mean / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(t)))
    return float(t), float(p)


def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        running = min(running, p[idx] * m / (rank + 1))
        adj[idx] = running
    return adj


def deflated_sharpe(sharpe: float, n_obs: int, n_trials: int,
                    sharpe_var: float = 0.0) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) as a probability.

    DSR = P(SR_true > E[max SR under the null across n_trials]).
    Computed as Phi((SR - E[max SR]) / sd(SR)) with
    sd(SR) = sqrt((1 + SR^2/2) / n_obs) (normal-returns approximation).
    Returns a value in [0, 1]; DSR > 0.95 is the usual bar for
    "unlikely to be just data mining". n_trials=1 reduces to the
    probabilistic Sharpe ratio vs 0.
    """
    if n_obs < 30 or n_trials < 1:
        return 0.0
    # Annualised Sharpe: T is in YEARS (n_obs/252), not days — using days
    # here would understate sd(SR) ~16x and fabricate significance.
    n_years = n_obs / 252.0
    if n_trials == 1:
        e_max = 0.0
    else:
        z_max = (1 - 0.5772156649) * stats.norm.ppf(1 - 1 / n_trials) \
            + 0.5772156649 * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        e_max = z_max / np.sqrt(n_years)
    sd_sr = np.sqrt((1 + 0.5 * sharpe ** 2) / n_years)
    return float(stats.norm.cdf((sharpe - e_max) / max(sd_sr, 1e-9)))


def long_short_returns(signal: pd.DataFrame, fwd1d: pd.DataFrame,
                       cost_bps: float = 5.0) -> pd.Series:
    """Daily long-top-tercile / short-bottom-tercile, 1-day hold, net of costs.

    Turnover is approximated as the daily change in tercile membership;
    cost = turnover * cost_bps. This is a deliberately simple, honest
    cost model — not a fill simulator.
    """
    rets = []
    prev_long = prev_short = None
    for idx in signal.index:
        s = signal.loc[idx]
        f = fwd1d.loc[idx]
        if s.isna().all() or f.isna().all():
            continue
        q = s.rank(pct=True)
        long_m = q >= 2 / 3
        short_m = q <= 1 / 3
        r = f[long_m].mean() - f[short_m].mean()
        if prev_long is not None:
            turn = (long_m != prev_long).mean() + (short_m != prev_short).mean()
            r -= turn * cost_bps / 10000
        prev_long, prev_short = long_m, short_m
        rets.append((idx, r))
    return pd.Series(dict(rets))


def sharpe_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return {"sharpe": 0.0, "mean_bps": 0.0, "vol_ann": 0.0, "n": len(r)}
    return {
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
        "mean_bps": float(r.mean() * 10000),
        "vol_ann": float(r.std() * np.sqrt(252)),
        "n": len(r),
    }


def signal_correlation(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Correlation of flattened (demeaned) signal scores — for deduplication."""
    cols = {}
    for name, df in signals.items():
        cols[name] = df.values.ravel()
    return pd.DataFrame(cols).corr()
