"""Walk-forward backtesting with purged splits + embargo.

Design (anti-leakage, the point of this project):
- Expanding-window walk-forward: 3 folds. Train = all data before the
  fold's test start, test = the next 252 days.
- PURGE: any training observation whose 5-day forward window overlaps
  the test period is dropped.
- EMBARGO: 5 trading days after each test period are excluded from the
  next training set.
- No signal is re-selected inside a fold on test data: the candidate set
  is fixed before walk-forward; per-fold we only re-estimate the IC
  (a diagnostic), we do not pick a new "best" signal per fold.
- Signal decay: IC is reported per fold and per half, so decay is visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import daily_ic, newey_west_t, sharpe_stats, long_short_returns


def purged_folds(index: pd.DatetimeIndex, n_folds: int = 3,
                 test_len: int = 252, embargo: int = 5,
                 horizon: int = 5) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Return [(train_idx, test_idx), ...] with purge + embargo applied."""
    folds = []
    n = len(index)
    # test blocks are the last n_folds*test_len days, contiguous
    test_start = n - n_folds * test_len
    for k in range(n_folds):
        ts = test_start + k * test_len
        te = ts + test_len
        test_idx = index[ts:te]
        # train ends horizon days before test start (purge overlap)
        # + embargo days are skipped at the start of the NEXT fold's train
        train_end = ts - horizon
        train_idx = index[:train_end]
        if k > 0:
            # embargo: drop the 5 days immediately before this test block
            # from training (they are adjacent to the previous test block)
            train_idx = train_idx[:-embargo] if len(train_idx) > embargo else train_idx[:0]
        folds.append((train_idx, test_idx))
    return folds


def walk_forward(signal: pd.DataFrame, fwd5: pd.DataFrame, fwd1d: pd.DataFrame,
                 n_folds: int = 3) -> pd.DataFrame:
    rows = []
    for k, (tr, te) in enumerate(purged_folds(signal.index, n_folds=n_folds)):
        ic_tr = daily_ic(signal.loc[tr], fwd5.loc[tr])
        ic_te = daily_ic(signal.loc[te], fwd5.loc[te])
        t_tr, _ = newey_west_t(ic_tr)
        t_te, _ = newey_west_t(ic_te)
        ls = long_short_returns(signal.loc[te], fwd1d.loc[te])
        st = sharpe_stats(ls)
        rows.append({
            "fold": k + 1,
            "train_days": len(tr), "test_days": len(te),
            "ic_train": ic_tr.mean(), "ic_train_t": t_tr,
            "ic_test": ic_te.mean(), "ic_test_t": t_te,
            "ic_decay": ic_tr.mean() - ic_te.mean(),
            "ls_sharpe_test": st["sharpe"],
            "ls_mean_bps_test": st["mean_bps"],
        })
    return pd.DataFrame(rows)
