"""Unit tests — run:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest import purged_folds, walk_forward
from src.data_generation import SEED, generate_prices
from src.signals import build_signals, forward_returns
from src.validation import (benjamini_hochberg, daily_ic, deflated_sharpe,
                            holm_bonferroni, newey_west_t)


class TestData(unittest.TestCase):
    def test_deterministic_seed(self):
        a = generate_prices(seed=SEED)
        b = generate_prices(seed=SEED)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(a.shape, (1512, 12))
        self.assertTrue(all(c.startswith("SYN_") for c in a.columns))

    def test_different_seed_differs(self):
        self.assertFalse(generate_prices(seed=SEED).equals(generate_prices(seed=1)))


class TestSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = generate_prices(seed=SEED)
        cls.signals = build_signals(cls.prices)

    def test_library_size(self):
        self.assertEqual(len(self.signals), 17)

    def test_no_lookahead_in_forward(self):
        # forward return at t must not use prices after t+h
        fwd = forward_returns(self.prices, 5)
        self.assertTrue(np.allclose(
            fwd.iloc[100].values,
            np.log(self.prices.iloc[105].values / self.prices.iloc[100].values)))

    def test_mom5_and_rev5_are_opposites(self):
        # REV_5d = -MOM_5d by construction
        self.assertTrue(np.allclose(
            self.signals["REV_5d"].values, -self.signals["MOM_5d"].values,
            atol=1e-9, equal_nan=True) or
            np.corrcoef(self.signals["REV_5d"].values.ravel(),
                        self.signals["MOM_5d"].values.ravel())[0, 1] < -0.99)


class TestValidation(unittest.TestCase):
    def test_bh_controls_fdr(self):
        # 100 uniform p-values: BH at 5% should reject ~5% (allow wide band)
        rng = np.random.default_rng(0)
        adj = benjamini_hochberg(rng.uniform(size=1000))
        self.assertLess((adj < 0.05).mean(), 0.15)

    def test_holm_more_conservative_than_raw(self):
        p = np.array([0.001, 0.01, 0.04, 0.5])
        adj = holm_bonferroni(p)
        self.assertTrue(np.all(adj >= p))
        self.assertTrue(np.all(adj <= 1.0))

    def test_newey_west_on_noise(self):
        rng = np.random.default_rng(0)
        t, p = newey_west_t(pd.Series(rng.normal(size=500)), lag=5)
        self.assertLess(abs(t), 3.0)
        self.assertGreater(p, 0.01)

    def test_dsr_bounds(self):
        for sr in (-2.0, 0.0, 0.94, 2.0):
            d = deflated_sharpe(sr, 1250, 17)
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 1.0)
        # a strong Sharpe must beat a zero Sharpe after deflation
        self.assertGreater(deflated_sharpe(2.0, 1250, 17),
                           deflated_sharpe(0.0, 1250, 17))
        # more trials must lower DSR for the same Sharpe
        self.assertGreater(deflated_sharpe(0.94, 1250, 1),
                           deflated_sharpe(0.94, 1250, 17))


class TestBacktest(unittest.TestCase):
    def test_purged_folds_no_overlap(self):
        prices = generate_prices(seed=SEED)
        idx = prices.index
        for tr, te in purged_folds(idx, n_folds=3):
            # last train day's 5d forward window must end before test starts:
            # gap >= 6 trading days (5d purge + 1), 11 for later folds (+5d embargo)
            gap = idx.get_loc(te[0]) - idx.get_loc(tr[-1])
            self.assertGreaterEqual(gap, 6)
            self.assertEqual(len(te), 252)
            self.assertTrue(len(tr) > 500)

    def test_walk_forward_shape(self):
        prices = generate_prices(seed=SEED)
        signals = build_signals(prices)
        fwd5 = forward_returns(prices, 5)
        from src.signals import forward_returns_1d
        wf = walk_forward(signals["MOM_60d"], fwd5, forward_returns_1d(prices))
        self.assertEqual(len(wf), 3)
        self.assertIn("ic_test", wf.columns)


if __name__ == "__main__":
    unittest.main()
