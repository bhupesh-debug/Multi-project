"""Tests for the Asymptote systematic-signal pipeline (stdlib unittest).

Run from the project root with the venv Python:
    .venv/bin/python -m unittest discover -s tests -v

All fixtures are synthetic and generated in-memory with the fixed seed.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest import (  # noqa: E402
    backtest_metrics,
    positions_from_proba,
    run_backtest,
)
from data_pipeline import (  # noqa: E402
    ASSETS,
    generate_synthetic_kpis,
    generate_synthetic_prices,
)
from feature_engineering import (  # noqa: E402
    FEATURE_COLUMNS,
    build_model_frame,
    time_split,
)
from model import rule_signal, train_binary_models  # noqa: E402
from scenario import (  # noqa: E402
    block_bootstrap_returns,
    deterministic_scenarios,
    monte_carlo_summary,
)


def _frame():
    return build_model_frame(generate_synthetic_prices())


class TestDataPipeline(unittest.TestCase):
    def test_prices_shape_and_columns(self):
        prices = generate_synthetic_prices()
        self.assertGreater(len(prices), 1500)
        for asset in ASSETS:
            self.assertIn(asset, prices.columns)
        self.assertTrue((prices["source"] == "synthetic").all())
        self.assertTrue((prices[ASSETS] > 0).all().all())

    def test_prices_are_business_days_and_sorted(self):
        prices = generate_synthetic_prices()
        self.assertTrue(prices["date"].is_monotonic_increasing)
        self.assertTrue(all(d.weekday() < 5 for d in prices["date"]))

    def test_regimes_are_valid_labels(self):
        prices = generate_synthetic_prices()
        self.assertTrue(set(prices["regime"]) <= {"bull", "range", "bear"})

    def test_reproducible_with_seed(self):
        a = generate_synthetic_prices(seed=42)
        b = generate_synthetic_prices(seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_kpis_are_synthetic_and_positive(self):
        kpis = generate_synthetic_kpis(generate_synthetic_prices())
        self.assertTrue((kpis["source"] == "synthetic").all())
        self.assertTrue((kpis["aum_usd_m"] > 0).all())
        self.assertTrue((kpis["revenue_usd_m"] > 0).all())
        self.assertTrue((kpis["headcount"] > 0).all())


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = _frame()

    def test_feature_columns_present_and_finite(self):
        for col in FEATURE_COLUMNS:
            self.assertIn(col, self.frame.columns)
        self.assertTrue(np.isfinite(self.frame[FEATURE_COLUMNS].to_numpy()).all())

    def test_rsi_bounded(self):
        self.assertTrue((self.frame["rsi_14"] >= 0).all())
        self.assertTrue((self.frame["rsi_14"] <= 100).all())

    def test_targets_cover_both_classes(self):
        self.assertEqual(set(self.frame["target_up"].unique()), {0, 1})
        self.assertTrue(set(self.frame["target_regime"].unique()) <= {"bull", "range", "bear"})

    def test_time_split_respects_chronology(self):
        train, test = time_split(self.frame)
        self.assertLessEqual(train["date"].max(), test["date"].min())
        self.assertEqual(len(train) + len(test), len(self.frame))

    def test_no_lookahead_in_features(self):
        # Features at date t must not use prices after t: ret_20d at the
        # first valid row is computed from trailing prices only, so the
        # frame must start only after the longest trailing window (252d).
        first_date = self.frame["date"].min()
        prices = generate_synthetic_prices()
        start = prices["date"].min()
        self.assertGreater((first_date - start).days, 200)


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = _frame()
        cls.train, cls.test = time_split(frame)
        cls.models = train_binary_models(
            cls.train[FEATURE_COLUMNS], cls.train["target_up"]
        )

    def test_probabilities_sum_to_one(self):
        for model in self.models.values():
            proba = model.predict_proba(self.test[FEATURE_COLUMNS])
            self.assertEqual(proba.shape, (len(self.test), 2))
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predictions_are_binary(self):
        for model in self.models.values():
            pred = model.predict(self.test[FEATURE_COLUMNS])
            self.assertTrue(set(np.unique(pred)) <= {0, 1})

    def test_rule_signal_is_pm_one(self):
        sig = rule_signal(self.test)
        self.assertTrue(set(np.unique(sig)) <= {-1, 1})


class TestBacktest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = _frame()
        _, cls.test = time_split(frame)
        cls.models = train_binary_models(
            time_split(frame)[0][FEATURE_COLUMNS],
            time_split(frame)[0]["target_up"],
        )
        proba = cls.models["rf"].predict_proba(cls.test[FEATURE_COLUMNS])[:, 1]
        cls.daily = run_backtest(cls.test, positions_from_proba(proba))

    def test_equity_starts_at_one_and_stays_positive(self):
        self.assertAlmostEqual(self.daily["equity"].iloc[0], 1 + self.daily["pnl"].iloc[0], places=9)
        self.assertTrue((self.daily["equity"] > 0).all())

    def test_metrics_are_finite_and_sane(self):
        m = backtest_metrics(self.daily)
        self.assertTrue(np.isfinite(m["sharpe"]))
        self.assertLessEqual(m["max_drawdown_pct"], 0)
        self.assertGreaterEqual(m["win_rate_pct"], 0)
        self.assertLessEqual(m["win_rate_pct"], 100)

    def test_positions_only_minus_one_zero_one(self):
        proba = self.models["rf"].predict_proba(self.test[FEATURE_COLUMNS])[:, 1]
        pos = positions_from_proba(proba)
        self.assertTrue(set(np.unique(pos)) <= {-1, 0, 1})


class TestScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = _frame()
        _, test = time_split(frame)
        models = train_binary_models(
            time_split(frame)[0][FEATURE_COLUMNS],
            time_split(frame)[0]["target_up"],
        )
        proba = models["rf"].predict_proba(test[FEATURE_COLUMNS])[:, 1]
        cls.daily = run_backtest(test, positions_from_proba(proba))

    def test_monte_carlo_percentiles_ordered(self):
        mc = monte_carlo_summary(self.daily["pnl"], n_paths=200)
        self.assertLess(mc["terminal_p5"], mc["terminal_p50"])
        self.assertLess(mc["terminal_p50"], mc["terminal_p95"])
        self.assertGreaterEqual(mc["prob_loss_pct"], 0)
        self.assertLessEqual(mc["prob_loss_pct"], 100)

    def test_block_bootstrap_shape(self):
        paths = block_bootstrap_returns(self.daily["pnl"].to_numpy(), n_paths=50, horizon=252)
        self.assertEqual(paths.shape, (50, 252))

    def test_deterministic_scenarios_ordered(self):
        scen = deterministic_scenarios(self.daily["pnl"])
        self.assertEqual(set(scen["scenario"]), {"bull", "base", "bear"})
        bull = scen.loc[scen["scenario"] == "bull", "expected_1y_return_pct"].iloc[0]
        bear = scen.loc[scen["scenario"] == "bear", "expected_1y_return_pct"].iloc[0]
        self.assertGreater(bull, bear)


if __name__ == "__main__":
    unittest.main()
