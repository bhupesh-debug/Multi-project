"""Tests: forecast outputs are well-formed; hold-out MAPE is finite."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation import generate_all
from src.forecast import forecast_firm, forecast_series


class TestForecast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.firm = generate_all()
        cls.out = forecast_firm(cls.firm)

    def test_outlook_shape(self):
        outlook = self.out["outlook"]
        self.assertEqual(len(outlook), 3)
        self.assertTrue((outlook["revenue_lower_95"] <= outlook["revenue_forecast"]).all())
        self.assertTrue((outlook["revenue_forecast"] <= outlook["revenue_upper_95"]).all())
        self.assertTrue(np.isfinite(outlook[["revenue_forecast", "cost_forecast"]].values).all())

    def test_mape_finite_and_reasonable(self):
        for col in ("revenue_actual", "cost_actual"):
            for model, mape in self.out[col]["mape"].items():
                self.assertTrue(np.isfinite(mape), f"{col}/{model}")
                self.assertLess(mape, 0.50, f"{col}/{model} MAPE implausibly high")

    def test_best_model_is_argmin(self):
        for col in ("revenue_actual", "cost_actual"):
            f = self.out[col]
            self.assertEqual(f["best_model"], min(f["mape"], key=f["mape"].get))

    def test_forecast_starts_after_history(self):
        outlook = self.out["outlook"]
        self.assertGreater(outlook["month"].min(), self.firm["month"].max())


if __name__ == "__main__":
    unittest.main()
