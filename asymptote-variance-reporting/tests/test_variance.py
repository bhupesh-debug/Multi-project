"""Tests: variance decomposition must tie out exactly; flags behave."""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation import generate_all
from src.variance import (cost_variance_by_line, decompose_revenue_variance,
                          firm_variance_table, profit_bridge,
                          strategy_variance_table)


class TestVariance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strat, cls.firm = generate_all()
        cls.strat_var = strategy_variance_table(cls.strat)
        cls.firm_var = firm_variance_table(cls.firm)

    def test_revenue_bridge_ties_out(self):
        """AUM + rate + perf effects == total revenue variance, every row."""
        for _, r in self.strat.iterrows():
            dec = decompose_revenue_variance(r)
            self.assertAlmostEqual(dec["bridge_check"], 0.0, places=4)

    def test_profit_bridge_ties_out(self):
        for month in self.firm["month"]:
            b = profit_bridge(self.firm, month)
            # budget + revenue var - cost var == actual profit
            self.assertAlmostEqual(b["cumulative_usd"].iloc[-1],
                                   self.firm.loc[self.firm["month"] == month,
                                                 "operating_profit_actual"].iloc[0], places=0)

    def test_injected_credit_shock_is_material(self):
        row = self.strat_var[(self.strat_var["month"] == "2025-03-01")
                             & (self.strat_var["strategy"] == "SYN_CREDIT")].iloc[0]
        self.assertTrue(row["revenue_material"])
        self.assertLess(row["aum_variance_pct"], -0.10)

    def test_cost_lines_sum_to_total_cost_variance(self):
        month = pd.Timestamp("2025-07-01")
        lines = cost_variance_by_line(self.firm, month)
        total = lines["variance_usd"].sum()
        r = self.firm.loc[self.firm["month"] == month].iloc[0]
        # Cost lines cover opex only; strategy cost is separate — check opex part.
        opex_var = sum(r[f"cost_{k}_actual"] - r[f"cost_{k}_budget"] for k in
                       ["compensation", "technology", "market_data",
                        "professional_services", "other_opex"])
        self.assertAlmostEqual(total, opex_var, places=0)
        # July 2025 technology spike must be the largest adverse line.
        self.assertEqual(lines.iloc[0]["cost_line"], "Technology")

    def test_variance_sign_convention(self):
        # Revenue above budget is favourable; cost above budget is adverse.
        row = self.firm_var.iloc[0]
        self.assertEqual((row["revenue_variance_usd"] > 0),
                         (row["revenue_variance_pct"] > 0))


if __name__ == "__main__":
    unittest.main()
