"""Tests: anomaly layer flags the injected shocks, not everything."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly import detect_anomalies
from src.data_generation import generate_all
from src.variance import firm_variance_table


class TestAnomaly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, firm = generate_all()
        cls.anom = detect_anomalies(firm_variance_table(firm), firm)

    def test_injected_shock_months_flagged(self):
        flagged = set(self.anom.loc[self.anom["verdict"] == "ANOMALY", "month"].dt.strftime("%Y-%m"))
        # Mar 2025 (credit redemption) and Jul 2025 (tech cost spike) must be caught.
        self.assertIn("2025-03", flagged)
        self.assertIn("2025-07", flagged)

    def test_most_months_normal(self):
        n_anom = (self.anom["verdict"] == "ANOMALY").sum()
        self.assertLessEqual(n_anom, 6, "Too many anomalies — detector is too eager")
        self.assertGreaterEqual(n_anom, 2, "Too few anomalies — injected shocks missed")

    def test_verdicts_valid(self):
        self.assertTrue(self.anom["verdict"].isin(["ANOMALY", "WATCH", "Normal"]).all())


if __name__ == "__main__":
    unittest.main()
