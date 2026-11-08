"""Tests: data generation is deterministic and correctly shaped."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generation import generate_all


class TestData(unittest.TestCase):
    def test_deterministic(self):
        s1, f1 = generate_all(seed=42)
        s2, f2 = generate_all(seed=42)
        assert s1.equals(s2) and f1.equals(f2)

    def test_shapes(self):
        strat, firm = generate_all()
        self.assertEqual(len(strat), 96)   # 4 strategies x 24 months
        self.assertEqual(len(firm), 24)
        self.assertEqual(strat["strategy"].nunique(), 4)

    def test_all_synthetic_labels(self):
        strat, _ = generate_all()
        self.assertTrue(strat["strategy"].str.startswith("SYN_").all())
        self.assertTrue((strat["aum_actual_m"] > 0).all())
        self.assertTrue((strat["mgmt_fee_actual"] > 0).all())

    def test_no_missing_values(self):
        strat, firm = generate_all()
        self.assertFalse(strat.isna().any().any())
        self.assertFalse(firm.isna().any().any())


if __name__ == "__main__":
    unittest.main()
