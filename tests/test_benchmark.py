import unittest

from benchmark import _aggregate_runs


class BenchmarkTest(unittest.TestCase):
    def test_aggregate_uses_independent_run_medians(self) -> None:
        aggregate = _aggregate_runs(
            [
                {"median_ms": 1.0},
                {"median_ms": 2.0},
                {"median_ms": 3.0},
            ]
        )
        self.assertEqual(aggregate["repeat_count"], 3)
        self.assertEqual(aggregate["median_ms"], 2.0)
        self.assertAlmostEqual(aggregate["run_to_run_stddev_ms"], (2 / 3) ** 0.5)
        self.assertEqual(aggregate["run_to_run_iqr_ms"], 1.0)
