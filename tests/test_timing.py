import unittest

import torch

from timing import _flush_cache, _percentile, measure_latency, summarize_samples


class TimingTest(unittest.TestCase):
    def test_percentile_and_summary_statistics(self) -> None:
        self.assertEqual(_percentile([1.0, 2.0, 3.0, 4.0], 0.25), 1.75)
        self.assertEqual(
            summarize_samples([1.0, 2.0, 3.0]),
            {"median_ms": 2.0, "stddev_ms": (2 / 3) ** 0.5, "iqr_ms": 1.0},
        )

    def test_cache_flush_writes_the_buffer(self) -> None:
        buffer = torch.zeros(4, dtype=torch.uint8)
        _flush_cache(buffer)
        self.assertTrue(torch.equal(buffer, torch.ones(4, dtype=torch.uint8)))

    def test_measurement_rejects_invalid_bounds_before_cuda_use(self) -> None:
        with self.assertRaisesRegex(ValueError, "warmups"):
            measure_latency(lambda: None, [], -1, 1)
        with self.assertRaisesRegex(ValueError, "iters"):
            measure_latency(lambda: None, [], 0, 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            measure_latency(lambda: None, [], 0, 1)
