import unittest

from plot import _metric_and_error, _series_style, _shape_sort_key


class PlotTest(unittest.TestCase):
    def test_throughput_error_scales_with_run_to_run_latency_error(self) -> None:
        item = {"median_ms": 2.0, "run_to_run_stddev_ms": 0.2, "tflops": 10.0}
        self.assertEqual(_metric_and_error(item, "latency_ms"), (2.0, 0.2))
        self.assertEqual(_metric_and_error(item, "tflops"), (10.0, 1.0))

    def test_workloads_sort_by_sequence_length(self) -> None:
        item = {"shape": {"batch": 16, "sequence": 128, "hidden": 768}}
        self.assertEqual(_shape_sort_key(item), (128, 768, 16))

    def test_mode_dtype_series_have_distinct_markers_and_offsets(self) -> None:
        styles = {
            _series_style(mode, dtype)
            for mode in ("eager", "compile")
            for dtype in ("fp32", "bf16")
        }
        self.assertEqual(len(styles), 4)
        self.assertEqual(_series_style("compile", "bf16"), ("o", "-", -0.18))
