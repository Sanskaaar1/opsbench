"""Plot repeat-aware benchmark JSON emitted by benchmark.py."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


_SERIES_STYLES = {
    ("compile", "bf16"): ("o", "-", -0.18),
    ("compile", "fp32"): ("s", "-", -0.06),
    ("eager", "bf16"): ("^", "--", 0.06),
    ("eager", "fp32"): ("D", "--", 0.18),
}


def _load_results(paths: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        results.extend(item for item in payload["results"] if item["status"] == "ok")
    return results


def _metric_and_error(item: dict[str, Any], metric: str) -> tuple[float, float]:
    latency = float(item["median_ms"])
    latency_error = float(item["run_to_run_stddev_ms"])
    if metric == "latency_ms":
        return latency, latency_error
    value = float(item[metric])
    return value, value * latency_error / latency


def _shape_sort_key(item: dict[str, Any]) -> tuple[int, int, int]:
    shape = item["shape"]
    return int(shape["sequence"]), int(shape["hidden"]), int(shape["batch"])


def _series_style(mode: str, dtype: str) -> tuple[str, str, float]:
    return _SERIES_STYLES.get((mode, dtype), ("x", ":", 0.0))


def _plot_metric(results: list[dict[str, Any]], metric: str, ylabel: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    operators = sorted({item["operator"] for item in results})
    figure, axes = plt.subplots(1, len(operators), figsize=(5 * len(operators), 4), squeeze=False)
    for axis, operator in zip(axes[0], operators):
        operator_items = [item for item in results if item["operator"] == operator]
        workloads = list(
            dict.fromkeys(item["shape_label"] for item in sorted(operator_items, key=_shape_sort_key))
        )
        workload_positions = {label: index for index, label in enumerate(workloads)}
        series: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in operator_items:
            series[(item["mode"], item["dtype"])].append(item)
        for (mode, dtype), items in sorted(series.items()):
            items.sort(key=_shape_sort_key)
            values, errors = zip(*(_metric_and_error(item, metric) for item in items))
            marker, linestyle, offset = _series_style(mode, dtype)
            axis.errorbar(
                [workload_positions[item["shape_label"]] + offset for item in items],
                values,
                yerr=errors,
                marker=marker,
                linestyle=linestyle,
                capsize=3,
                label=f"{mode} / {dtype}",
            )
        axis.set_xticks(range(len(workloads)), workloads)
        axis.set_title(operator)
        axis.set_xlabel("workload")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=45)
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_results(paths: list[Path], output_dir: Path) -> None:
    """Create all plots for successful cases in benchmark result files."""
    results = _load_results(paths)
    if not results:
        raise ValueError("no successful benchmark results to plot")
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_metric(results, "latency_ms", "latency (ms)", output_dir / "latency.png")
    _plot_metric(results, "tflops", "throughput (TFLOPS)", output_dir / "tflops.png")
    _plot_metric(results, "bandwidth_gbps", "bandwidth (GB/s)", output_dir / "bandwidth.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    plot_results(args.inputs, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
