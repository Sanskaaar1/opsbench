"""Run repeat-aware eager and compiled CUDA operator benchmarks."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from operators import OPERATOR_SPECS, OperatorSpec, Shape, arithmetic_intensity, workload_label
from plot import plot_results
from system_info import collect_system_info
from timing import CACHE_FLUSH_BYTES, make_cache_flush_buffer, measure_latency, summarize_samples


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, float | int]:
    medians = [float(run["median_ms"]) for run in runs]
    summary = summarize_samples(medians)
    return {
        "repeat_count": len(runs),
        "median_ms": summary["median_ms"],
        "run_to_run_stddev_ms": summary["stddev_ms"],
        "run_to_run_iqr_ms": summary["iqr_ms"],
    }


def _result_for_case(
    spec: OperatorSpec,
    mode: str,
    dtype_name: str,
    dtype: torch.dtype,
    shape: Shape,
    pool_size: int,
    warmups: int,
    iters: int,
    repeats: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    pool = [spec.input_factory(shape, dtype, device) for _ in range(pool_size)]
    cache_flush_buffer = make_cache_flush_buffer(device)
    fn = spec.fn if mode == "eager" else torch.compile(spec.fn)
    runs = []
    for repeat in range(repeats):
        timing = measure_latency(fn, pool, warmups, iters, cache_flush_buffer)
        runs.append({"repeat": repeat, **timing})

    flops = spec.flops_fn(shape)
    bytes_moved = spec.bytes_fn(shape, dtype)
    aggregate = _aggregate_runs(runs)
    median_s = float(aggregate["median_ms"]) / 1_000
    batch, sequence, hidden = shape
    return {
        "operator": spec.name,
        "shape": {"batch": batch, "sequence": sequence, "hidden": hidden},
        "shape_label": workload_label(shape),
        "mode": mode,
        "dtype": dtype_name,
        "status": "ok",
        "flops": flops,
        "bytes_moved": bytes_moved,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity(spec, shape, dtype),
        "tflops": flops / median_s / 1e12,
        "bandwidth_gbps": bytes_moved / median_s / 1e9,
        "repeat_runs": runs,
        **aggregate,
    }


def run_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this inside the reserved GPU allocation")

    requested = [OPERATOR_SPECS[name] for name in args.operators]
    started_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    for spec in requested:
        for mode in args.modes:
            for dtype_name in args.dtypes:
                dtype = DTYPES[dtype_name]
                for shape in spec.shapes:
                    try:
                        results.append(
                            _result_for_case(
                                spec,
                                mode,
                                dtype_name,
                                dtype,
                                shape,
                                args.pool_size,
                                args.warmups,
                                args.iters,
                                args.repeats,
                            )
                        )
                    except Exception as error:
                        results.append(
                            {
                                "operator": spec.name,
                                "shape": dict(zip(("batch", "sequence", "hidden"), shape)),
                                "shape_label": workload_label(shape),
                                "mode": mode,
                                "dtype": dtype_name,
                                "status": "error",
                                "error_type": type(error).__name__,
                                "message": str(error),
                            }
                        )
                    torch.cuda.synchronize()
                    time.sleep(0.05)

    return {
        "schema_version": 2,
        "started_at": started_at,
        "system_info": collect_system_info(),
        "configuration": {
            "operators": args.operators,
            "modes": args.modes,
            "dtypes": args.dtypes,
            "warmups": args.warmups,
            "iterations_per_run": args.iters,
            "repeats": args.repeats,
            "pool_size": args.pool_size,
            "cache_flush_bytes": CACHE_FLUSH_BYTES,
        },
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operators", nargs="+", choices=sorted(OPERATOR_SPECS), default=sorted(OPERATOR_SPECS)
    )
    parser.add_argument(
        "--modes", nargs="+", choices=("eager", "compile"), default=("eager", "compile")
    )
    parser.add_argument("--dtypes", nargs="+", choices=tuple(DTYPES), default=tuple(DTYPES))
    parser.add_argument("--warmups", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.iters <= 0 or args.repeats <= 0 or args.pool_size <= 0:
        raise SystemExit("--warmups must be >= 0; --iters, --repeats, and --pool-size must be > 0")
    payload = run_benchmarks(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"benchmark-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    if any(item["status"] == "ok" for item in payload["results"]):
        plot_results([output], args.output_dir / "plots")
    return 0 if all(item["status"] == "ok" for item in payload["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
