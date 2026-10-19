"""CUDA-event timing with cache flushing and sample statistics."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from typing import Any

import torch


CACHE_FLUSH_BYTES = 128 * 1024 * 1024


def make_cache_flush_buffer(device: torch.device) -> torch.Tensor:
    """Allocate a buffer larger than the target H200 MIG L2 cache."""
    return torch.empty(CACHE_FLUSH_BYTES, dtype=torch.uint8, device=device)


def _validate_inputs_pool(inputs_pool: Sequence[tuple[torch.Tensor, ...]]) -> torch.device:
    if not inputs_pool:
        raise ValueError("inputs_pool must not be empty")

    device: torch.device | None = None
    for inputs in inputs_pool:
        if not inputs:
            raise ValueError("each inputs_pool entry must contain CUDA tensors")
        for tensor in inputs:
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                raise ValueError("inputs_pool must contain only CUDA tensors")
            if device is None:
                device = tensor.device
            elif tensor.device != device:
                raise ValueError("all inputs_pool tensors must use one CUDA device")

    assert device is not None
    return device


def _percentile(sorted_values: list[float], percent: float) -> float:
    position = (len(sorted_values) - 1) * percent
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def summarize_samples(samples: Sequence[float]) -> dict[str, float]:
    """Return median, population standard deviation, and IQR for samples."""
    if not samples:
        raise ValueError("samples must not be empty")
    ordered = sorted(samples)
    return {
        "median_ms": float(statistics.median(samples)),
        "stddev_ms": float(statistics.pstdev(samples)),
        "iqr_ms": _percentile(ordered, 0.75) - _percentile(ordered, 0.25),
    }


def _flush_cache(cache_flush_buffer: torch.Tensor) -> None:
    cache_flush_buffer.add_(1)


def measure_latency(
    fn: Callable[..., Any],
    inputs_pool: Sequence[tuple[torch.Tensor, ...]],
    warmups: int,
    iters: int,
    cache_flush_buffer: torch.Tensor | None = None,
) -> dict[str, float | int | list[float]]:
    """Measure ``fn`` with CUDA events after flushing cache before each call."""
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if iters <= 0:
        raise ValueError("iters must be positive")

    device = _validate_inputs_pool(inputs_pool)
    if cache_flush_buffer is None:
        cache_flush_buffer = make_cache_flush_buffer(device)
    if not cache_flush_buffer.is_cuda or cache_flush_buffer.device != device:
        raise ValueError("cache_flush_buffer must be a CUDA tensor on the inputs device")

    with torch.cuda.device(device):
        for index in range(warmups):
            _flush_cache(cache_flush_buffer)
            result = fn(*inputs_pool[index % len(inputs_pool)])
        torch.cuda.synchronize(device)

        latencies_ms: list[float] = []
        for index in range(iters):
            _flush_cache(cache_flush_buffer)
            inputs = inputs_pool[index % len(inputs_pool)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn(*inputs)
            end.record()
            torch.cuda.synchronize(device)
            latencies_ms.append(start.elapsed_time(end))

    return {"latencies_ms": latencies_ms, **summarize_samples(latencies_ms), "iterations": iters}
