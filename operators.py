"""Operator descriptions for the benchmark's first B/S/H milestone."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


Shape = tuple[int, int, int]
InputFactory = Callable[[Shape, torch.dtype, torch.device], tuple[torch.Tensor, ...]]
FlopsFn = Callable[[Shape], float]
# shortcut: modeled payload excludes kernel temporaries, use profiler counters for physical DRAM traffic.
BytesFn = Callable[[Shape, torch.dtype], int]

MATMUL_SHAPES: list[Shape] = [
    # Batch sweep: fixed S=512, H=2048.
    (1, 512, 2048),
    (8, 512, 2048),
    (32, 512, 2048),
    # Sequence sweep: fixed B=8, H=2048. The anchor is shared above.
    (8, 128, 2048),
    (8, 4096, 2048),
    # Hidden-size sweep: fixed B=8, S=512. The anchor is shared above.
    (8, 512, 768),
    (8, 512, 8192),
]
LINEAR_GELU_SHAPES: list[Shape] = [
    (32, 128, 768),
    (16, 256, 1024),
    (8, 512, 2048),
    (2, 1024, 4096),
]
SOFTMAX_SHAPES: list[Shape] = [
    (64, 128, 768),
    (32, 256, 1024),
    (16, 512, 2048),
    (4, 1024, 4096),
    (2, 2048, 4096),
]


@dataclass(frozen=True)
class OperatorSpec:
    """A pure operator and the workload information needed to measure it."""

    name: str
    fn: Callable[..., torch.Tensor]
    shapes: list[Shape]
    input_factory: InputFactory
    flops_fn: FlopsFn
    bytes_fn: BytesFn


def workload_label(shape: Shape) -> str:
    """Return an unambiguous model-workload label for a B/S/H shape."""
    batch, sequence, hidden = shape
    return f"B{batch}_S{sequence}_H{hidden}"


def _element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def _matmul_inputs(
    shape: Shape, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, ...]:
    batch, sequence, hidden = shape
    return (
        torch.randn((batch, sequence, hidden), dtype=dtype, device=device),
        torch.randn((batch, hidden, hidden), dtype=dtype, device=device),
    )


def _matmul_flops(shape: Shape) -> float:
    batch, sequence, hidden = shape
    return float(2 * batch * sequence * hidden**2)


def _matmul_bytes(shape: Shape, dtype: torch.dtype) -> int:
    batch, sequence, hidden = shape
    elements = batch * sequence * hidden + batch * hidden**2 + batch * sequence * hidden
    return elements * _element_size(dtype)


def _softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def _softmax_inputs(
    shape: Shape, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return (torch.randn(shape, dtype=dtype, device=device),)


def _softmax_flops(shape: Shape) -> float:
    batch, sequence, hidden = shape
    return float(5 * batch * sequence * hidden)


def _softmax_bytes(shape: Shape, dtype: torch.dtype) -> int:
    batch, sequence, hidden = shape
    return 2 * batch * sequence * hidden * _element_size(dtype)


def _linear_gelu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.gelu(F.linear(x, weight))


def _linear_gelu_inputs(
    shape: Shape, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, ...]:
    batch, sequence, hidden = shape
    return (
        torch.randn(shape, dtype=dtype, device=device),
        torch.randn((hidden, hidden), dtype=dtype, device=device),
    )


def _linear_gelu_flops(shape: Shape) -> float:
    batch, sequence, hidden = shape
    elements = batch * sequence * hidden
    return float(2 * elements * hidden + 8 * elements)


def _linear_gelu_bytes(shape: Shape, dtype: torch.dtype) -> int:
    batch, sequence, hidden = shape
    elements = batch * sequence * hidden
    return (2 * elements + hidden**2) * _element_size(dtype)


def _layernorm(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],))


def _layernorm_inputs(
    shape: Shape, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, ...]:
    return (torch.randn(shape, dtype=dtype, device=device),)


def _layernorm_flops(shape: Shape) -> float:
    batch, sequence, hidden = shape
    return float(5 * batch * sequence * hidden)


def _layernorm_bytes(shape: Shape, dtype: torch.dtype) -> int:
    return 2 * shape[0] * shape[1] * shape[2] * _element_size(dtype)


MATMUL = OperatorSpec(
    name="matmul",
    fn=_matmul,
    shapes=MATMUL_SHAPES,
    input_factory=_matmul_inputs,
    flops_fn=_matmul_flops,
    bytes_fn=_matmul_bytes,
)
SOFTMAX = OperatorSpec(
    name="softmax",
    fn=_softmax,
    shapes=SOFTMAX_SHAPES,
    input_factory=_softmax_inputs,
    flops_fn=_softmax_flops,
    bytes_fn=_softmax_bytes,
)

LINEAR_GELU = OperatorSpec(
    name="linear_gelu",
    fn=_linear_gelu,
    shapes=LINEAR_GELU_SHAPES,
    input_factory=_linear_gelu_inputs,
    flops_fn=_linear_gelu_flops,
    bytes_fn=_linear_gelu_bytes,
)
LAYERNORM = OperatorSpec(
    name="layernorm",
    fn=_layernorm,
    shapes=SOFTMAX_SHAPES,
    input_factory=_layernorm_inputs,
    flops_fn=_layernorm_flops,
    bytes_fn=_layernorm_bytes,
)

OPERATORS = (MATMUL, SOFTMAX, LINEAR_GELU, LAYERNORM)
OPERATOR_SPECS = {spec.name: spec for spec in OPERATORS}


def arithmetic_intensity(spec: OperatorSpec, shape: Shape, dtype: torch.dtype) -> float:
    """Return modeled FLOPs per modeled payload byte for one operator case."""
    return spec.flops_fn(shape) / spec.bytes_fn(shape, dtype)
