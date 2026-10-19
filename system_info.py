"""Collect a JSON-safe snapshot of the CUDA and PyTorch environment."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any


def _json_safe(value: Any) -> Any:
    """Convert values returned by optional system libraries to JSON values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _device_properties(properties: Any) -> dict[str, object]:
    fields = (
        "name",
        "major",
        "minor",
        "total_memory",
        "multi_processor_count",
        "pci_bus_id",
        "uuid",
    )
    return {
        field: _json_safe(getattr(properties, field))
        for field in fields
        if hasattr(properties, field)
    }


def collect_system_info() -> dict[str, object]:
    """Return environment metadata without requiring a working CUDA/NVML stack."""
    info: dict[str, object] = {
        "gpus": [],
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    cuda_device_count = 0

    try:
        import torch

        info.update(
            {
                "pytorch_version": _json_safe(torch.__version__),
                "pytorch_cuda_version": _json_safe(torch.version.cuda),
                "pytorch_build_config": _json_safe(torch.__config__.show()),
            }
        )
        try:
            cuda_available = torch.cuda.is_available()
            info["cuda_available"] = cuda_available
            if cuda_available:
                cuda_device_count = torch.cuda.device_count()
                info["cuda_device_count"] = cuda_device_count
                info["cuda_current_device"] = torch.cuda.current_device()
                info["gpus"] = [
                    _device_properties(torch.cuda.get_device_properties(index))
                    for index in range(cuda_device_count)
                ]
            else:
                info["cuda_device_count"] = 0
        except Exception as exc:
            info["cuda_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"

    nvml_errors: list[str] = []
    initialized = False
    try:
        import pynvml

        pynvml.nvmlInit()
        initialized = True
        info["nvidia_driver_version"] = _json_safe(pynvml.nvmlSystemGetDriverVersion())
        nvml_count = pynvml.nvmlDeviceGetCount()
        if cuda_device_count == 0:
            cuda_device_count = nvml_count
            info["cuda_device_count"] = cuda_device_count
            info["gpus"] = [{} for _ in range(cuda_device_count)]

        gpus = info["gpus"]
        assert isinstance(gpus, list)
        for index in range(min(cuda_device_count, nvml_count)):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                gpu = gpus[index]
                assert isinstance(gpu, dict)
                gpu["nvml_name"] = _json_safe(pynvml.nvmlDeviceGetName(handle))
                gpu["nvml_uuid"] = _json_safe(pynvml.nvmlDeviceGetUUID(handle))
                try:
                    current, pending = pynvml.nvmlDeviceGetMigMode(handle)
                    gpu["mig_mode"] = {"current": int(current), "pending": int(pending)}
                except Exception as exc:
                    gpu["mig_error"] = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                nvml_errors.append(f"device {index}: {type(exc).__name__}: {exc}")
        if cuda_device_count > nvml_count:
            nvml_errors.append(
                f"NVML reports {nvml_count} devices, CUDA exposes {cuda_device_count}"
            )
    except Exception as exc:
        nvml_errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception as exc:
                nvml_errors.append(f"shutdown: {type(exc).__name__}: {exc}")

    if nvml_errors:
        info["nvml_error"] = "; ".join(nvml_errors)
    return info
