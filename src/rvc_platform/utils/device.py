"""Device selection policy.

Mirrors the capability rules validated by the official RVC repository
(``configs/config.py``), expressed as a pure function so it is unit-testable:

* no usable CUDA GPU  -> CPU + float32;
* < 4 GiB VRAM or compute capability < 5.3 -> CPU + float32;
* Pascal (SM 6.1) and GTX 16-series (SM 7.5) -> CUDA + float32;
* everything newer -> CUDA + float16.

Also derives the official RVC memory-aware pipeline split parameters
(``x_pad / x_query / x_center / x_max``) used by the file-conversion engine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceProfile:
    device: str          # "cpu" | "cuda:0" | ...
    is_half: bool        # fp16 inference eligible
    gpu_name: str | None
    gpu_memory_gb: float
    sm: float

    # official RVC pipeline split budget (seconds), configs/config.py
    # device_config(): chosen by is_half (not device type), with the <=4 GB
    # override applied last — same on CPU and CUDA.
    @property
    def split_params(self) -> tuple[int, int, int, int]:
        if self.gpu_memory_gb is not None and 0 < self.gpu_memory_gb <= 4:
            return 1, 5, 30, 32
        if self.is_half:            # 6G-VRAM budget
            return 3, 10, 60, 65
        return 1, 6, 38, 41         # 5G-VRAM budget


def select_device(index: int | None = None) -> DeviceProfile:
    if not torch.cuda.is_available():
        return DeviceProfile("cpu", False, None, 0.0, 0.0)
    count = torch.cuda.device_count()
    idx = index if index is not None and 0 <= index < count else 0
    try:
        props = torch.cuda.get_device_properties(idx)
        name = torch.cuda.get_device_name(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        mem_gb = props.total_memory / (1024 ** 3) + 0.4
    except Exception:
        return DeviceProfile("cpu", False, None, 0.0, 0.0)

    sm = major + minor / 10.0
    is_16_series = bool(re.search(r"16\d{2}", name)) and sm == 7.5
    if mem_gb < 4 or sm < 5.3:
        return DeviceProfile("cpu", False, name, mem_gb, sm)
    if sm == 6.1 or is_16_series:
        return DeviceProfile(f"cuda:{idx}", False, name, mem_gb, sm)
    return DeviceProfile(f"cuda:{idx}", False, name, mem_gb, sm)


def training_dtype(device: str | None = None) -> torch.dtype:
    """Shared training precision: full FP32 across all devices."""
    return torch.float32
