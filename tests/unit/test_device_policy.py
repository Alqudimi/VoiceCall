"""Unit tests: device-selection policy + environment detection basics."""
from __future__ import annotations

import torch

from rvc_platform.utils.device import DeviceProfile, select_device, training_dtype


class FakeProps:
    def __init__(self, total_memory: int) -> None:
        self.total_memory = total_memory


def _fake_gpu(monkeypatch, name: str, sm: tuple[int, int], vram_gb: float):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: sm)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: FakeProps(int(vram_gb * (1024 ** 3))))


def test_select_device_cpu_when_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    profile = select_device(None)
    assert profile.device == "cpu"
    assert profile.is_half is False


def test_select_device_small_vram_falls_back_to_cpu(monkeypatch):
    _fake_gpu(monkeypatch, "GT 710", (3, 5), 2.0)
    profile = select_device(None)
    assert profile.device == "cpu", "<4 GiB or SM<5.3 must fall back to CPU"


def test_select_device_low_compute_falls_back_to_cpu(monkeypatch):
    _fake_gpu(monkeypatch, "Quadro K620", (5, 0), 8.0)
    profile = select_device(None)
    assert profile.device == "cpu", "SM 5.0 < 5.3 must fall back to CPU"


def test_select_device_pascal_fp32(monkeypatch):
    _fake_gpu(monkeypatch, "GTX 1050 Ti", (6, 1), 6.0)
    profile = select_device(None)
    assert profile.device.startswith("cuda")
    assert profile.is_half is False, "Pascal (SM 6.1) must use fp32"


def test_select_device_16series_fp32(monkeypatch):
    _fake_gpu(monkeypatch, "GTX 1660", (7, 5), 8.0)
    profile = select_device(None)
    assert profile.device.startswith("cuda")
    assert profile.is_half is False, "GTX 16-series must use fp32"


def test_select_device_modern_gpu_fp32(monkeypatch):
    _fake_gpu(monkeypatch, "RTX 3060", (8, 6), 12.0)
    profile = select_device(None)
    assert profile.device.startswith("cuda")
    assert profile.is_half is False
    assert training_dtype(profile.device) == torch.float32


def test_split_params_budgets():
    # CPU: conservative budget
    cpu = DeviceProfile("cpu", False, None, 0.0, 0.0)
    assert cpu.split_params == (1, 6, 38, 41)
    # tiny CUDA card (<= 4 GiB): tightest budget
    small = DeviceProfile("cuda:0", False, "GT 1030", 3.5, 6.1)
    assert small.split_params == (1, 5, 30, 32)
    # modern fp16 card
    modern = DeviceProfile("cuda:0", True, "RTX 3060", 12.0, 8.6)
    assert modern.split_params == (3, 10, 60, 65)
    # fp32 CUDA card > 4 GiB
    pascal = DeviceProfile("cuda:0", False, "GTX 1080", 8.0, 6.1)
    assert pascal.split_params == (1, 6, 38, 41)


def test_training_dtype_cpu():
    assert training_dtype("cpu") == torch.float32


def test_detect_system_smoke():
    from rvc_platform.utils.environment import detect_system

    profile = detect_system()
    summary = profile.summary
    assert summary["os"].lower() in ("linux", "windows", "darwin")
    assert "arch" in summary
    assert "python" in summary
    assert "cuda" in summary          # None on CPU boxes, dict on CUDA ones
    assert "ram_mb" in summary
    assert "uv" in summary
    assert "gpus" in summary
