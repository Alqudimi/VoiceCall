"""Automatic system / environment detection.

Produces a single :class:`SystemProfile` describing OS, architecture,
Python, uv availability, CUDA toolchain, GPUs, RAM/disk, execution context
(Colab / Drive) — everything the installers, the CLI doctor and the runtime
device policy need. No assumption is hard-coded: every capability is probed.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from rvc_platform import __version__


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_mb: int
    compute_capability: tuple[int, int] | None = None
    driver: str | None = None


@dataclass
class SystemProfile:
    os: str
    os_release: str
    arch: str
    python_version: str
    python_executable: str
    uv_version: str | None
    platform_version: str | None
    is_colab: bool
    is_notebook: bool
    google_drive_mounted: bool
    google_drive_path: Path | None
    cuda_available: bool
    cuda_version: str | None
    cudnn_version: str | None
    nvidia_smi: str | None
    gpus: list[GPUInfo] = field(default_factory=list)
    ram_mb: int = 0
    disk_free_gb: float = 0.0
    ffmpeg_available: bool = False
    cloudflared_available: bool = False
    make_available: bool = False
    platform_version_str: str = ""
    app_version: str = __version__

    @property
    def summary(self) -> dict:
        return {
            "os": self.os,
            "os_release": self.os_release,
            "arch": self.arch,
            "python": self.python_version,
            "uv": self.uv_version,
            "colab": self.is_colab,
            "drive_mounted": self.google_drive_mounted,
            "cuda": self.cuda_version,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "memory_mb": g.total_memory_mb,
                    "compute_capability": list(g.compute_capability or ()),
                }
                for g in self.gpus
            ],
            "ram_mb": self.ram_mb,
            "disk_free_gb": round(self.disk_free_gb, 2),
            "ffmpeg": self.ffmpeg_available,
            "cloudflared": self.cloudflared_available,
            "make": self.make_available,
        }


def _probe_gpu_torch() -> tuple[bool, list[GPUInfo], str | None, str | None]:
    """Probe CUDA through torch — the same runtime that will use the GPU."""
    try:
        import torch
    except Exception:
        return False, [], None, None
    if not torch.cuda.is_available():
        return False, [], None, None
    gpus: list[GPUInfo] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        gpus.append(
            GPUInfo(
                index=idx,
                name=torch.cuda.get_device_name(idx),
                total_memory_mb=int(props.total_memory // (1024 * 1024)),
                compute_capability=(props.major, props.minor),
                driver=getattr(props, "driver_version", None),
            )
        )
    return True, gpus, torch.version.cuda, torch.backends.cudnn.version()


def _probe_gpu_smi() -> tuple[bool, str | None, list[GPUInfo]]:
    """Probe with nvidia-smi (works even when torch is CPU-only build)."""
    if shutil.which("nvidia-smi") is None:
        return False, None, []
    code, out = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                      "--format=csv,noheader,nounits"])
    if code != 0:
        return False, None, []
    gpus: list[GPUInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append(GPUInfo(index=int(parts[0]), name=parts[1],
                                    total_memory_mb=int(float(parts[2])),
                                    driver=parts[3] if len(parts) > 3 else None))
            except ValueError:
                continue
    return True, out.strip()[:500], gpus


def detect_system() -> SystemProfile:
    is_colab = bool(os.environ.get("COLAB_GPU")) or Path("/usr/lib/python3/dist-packages/google").is_dir() or "google.colab" in sys_modules()
    is_notebook = "ipykernel" in sys_modules()
    drive_path = Path("/content/drive/MyDrive")
    drive_mounted = drive_path.is_dir() and any(drive_path.iterdir())

    torch_cuda, torch_gpus, cuda_ver, cudnn_ver = _probe_gpu_torch()
    smi_ok, smi_out, smi_gpus = _probe_gpu_smi()
    gpus = torch_gpus if torch_gpus else smi_gpus
    cuda_version = cuda_ver
    if not cuda_version and smi_ok:
        code, v = _run(["nvidia-smi"], timeout=5)
        for line in (v or "").splitlines():
            if "CUDA Version" in line:
                # header row: "| NVIDIA-SMI ... Driver Version: 550.163 CUDA
                # Version: 13.0 |" — extract the bare number, not ": 13.0".
                match = re.search(r"CUDA Version:\s*([0-9][0-9.]*)", line)
                if match:
                    cuda_version = match.group(1)
                    break

    try:
        import psutil  # type: ignore
        ram_mb = int(psutil.virtual_memory().total // (1024 * 1024))
        disk_free_gb = psutil.disk_usage(str(Path.home())).free / (1024 ** 3)
    except Exception:
        ram_mb, disk_free_gb = 0, 0.0

    uv_version = None
    uv_bin = shutil.which("uv")
    if uv_bin:
        code, out = _run([uv_bin, "--version"])
        uv_version = out.strip().split(" ")[-1] if code == 0 else uv_bin

    return SystemProfile(
        os=platform.system() or "Unknown",
        os_release=platform.release() or "",
        arch=platform.machine() or "unknown",
        python_version=platform.python_version(),
        python_executable=sys_executable(),
        uv_version=uv_version,
        platform_version=platform.version(),
        is_colab=is_colab,
        is_notebook=is_notebook,
        google_drive_mounted=drive_mounted,
        google_drive_path=drive_path if drive_mounted else None,
        cuda_available=torch_cuda or smi_ok,
        cuda_version=cuda_version,
        cudnn_version=str(cudnn_ver) if cudnn_ver else None,
        nvidia_smi=smi_out,
        gpus=gpus,
        ram_mb=ram_mb,
        disk_free_gb=disk_free_gb,
        ffmpeg_available=shutil.which("ffmpeg") is not None,
        cloudflared_available=shutil.which("cloudflared") is not None,
        make_available=shutil.which("make") is not None,
    )


def sys_modules() -> list[str]:
    import sys
    return list(sys.modules)


def sys_executable() -> str:
    import sys
    return sys.executable
