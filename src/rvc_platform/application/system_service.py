"""System status aggregation for the dashboard & API."""
from __future__ import annotations

import time
from typing import Any

import torch

from rvc_platform import __version__
from rvc_platform.application.realtime_service import RealtimeService
from rvc_platform.core.entities import (
    EngineStatus,
    ModelInfo,
    SystemStatus,
    TrainingProgress,
)
from rvc_platform.utils.environment import detect_system


class SystemService:
    def __init__(self, engine, profile, training_service: Any = None,
                 realtime_service: RealtimeService | None = None,
                 tunnel_service: Any = None) -> None:
        self.engine = engine
        self.profile = profile
        self.training = training_service
        self.realtime = realtime_service
        self.tunnel = tunnel_service
        self.profile_cache: Any = None
        self.profile_ts: float = 0.0

    def _system_profile(self):
        if self.profile_cache is None or time.time() - self.profile_ts > 5:
            self.profile_cache = detect_system()
            self.profile_ts = time.time()
        return self.profile_cache

    def _vram(self) -> tuple[int | None, int | None]:
        if not (torch.cuda.is_available() and self.profile.device.startswith("cuda")):
            return None, None
        idx = int(self.profile.device.split(":")[1]) if ":" in self.profile.device else 0
        total = torch.cuda.get_device_properties(idx).total_memory // (1024 * 1024)
        used = torch.cuda.memory_allocated(idx) // (1024 * 1024)
        return int(total), int(used)

    def _ram(self) -> tuple[int, int]:
        try:
            import psutil

            vm = psutil.virtual_memory()
            return int(vm.total // (1024 * 1024)), int(vm.used // (1024 * 1024))
        except Exception:
            return 0, 0

    def status(self) -> SystemStatus:
        sysp = self._system_profile()
        vram_total, vram_used = self._vram()
        ram_total, ram_used = self._ram()
        loaded: ModelInfo | None = self.engine.loaded_model
        engine_state = EngineStatus(self.engine.status)
        training: TrainingProgress | None = None
        if self.training is not None:
            try:
                training = self.training.status()
            except Exception:
                training = None
        tunnel = None
        if self.tunnel is not None:
            try:
                tunnel = self.tunnel.status()
            except Exception:
                tunnel = None
        return SystemStatus(
            app_version=__version__,
            os=sysp.os,
            arch=sysp.arch,
            python=sysp.python_version,
            device=self.profile.device,
            is_half=self.profile.is_half,
            gpu_name=self.profile.gpu_name,
            vram_total_mb=vram_total,
            vram_used_mb=vram_used,
            ram_total_mb=ram_total,
            ram_used_mb=ram_used,
            disk_free_gb=sysp.disk_free_gb,
            engine=engine_state,
            engine_message=self.engine.status_message,
            loaded_model=loaded,
            training=training,
            conversion_server=self.engine.status == "ready",
            realtime_sessions=self.realtime.active() if self.realtime else 0,
            tunnel=tunnel,
        )
