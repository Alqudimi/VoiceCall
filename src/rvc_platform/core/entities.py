"""Domain entities — pure data, no framework imports (only stdlib + numpy types).

These objects cross every layer boundary; application services and engines
operate on them, adapters translate them to/from the outside world.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class F0Method(str, Enum):
    """F0 extraction algorithms actually implemented by the engine.

    Selection rationale (research phase, see docs/ARCHITECTURE.md):
    * ``rmvpe``  — InterSpeech 2023 model; best robustness/accuracy on
      speech (including noisy Iraqi-dialect recordings), GPU fast path.
    * ``fcpe``   — fast context-based estimator, low latency, good quality.
    * ``pm``     — Praat autocorrelation; pure-CPU fallback, no weights.
    crepe/harvest/dio were evaluated and rejected: crepe needs fairseq-era
    torchcrepe pinned against old librosa; harvest/dio are slower with
    more gross-pitch errors on Arabic speech.
    """

    RMVPE = "rmvpe"
    FCPE = "fcpe"
    PM = "pm"


class JobState(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class EngineStatus(str, Enum):
    UNAVAILABLE = "unavailable"
    READY = "ready"
    LOADING = "loading"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class ModelInfo:
    """Discovered voice model on disk (RVC .pth checkpoint)."""

    name: str
    path: Path
    sample_rate: int
    f0: bool
    version: str                    # "v1" | "v2"
    info: str = ""
    size_mb: float = 0.0
    created_at: float = 0.0
    index_path: Path | None = None  # optional retrieval index
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "sample_rate": self.sample_rate,
            "f0": self.f0,
            "version": self.version,
            "info": self.info,
            "size_mb": round(self.size_mb, 2),
            "created_at": self.created_at,
            "index_path": str(self.index_path) if self.index_path else None,
        }


@dataclass
class ConversionSettings:
    """Parameters actually consumed by the conversion engine.

    Every field is implemented — the UI never shows unsupported knobs.
    """

    pitch_shift: int = 0                       # semitones, -24..24
    f0_method: str = F0Method.RMVPE.value
    index_rate: float = 0.0                     # 0..1 retrieval blend
    filter_radius: int = 3                     # median filter radius on F0 (>=2 enables)
    rms_mix_rate: float = 0.25                 # 0..1 output loudness-envelope share
    protect: float = 0.33                      # 0..0.5 voiceless-consonant protection
    output_sample_rate: int | None = None       # None -> model native rate
    denoise: bool = False                      # spectral-gate pre-pass
    normalize: bool = True                     # peak-normalize the result

    def validate(self) -> None:
        if not -24 <= self.pitch_shift <= 24:
            raise ValueError("pitch_shift must be within [-24, 24] semitones")
        if self.f0_method not in {m.value for m in F0Method}:
            raise ValueError(f"unsupported f0 method: {self.f0_method}")
        if not 0.0 <= self.index_rate <= 1.0:
            raise ValueError("index_rate must be within [0, 1]")
        if not 0.0 <= self.rms_mix_rate <= 1.0:
            raise ValueError("rms_mix_rate must be within [0, 1]")
        if not 0.0 <= self.protect <= 0.5:
            raise ValueError("protect must be within [0, 0.5]")


@dataclass
class ConversionResult:
    input_path: Path | None
    output_path: Path
    duration_seconds: float
    processing_seconds: float
    settings: ConversionSettings
    realtime_factor: float = 0.0               # processing / duration (<1 => faster than realtime)

    @property
    def speed_report(self) -> dict[str, Any]:
        return {
            "duration_s": round(self.duration_seconds, 3),
            "processing_s": round(self.processing_seconds, 3),
            "realtime_factor": round(self.realtime_factor, 3),
        }


@dataclass
class LatencyRecord:
    captured_at: float
    inference_ms: float
    total_ms: float

    def to_dict(self) -> dict[str, float]:
        return {"t": self.captured_at, "inference_ms": self.inference_ms, "total_ms": self.total_ms}


@dataclass
class LatencyStats:
    """Streaming latency measured on the wire (capture -> playback bytes)."""

    count: int = 0
    min_ms: float = 0.0
    avg_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0

    @classmethod
    def from_records(cls, records: list[LatencyRecord]) -> "LatencyStats":
        if not records:
            return cls()
        totals = sorted(r.total_ms for r in records)
        n = len(totals)
        p95_idx = min(n - 1, max(0, int(round(0.95 * (n - 1)))))
        return cls(
            count=n,
            min_ms=round(totals[0], 1),
            avg_ms=round(sum(totals) / n, 1),
            p95_ms=round(totals[p95_idx], 1),
            max_ms=round(totals[-1], 1),
        )


@dataclass
class CheckpointInfo:
    """A resumable training checkpoint on disk."""

    run_id: str
    path: Path
    kind: str                                     # "G" | "D" | "state"
    epoch: int
    global_step: int
    created_at: float
    size_mb: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "kind": self.kind,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "created_at": self.created_at,
            "size_mb": round(self.size_mb, 2),
        }


@dataclass
class TrainingProgress:
    state: JobState = JobState.IDLE
    epoch: int = 0
    total_epochs: int = 0
    global_step: int = 0
    steps_per_epoch: int = 0
    latest_losses: dict[str, float] = field(default_factory=dict)
    started_at: float | None = None
    updated_at: float | None = None
    message: str = ""

    @property
    def epoch_fraction(self) -> float:
        return (self.epoch / self.total_epochs) if self.total_epochs else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "global_step": self.global_step,
            "steps_per_epoch": self.steps_per_epoch,
            "latest_losses": self.latest_losses,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "message": self.message,
        }


@dataclass
class SystemStatus:
    app_version: str
    os: str
    arch: str
    python: str
    device: str
    is_half: bool
    gpu_name: str | None
    vram_total_mb: int | None
    vram_used_mb: int | None
    ram_total_mb: int
    ram_used_mb: int
    disk_free_gb: float
    engine: EngineStatus = EngineStatus.UNAVAILABLE
    engine_message: str = ""
    loaded_model: ModelInfo | None = None
    training: TrainingProgress | None = None
    conversion_server: bool = False
    realtime_sessions: int = 0
    tunnel: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_version": self.app_version,
            "os": self.os,
            "arch": self.arch,
            "python": self.python,
            "device": self.device,
            "is_half": self.is_half,
            "gpu": {
                "name": self.gpu_name,
                "vram_total_mb": self.vram_total_mb,
                "vram_used_mb": self.vram_used_mb,
            },
            "ram": {"total_mb": self.ram_total_mb, "used_mb": self.ram_used_mb},
            "disk_free_gb": round(self.disk_free_gb, 2),
            "engine": self.engine.value,
            "engine_message": self.engine_message,
            "loaded_model": self.loaded_model.to_dict() if self.loaded_model else None,
            "training": self.training.to_dict() if self.training else None,
            "conversion_server": self.conversion_server,
            "realtime_sessions": self.realtime_sessions,
            "tunnel": self.tunnel,
            "timestamp": self.timestamp,
        }
