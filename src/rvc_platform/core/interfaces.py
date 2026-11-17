"""Component contracts (Protocols) — the seams of the system.

Application services depend on these abstractions only; concrete engines
(RVC v2 today, future alternatives) and infrastructure adapters implement
them. Swapping an implementation must never ripple past a service boundary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from rvc_platform.core.entities import (
    CheckpointInfo,
    ConversionResult,
    ConversionSettings,
    JobState,
    ModelInfo,
    TrainingProgress,
)


@runtime_checkable
class VoiceModelEngine(Protocol):
    """Loads RVC checkpoints and converts audio (file mode)."""

    @property
    def status(self) -> str: ...

    def load_model(self, model_path: Path, index_path: Path | None) -> ModelInfo: ...

    def unload(self) -> None: ...

    @property
    def loaded_model(self) -> ModelInfo | None: ...

    def convert_file(
        self,
        input_path: Path,
        output_path: Path,
        settings: ConversionSettings,
        progress: Callable[[float], None] | None = None,
    ) -> ConversionResult: ...


@runtime_checkable
class AudioProcessor(Protocol):
    """Input conditioning shared by training and inference paths."""

    def load_any(self, path: Path, target_sr: int) -> np.ndarray: ...

    def resample(self, audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray: ...

    def normalize(self, audio: np.ndarray, peak: float = 0.9) -> np.ndarray: ...

    def denoise(self, audio: np.ndarray, sr: int, strength: float = 1.0) -> np.ndarray: ...

    def detect_silence(self, audio: np.ndarray, sr: int) -> list[tuple[int, int]]: ...

    def trim_silence(self, audio: np.ndarray, sr: int, margin_ms: int = 120) -> np.ndarray: ...

    def validate_audio_file(self, path: Path) -> dict[str, Any]: ...


@runtime_checkable
class FeatureIndex(Protocol):
    """FAISS retrieval index over training-set HuBERT features."""

    def search(self, feats: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]: ...

    @property
    def size(self) -> int: ...


@runtime_checkable
class TrainingRunner(Protocol):
    """Full lifecycle control of a training run."""

    def start(self, config: dict[str, Any]) -> None: ...

    def stop(self) -> None: ...

    @property
    def state(self) -> JobState: ...

    @property
    def progress(self) -> TrainingProgress: ...

    def list_checkpoints(self) -> list[CheckpointInfo]: ...

    def resume(self, checkpoint: CheckpointInfo | None = None) -> None: ...

    def export(self, epoch: int | None = None) -> Path: ...


@runtime_checkable
class TunnelManager(Protocol):
    """Cloudflare quick-tunnel supervision."""

    def start(self) -> dict[str, Any]: ...

    def stop(self) -> None: ...

    def status(self) -> dict[str, Any]: ...


ProgressCallback = Callable[[dict[str, Any]], None]
