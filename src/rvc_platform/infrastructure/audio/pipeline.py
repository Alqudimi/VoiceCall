"""AudioProcessingEngine — the facade implementing the AudioProcessor protocol.

Single entry point consumed by the conversion service, the training data
pipeline and the real-time engine for input conditioning. Keeps every DSP
decision in one place and therefore testable in isolation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from rvc_platform.infrastructure.audio import dsp, io

logger = logging.getLogger(__name__)


class AudioProcessingEngine:
    def __init__(self, ffmpeg: bool | None = None) -> None:
        self._ffmpeg = ffmpeg if ffmpeg is not None else io.probe_ffmpeg()
        if not self._ffmpeg:
            logger.warning(
                "ffmpeg not found — MP3/M4A/OPUS uploads need it; WAV/FLAC still work"
            )

    @property
    def ffmpeg(self) -> bool:
        return self._ffmpeg

    # ---- AudioProcessor protocol ----
    def load_any(self, path: Path, target_sr: int) -> np.ndarray:
        return io.load_audio(path, target_sr, ffmpeg=self._ffmpeg)

    def resample(self, audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        return dsp.resample(audio, src_sr, dst_sr)

    def normalize(self, audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
        return dsp.normalize(audio, peak)

    def denoise(self, audio: np.ndarray, sr: int, strength: float = 1.0) -> np.ndarray:
        return dsp.denoise_spectral_gate(audio, sr, strength=strength)

    def detect_silence(self, audio: np.ndarray, sr: int) -> list[tuple[int, int]]:
        return dsp.detect_silence(audio, sr)

    def trim_silence(self, audio: np.ndarray, sr: int, margin_ms: int = 120) -> np.ndarray:
        return dsp.trim_silence(audio, sr, margin_ms=margin_ms)

    def validate_audio_file(self, path: Path, max_mb: int = 200) -> dict[str, Any]:
        return io.validate_audio_file(path, max_mb=max_mb)

    def highpass(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return dsp.highpass(audio, sr)

    def write_wav(self, path: Path, audio: np.ndarray, sr: int) -> None:
        io.write_wav(path, audio, sr)

    # high-level conditioning shared by file conversion + training preprocess
    def condition(
        self,
        audio: np.ndarray,
        sr: int,
        denoise: bool = False,
        normalize_peak: float = 0.9,
    ) -> np.ndarray:
        audio = dsp.highpass(audio, sr)
        if denoise:
            audio = self.denoise(audio, sr)
        if normalize_peak > 0:
            audio = dsp.normalize(audio, normalize_peak)
        return audio.astype(np.float32)
