"""F0 extraction strategies + the shared Hz->coarse-bin mapping.

All inference paths (file + realtime) go through this registry so the F0
algorithm is a configuration choice, never a hard-coded one. The Hz->mel-bin
quantisation below is format-critical: it must match the embedding table of
the venduated SynthesizerTrn models (256 pitch bins, mel-warped 50..1100 Hz).
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import torch

from rvc_platform.core.errors import ConfigurationError
from rvc_platform.infrastructure.paths import get_paths

logger = logging.getLogger(__name__)

F0_MIN = 50.0
F0_MAX = 1100.0
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)


class F0Extractor(Protocol):
    name: str

    def extract(self, audio_16k: np.ndarray) -> np.ndarray:
        """Return per-frame F0 in Hz (0 = unvoiced), hop 10 ms."""
        ...


# --------------------------------------------------------------------------- #
# Hz <-> coarse pitch bin mapping (format-critical, mirrors upstream exactly)
# --------------------------------------------------------------------------- #
def f0_to_coarse(f0_hz: np.ndarray | torch.Tensor):
    """Return (coarse_bins int64 1..255, fine_hz float32) — same shapes as input."""
    if isinstance(f0_hz, torch.Tensor):
        f0 = f0_hz.float().cpu().numpy()
    else:
        f0 = np.asarray(f0_hz, dtype=np.float32)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    coarse = np.rint(f0_mel).astype(np.int64)
    return coarse, f0


def interp_unvoiced(f0: np.ndarray) -> np.ndarray:
    """Linearly interpolate unvoiced (0 Hz) frames from voiced neighbours."""
    uv = f0 == 0
    if np.any(~uv):
        indices = np.arange(len(f0))
        f0[uv] = np.interp(indices[uv], indices[~uv], f0[~uv])
    return f0


def median_filter_f0(f0: np.ndarray, radius: int) -> np.ndarray:
    """RVC's legacy 'filter radius' knob — OFFICIAL semantics.

    Upstream WebUI applied ``signal.medfilt(f0, 3)`` — a FIXED kernel of 3 —
    only when ``filter_radius > 2``; the current official pipeline removed
    median filtering from inference entirely (infer/vc/pipeline.py get_f0 has
    no filter_radius). We preserve the legacy knob for opt-in users with its
    exact kernel (3, never 2r+1 — a kernel-7 default over-smooths vibrato and
    pitch dynamics); the platform default (radius 0) matches current official.
    """
    if radius <= 2 or f0.size == 0:
        return f0
    from scipy.signal import medfilt

    return medfilt(f0, kernel_size=3).astype(f0.dtype)


def shift_f0(f0: np.ndarray, semitones: float) -> np.ndarray:
    return f0 * (2.0 ** (semitones / 12.0))


# --------------------------------------------------------------------------- #
# Concrete extractors
# --------------------------------------------------------------------------- #
class RMVPEExtractor:
    """InterSpeech-2023 neural pitch tracker — default for Arabic speech."""

    name = "rmvpe"

    def __init__(self, device: str, is_half: bool, model_path: Path | None = None) -> None:
        from rvc_platform.engine.rvc.rmvpe import RMVPE

        path = model_path or get_paths().rmvpe
        if not path.is_file():
            from rvc_platform.core.errors import AssetMissingError

            raise AssetMissingError(
                "rmvpe.pt", "run `rvc-platform doctor --fix` or `make download-models`"
            )
        self.model = RMVPE(str(path), is_half=is_half, device=device)

    def extract(self, audio_16k: np.ndarray) -> np.ndarray:
        f0 = self.model.infer_from_audio(audio_16k, thred=0.03)
        return np.asarray(f0, dtype=np.float32)


class FCPEExtractor:
    """Fast context-based estimator — the low-latency alternative."""

    name = "fcpe"

    def __init__(self, device: str, is_half: bool = False) -> None:
        from rvc_platform.engine.rvc.fcpe import FCPEInfer

        self.model = FCPEInfer(torch.device(device) if ":" not in device else torch.device(device))

    def extract(self, audio_16k: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(audio_16k, dtype=np.float32))
        f0 = self.model.infer(
            x.unsqueeze(0), sr=16000, decoder_mode="local_argmax", threshold=0.006,
        ).squeeze().detach().cpu().numpy()
        return np.asarray(f0, dtype=np.float32)


class ParselmouthExtractor:
    """Praat autocorrelation — dependency-free CPU fallback (no weights)."""

    name = "pm"

    def __init__(self, device: str = "cpu", is_half: bool = False) -> None:
        self.device = device

    def extract(self, audio_16k: np.ndarray) -> np.ndarray:
        import parselmouth

        # upstream get_f0('pm'): f0_min=50, time_step=window/sr*1000 = 10 ms,
        # voicing_threshold=0.6, pitch_ceiling=1100 (infer/vc/pipeline.py).
        # 50 Hz floor (not Praat's 65 default) keeps deep male voices — a
        # 65 Hz floor zeroes the bottom of many Iraqi male speakers.
        f0_min = 50.0
        l_pad = int(np.ceil(1.5 / f0_min * 16000))
        r_pad = l_pad + 1
        padded = np.pad(np.asarray(audio_16k), (l_pad, r_pad))
        sound = parselmouth.Sound(padded, 16000).to_pitch_ac(
            time_step=0.01,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=1100,
        )
        f0 = sound.selected_array["frequency"].astype(np.float32)
        # upstream centres the length mismatch: pad_size=(p_len-len+1)//2 on
        # BOTH sides so the F0 timeline stays aligned with HuBERT frames —
        # right-only padding shifts pitch frames relative to features.
        p_len = len(audio_16k) // 160 + 1
        if len(f0) < p_len:
            pad_size = (p_len - len(f0) + 1) // 2
            f0 = np.pad(f0, (pad_size, p_len - len(f0) - pad_size))
        elif len(f0) > p_len:
            start = (len(f0) - p_len) // 2
            f0 = f0[start : start + p_len]
        return f0[:p_len]


class CrepeExtractor:
    """CREPE pitch tracker via ``torchcrepe`` — optional quality method.

    Kim et al., "CREPE: A Convolutional Representation for Pitch Estimation"
    (Interspeech 2018): monophonic pitch tracking with strong robustness to
    noise. Offered as an alternative F0 source for datasets where RMVPE
    struggles (breathy/whispery/noisy recordings); frames with low periodicity
    (< 0.1) are treated as unvoiced, matching the CREPE evaluation protocol.

    Optional dependency (Colab quality group): ``uv pip install torchcrepe``;
    selecting ``--f0-method crepe`` without the library is a clear
    ConfigurationError, never a silent fallback.
    """

    name = "crepe"

    def __init__(self, device: str, is_half: bool = False,
                 capacity: str = "full") -> None:
        try:
            import torchcrepe
        except ImportError as exc:
            raise ConfigurationError(
                "f0 method 'crepe' requires the torchcrepe package — "
                "install it with: uv pip install torchcrepe"
            ) from exc
        self._torchcrepe = torchcrepe
        self.device = device if (device == "cpu" or ":" in device) else "cpu"
        self.capacity = capacity

    def extract(self, audio_16k: np.ndarray) -> np.ndarray:
        audio = torch.from_numpy(
            np.ascontiguousarray(audio_16k, dtype=np.float32),
        ).unsqueeze(0)
        with torch.no_grad():
            f0, periodicity = self._torchcrepe.predict(
                audio, 16_000, hop_length=160, batch_size=512,
                device=self.device, return_periodicity=True,
                capacity=self.capacity,
            )
        f0 = f0.squeeze(0)
        periodicity = periodicity.squeeze(0)
        voiced = periodicity >= 0.1
        f0 = torch.where(voiced, f0, torch.zeros_like(f0))
        return f0.cpu().numpy().astype(np.float32)


_EXTRACTORS = {
    "rmvpe": RMVPEExtractor,
    "fcpe": FCPEExtractor,
    "pm": ParselmouthExtractor,
    "crepe": CrepeExtractor,
}


def available_methods() -> list[str]:
    return list(_EXTRACTORS)


def get_f0_extractor(method: str, device: str, is_half: bool) -> F0Extractor:
    method = (method or "").lower()
    if method not in _EXTRACTORS:
        raise ConfigurationError(
            f"unsupported F0 method {method!r}; available: {available_methods()}"
        )
    return _EXTRACTORS[method](device=device, is_half=is_half)


class CachedF0:
    """Lazily constructed per-device extractor cache (models stay resident)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, bool], F0Extractor] = {}

    def get(self, method: str, device: str, is_half: bool) -> F0Extractor:
        key = (method, str(device), bool(is_half))
        if key not in self._cache:
            self._cache[key] = get_f0_extractor(method, device, is_half)
        return self._cache[key]


# process-wide shared cache (inference + realtime engines reuse weights)
SHARED_CACHE = CachedF0()
