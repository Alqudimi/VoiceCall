"""Audio-to-Audio conversion service (file mode)."""
from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

from rvc_platform.core.entities import ConversionResult, ConversionSettings
from rvc_platform.core.errors import ConversionError, ModelNotFoundError
from rvc_platform.infrastructure.audio.io import ALLOWED_AUDIO_SUFFIXES, validate_audio_file
from rvc_platform.infrastructure.paths import get_paths

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")


class ConversionService:
    def __init__(self, engine, audio_engine, defaults, engine_cfg=None,
                 max_upload_mb: int = 40) -> None:
        self.engine = engine
        self.audio = audio_engine
        self.defaults = defaults
        self.engine_cfg = engine_cfg        # EngineConfig (f0_method default knob)
        self.max_upload_mb = int(max_upload_mb)
        self.paths = get_paths()
        self._results: dict[str, str] = {}  # result_id -> exact output filename

    # ------------------------------------------------------------------ settings
    @property
    def default_f0_method(self) -> str:
        """Config-driven default F0 method (engine.f0_method / RVC_F0_METHOD)."""
        return getattr(self.engine_cfg, "f0_method", "rmvpe") if self.engine_cfg else "rmvpe"

    def build_settings(self, overrides: dict[str, Any] | None = None) -> ConversionSettings:
        merged: dict[str, Any] = {
            "pitch_shift": self.defaults.pitch_shift,
            "index_rate": self.defaults.index_rate,
            "filter_radius": self.defaults.filter_radius,
            "rms_mix_rate": self.defaults.rms_mix_rate,
            "protect": self.defaults.protect,
            "denoise": self.defaults.denoise,
            "normalize": self.defaults.normalize,
            "output_sample_rate": self.defaults.output_sample_rate,
        }
        merged["f0_method"] = (overrides or {}).get("f0_method") or self.default_f0_method
        for key, value in (overrides or {}).items():
            if key in merged and value is not None:
                merged[key] = value
        settings = ConversionSettings(**{
            k: v for k, v in merged.items() if k in ConversionSettings.__dataclass_fields__
        })
        settings.validate()
        return settings

    # ------------------------------------------------------------------ upload
    def store_upload(self, filename: str, payload: bytes) -> Path:
        if not payload:
            raise ConversionError("empty upload")
        if len(payload) > self.max_upload_mb * 1024 * 1024:
            raise ConversionError(f"upload too large (>{self.max_upload_mb} MB)")
        safe = _SAFE_ID.sub("_", filename) or "upload.wav"
        suffix = Path(safe).suffix.lower()
        if suffix not in ALLOWED_AUDIO_SUFFIXES:
            raise ConversionError(f"unsupported audio format: {suffix}")
        dest = self.paths.uploads / f"{uuid.uuid4().hex[:8]}_{safe}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        validate_audio_file(dest, max_mb=self.max_upload_mb * 5)  # magic + size; raises on junk
        return dest

    # ------------------------------------------------------------------ convert
    def convert(
        self,
        input_path: Path,
        settings: ConversionSettings,
        name_hint: str | None = None,
        progress=None,
    ) -> ConversionResult:
        if self.engine.net_g is None:
            raise ModelNotFoundError("no model loaded into the engine")
        # name_hint arrives as the raw upload filename (may carry its own
        # extension) — strip it so the output is <...>_<stem>.wav, never .wav.wav
        raw_name = name_hint or Path(input_path).name
        stem = _SAFE_ID.sub("_", Path(raw_name).stem or "audio")[:60]
        out = self.paths.conversions / f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{stem}.wav"
        result = self.engine.convert_file(Path(input_path), out, settings, progress=progress)
        logger.info("converted %s -> %s (%s)", input_path, out, result.speed_report)
        # exact registry: result_id -> filename (no glob guessing on download)
        self._results[out.stem] = out.name
        if len(self._results) > 500:
            self._results = dict(list(self._results.items())[-250:])
        return result

    def result_path(self, result_id: str) -> Path:
        """Exact lookup by stored result id (filename stem), traversal-safe."""
        safe = _SAFE_ID.sub("_", result_id.strip())
        if safe in self._results:
            candidate = self.paths.conversions / self._results[safe]
            if candidate.is_file():
                return candidate
        # exact-name match: the API returns result.output_path.name as the id
        direct = self.paths.conversions / safe
        if direct.is_file():
            return direct
        without_ext = direct.with_suffix("")
        if without_ext.is_file() and without_ext.with_suffix(".wav").is_file():
            return without_ext.with_suffix(".wav")
        raise ConversionError(f"result not found: {result_id}")
