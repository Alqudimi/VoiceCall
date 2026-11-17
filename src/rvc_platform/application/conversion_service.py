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
    def __init__(self, engine, audio_engine, defaults) -> None:
        self.engine = engine
        self.audio = audio_engine
        self.defaults = defaults
        self.paths = get_paths()

    # ------------------------------------------------------------------ settings
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
        merged["f0_method"] = (overrides or {}).get("f0_method") or "rmvpe"
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
        if len(payload) > 40 * 1024 * 1024:
            raise ConversionError("upload too large (>40 MB)")
        safe = _SAFE_ID.sub("_", filename) or "upload.wav"
        suffix = Path(safe).suffix.lower()
        if suffix not in ALLOWED_AUDIO_SUFFIXES:
            raise ConversionError(f"unsupported audio format: {suffix}")
        dest = self.paths.uploads / f"{uuid.uuid4().hex[:8]}_{safe}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        validate_audio_file(dest, max_mb=200)   # magic + size checks; raises on junk
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
        stem = _SAFE_ID.sub("_", name_hint or Path(input_path).stem)[:60]
        out = self.paths.conversions / f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{stem}.wav"
        result = self.engine.convert_file(Path(input_path), out, settings, progress=progress)
        logger.info("converted %s -> %s (%s)", input_path, out, result.speed_report)
        return result

    def result_path(self, result_id: str) -> Path:
        # conversion results are addressed by their file name (hex-prefixed)
        for path in self.paths.conversions.glob(f"*_{_SAFE_ID.sub('_', result_id)}*"):
            if path.is_file():
                return path
        raise ConversionError(f"result not found: {result_id}")
