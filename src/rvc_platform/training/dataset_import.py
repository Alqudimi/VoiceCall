"""Import and clean an arbitrary audio folder into ``data/raw``.

This is the user-facing ingestion boundary before the existing RVC pipeline:
- recursively discovers common audio containers/codecs;
- decodes through the project's FFmpeg-aware loader to mono/target rate;
- rejects corrupt, empty, silent-only, and too-short files;
- removes exact duplicates after decoding (not by filename or extension);
- trims only leading/trailing silence with a speech-preserving margin;
- writes deterministic PCM-16 WAV files atomically and emits JSON/CSV reports.

The existing ``prepare-dataset`` command remains responsible for RVC slicing,
normalisation, F0, HuBERT features, and FAISS index generation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rvc_platform.core.errors import DatasetError
from rvc_platform.infrastructure.audio.dsp import trim_silence
from rvc_platform.infrastructure.audio.io import load_audio, write_wav
from rvc_platform.infrastructure.paths import get_paths

logger = logging.getLogger(__name__)

# FFmpeg supports more codecs than these extensions, but an explicit allowlist
# prevents accidentally feeding documents, videos, or arbitrary files to it.
AUDIO_EXTS = {
    ".wav", ".wave", ".aiff", ".aif", ".flac", ".mp3", ".ogg", ".oga", ".opus",
    ".m4a", ".mp4", ".aac", ".wma", ".webm", ".amr", ".ape", ".ac3", ".dts",
    ".mka", ".3gp", ".3gpp", ".caf", ".au", ".snd", ".m4b",
}
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass
class ImportReport:
    source_path: Path
    output_path: Path
    sample_rate: int
    total_candidates: int = 0
    included_files: int = 0
    duplicate_files: list[dict[str, str]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    total_duration_s: float = 0.0
    trimmed_duration_s: float = 0.0
    started_at: float = field(default_factory=time.time)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source_path": str(self.source_path),
            "output_path": str(self.output_path),
            "sample_rate": self.sample_rate,
            "channels": 1,
            "format": "PCM_16 WAV",
            "total_candidates": self.total_candidates,
            "included_files": self.included_files,
            "duplicate_files": len(self.duplicate_files),
            "duplicates": self.duplicate_files,
            "skipped_count": len(self.skipped),
            "skipped": self.skipped,
            "total_duration_s": round(self.total_duration_s, 3),
            "trimmed_duration_s": round(self.trimmed_duration_s, 3),
            "elapsed_s": round(self.elapsed_s, 2),
        }


class AudioDatasetImporter:
    """Prepare an external folder as a project dataset under ``data/raw``."""

    def __init__(self, target_sr: int = 40_000, min_duration_s: float = 0.35,
                 trim_margin_ms: int = 120, silence_threshold_db: float = -40.0) -> None:
        if target_sr not in (32_000, 40_000, 48_000):
            raise DatasetError("target sample rate must be 32k, 40k, or 48k")
        if min_duration_s <= 0:
            raise DatasetError("minimum duration must be greater than zero")
        self.target_sr = target_sr
        self.min_duration_s = min_duration_s
        self.trim_margin_ms = trim_margin_ms
        self.silence_threshold_db = silence_threshold_db

    def import_folder(self, source: Path, dataset_name: str, *, force: bool = False,
                      output_root: Path | None = None, progress=None) -> ImportReport:
        source = Path(source).expanduser().resolve()
        if not source.is_dir():
            raise DatasetError(f"source folder missing: {source}")
        if not _NAME_RE.fullmatch(dataset_name):
            raise DatasetError(
                "dataset name must start with a letter/number and contain only "
                "letters, numbers, '_' or '-'")

        root = Path(output_root).expanduser().resolve() if output_root else get_paths().datasets_raw
        destination = root / dataset_name
        try:
            destination.relative_to(source)
            overlaps_source = True
        except ValueError:
            try:
                source.relative_to(destination)
                overlaps_source = True
            except ValueError:
                overlaps_source = False
        if overlaps_source:
            raise DatasetError("output dataset directory must not overlap the source folder")
        if destination.exists():
            if not force:
                raise DatasetError(
                    f"destination already exists: {destination}; use --force to replace it")
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)

        # Snapshot the input before writing, so a destination inside the source
        # cannot make the scan grow while it is running.
        candidates = sorted(
            p for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        )
        report = ImportReport(source, destination, self.target_sr,
                              total_candidates=len(candidates))
        if not candidates:
            shutil.rmtree(destination, ignore_errors=True)
            raise DatasetError(f"no supported audio files found in {source}")

        seen: dict[str, str] = {}
        included_index = 0
        for index, path in enumerate(candidates, 1):
            relative = str(path.relative_to(source))
            try:
                audio = load_audio(path, self.target_sr)
                original_duration = len(audio) / self.target_sr
                if not np.all(np.isfinite(audio)):
                    raise ValueError("decoded samples contain NaN or infinity")
                if original_duration < self.min_duration_s:
                    report.skipped.append({"file": relative, "reason": "too_short", "duration_s": f"{original_duration:.3f}"})
                    continue
                if float(np.max(np.abs(audio))) < 10 ** (self.silence_threshold_db / 20):
                    report.skipped.append({"file": relative, "reason": "silent_only"})
                    continue

                cleaned = trim_silence(audio, self.target_sr, margin_ms=self.trim_margin_ms,
                                       threshold_db=self.silence_threshold_db)
                cleaned_duration = len(cleaned) / self.target_sr
                if cleaned_duration < self.min_duration_s:
                    report.skipped.append({"file": relative, "reason": "silent_or_too_short_after_trim"})
                    continue

                # Canonical hash makes WAV/MP3/OGG copies of the same cleaned
                # signal duplicates, even when their leading/trailing silence
                # differs or their container metadata is different.
                canonical = np.clip(np.asarray(cleaned, dtype=np.float32), -1.0, 1.0)
                digest = hashlib.sha256(np.rint(canonical * 32767.0).astype("<i2").tobytes()).hexdigest()
                if digest in seen:
                    report.duplicate_files.append({"file": relative, "duplicate_of": seen[digest]})
                    continue

                included_index += 1
                output_name = f"voice_{included_index:04d}.wav"
                final_path = destination / output_name
                # Keep the .wav suffix: soundfile infers the container from it.
                temp_path = destination / f".{output_name}"
                write_wav(temp_path, cleaned.astype(np.float32), self.target_sr, subtype="PCM_16")
                temp_path.replace(final_path)
                seen[digest] = output_name
                report.included_files += 1
                report.total_duration_s += original_duration
                report.trimmed_duration_s += cleaned_duration
            except Exception as exc:  # one bad file must not discard a good dataset
                logger.warning("skipping %s: %s", path, exc)
                report.skipped.append({"file": relative, "reason": "decode_or_process_error", "detail": str(exc)[:300]})
            if progress:
                progress(index / len(candidates), relative)

        report.elapsed_s = time.time() - report.started_at
        if report.included_files == 0:
            shutil.rmtree(destination, ignore_errors=True)
            raise DatasetError("no usable audio files remained after validation")
        self._write_reports(destination, report)
        return report

    @staticmethod
    def _write_reports(destination: Path, report: ImportReport) -> None:
        (destination / "import_report.json").write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (destination / "import_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=("type", "file", "duplicate_of", "reason", "detail"))
            writer.writeheader()
            for item in report.duplicate_files:
                writer.writerow({"type": "duplicate", "file": item["file"], "duplicate_of": item["duplicate_of"], "reason": "", "detail": ""})
            for item in report.skipped:
                writer.writerow({"type": "skipped", "file": item["file"], "duplicate_of": "", "reason": item["reason"], "detail": item.get("detail", "")})


def parse_sample_rate(value: str) -> int:
    """Parse CLI values such as ``40k`` or ``40000``."""
    normalized = value.strip().lower()
    if normalized.endswith("k"):
        normalized = normalized[:-1]
    try:
        rate = int(normalized) * (1000 if value.strip().lower().endswith("k") else 1)
    except ValueError as exc:
        raise DatasetError(f"invalid sample rate: {value}") from exc
    if rate not in (32_000, 40_000, 48_000):
        raise DatasetError("sample rate must be 32k, 40k, or 48k")
    return rate
