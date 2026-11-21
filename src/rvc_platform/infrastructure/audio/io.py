"""Audio input/output: any-format loading, metadata, validation, writing.

The upstream RVC repository decodes arbitrary containers by shelling out to
``ffmpeg`` (wav/flac/mp3/ogg/m4a/webm...). We keep that behaviour — it is the
most robust decoder available — and add a pure-``soundfile`` fallback for
wav/flac when ffmpeg is absent. Uploaded files are treated as untrusted:
extensions and headers are validated, magic sniffed, sizes bounded, and
decoding always runs with a byte/time ceiling.
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from rvc_platform.core.errors import ConversionError, SecurityError

ALLOWED_AUDIO_SUFFIXES = {
    ".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".webm", ".aiff", ".aif"
}
AUDIO_MAGIC = {
    b"RIFF", b"fLaC", b"OggS", b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2",
    b"\x00\x00\x00\x20ftyp", b"\x00\x00\x00\x18ftyp", b"\x00\x00\x00 ftyp",
}
MAX_DURATION_S = 60 * 30          # 30 minutes of audio is plenty
FFMPEG_TIMEOUT_S = 120


def probe_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def validate_audio_file(path: Path, max_mb: int = 200) -> dict[str, Any]:
    """Security + sanity validation for a user-supplied audio file."""
    if not path.exists() or not path.is_file():
        raise SecurityError(f"file not found: {path}")
    if path.suffix.lower() not in ALLOWED_AUDIO_SUFFIXES:
        raise SecurityError(f"unsupported audio extension: {path.suffix}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise SecurityError(f"file too large: {size_mb:.1f} MB > {max_mb} MB")
    head = path.open("rb").read(12)
    if not any(head.startswith(m) for m in AUDIO_MAGIC) and not head.startswith(b"ftyp"):
        raise SecurityError("audio container signature not recognised")
    try:
        info = sf.info(str(path))
        return {
            "path": str(path),
            "size_mb": round(size_mb, 2),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration_s": round(float(info.duration), 3),
            "format": info.format,
            "subtype": info.subtype,
        }
    except Exception:
        # not decodable by soundfile (mp3/m4a...) — ffmpeg path will handle it
        return {
            "path": str(path),
            "size_mb": round(size_mb, 2),
            "sample_rate": None,
            "channels": None,
            "duration_s": None,
            "format": path.suffix.lstrip(".").lower(),
            "subtype": None,
        }


def load_audio(path: Path, target_sr: int, ffmpeg: bool | None = None) -> np.ndarray:
    """Decode any supported audio file to mono float32 at ``target_sr``."""
    if ffmpeg is None:
        ffmpeg = probe_ffmpeg()
    if ffmpeg:
        # float32 PCM f32le -> numpy (current official infer/audio.py
        # _load_audio_ffmpeg: full precision, no int16 quantisation loss)
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-i", str(path),
            "-t", str(MAX_DURATION_S),
            "-vn", "-ac", "1", "-ar", str(target_sr),
            "-f", "f32le", "-acodec", "pcm_f32le", "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise ConversionError(f"ffmpeg timed out decoding {path.name}") from exc
        if proc.returncode != 0 or not proc.stdout:
            raise ConversionError(
                f"ffmpeg failed to decode {path.name}: "
                f"{proc.stderr.decode(errors='ignore')[-500:]}"
            )
        return np.frombuffer(proc.stdout, dtype=np.float32).flatten()
    # fallback: soundfile readers + scipy/librosa resample on read is not
    # supported directly, so decode then resample via the DSP module
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise ConversionError(
            f"cannot decode {path.name} without ffmpeg ({exc}); install ffmpeg "
            "or upload WAV/FLAC"
        ) from exc
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        from rvc_platform.infrastructure.audio.dsp import resample

        audio = resample(audio, sr, target_sr)
    return audio.astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    """Write float [-1,1] audio; normalises only when clipping is detected."""
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak * 0.99
    if subtype == "PCM_16":
        sf.write(str(path), audio, sr, subtype="PCM_16")
    else:
        sf.write(str(path), audio, sr, subtype=subtype)


def wav_metadata(path: Path) -> dict[str, Any]:
    """Best-effort metadata incl. pure-stdlib WAV path (used by the tests)."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                return {
                    "sample_rate": w.getframerate(),
                    "channels": w.getnchannels(),
                    "frames": w.getnframes(),
                    "duration_s": round(w.getnframes() / w.getframerate(), 3),
                    "width_bytes": w.getsampwidth(),
                }
        except wave.Error:
            pass
    return validate_audio_file(path)


def detect_wav_sr(path: Path) -> int | None:
    try:
        info = sf.info(str(path))
        return int(info.samplerate)
    except Exception:
        return None
