"""Audio I/O + path/security validation tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rvc_platform.core.errors import ConversionError, SecurityError
from rvc_platform.infrastructure.audio import io as audio_io
from rvc_platform.infrastructure.paths import Paths, get_paths


def test_paths_redirect_and_singleton(project_env, monkeypatch):
    p = get_paths()
    assert p.root == project_env
    monkeypatch.setenv("RVC_MODELS_DIR", str(project_env / "elsewhere" / "models"))
    from rvc_platform.infrastructure.paths import reset_paths

    reset_paths()
    p2 = get_paths()
    assert p2.models == project_env / "elsewhere" / "models"
    assert p2.root == project_env


def test_write_and_load_wav_roundtrip(project_env, tmp_path):
    from tests.conftest import speech_like

    audio = speech_like(16_000, 1.0)
    path = tmp_path / "a.wav"
    audio_io.write_wav(path, audio, 16_000)
    loaded = audio_io.load_audio(path, 16_000, ffmpeg=False)
    assert loaded.dtype == np.float32
    assert abs(len(loaded) - len(audio)) <= 1
    assert float(np.max(np.abs(loaded))) > 0.1


def test_load_audio_resamples_without_ffmpeg(project_env, tmp_path):
    from tests.conftest import speech_like

    audio = speech_like(48_000, 1.0)
    path = tmp_path / "b.wav"
    sf.write(str(path), audio, 48_000, subtype="PCM_16")
    loaded = audio_io.load_audio(path, 16_000, ffmpeg=False)
    assert abs(len(loaded) - 16_000) <= 200  # resampled to 16 kHz


def test_write_wav_clips_safely(project_env, tmp_path):
    loud = np.full(4_000, 2.5, dtype=np.float32)  # over 1.0
    path = tmp_path / "loud.wav"
    audio_io.write_wav(path, loud, 16_000)
    data, sr = sf.read(str(path))
    assert float(np.max(np.abs(data))) <= 1.0


def test_validate_audio_file_rejects_junk(project_env, tmp_path):
    fake = tmp_path / "fake.wav"
    fake.write_bytes(b"NOTAUDIO" * 100)
    with pytest.raises(SecurityError):
        audio_io.validate_audio_file(fake)


def test_validate_audio_file_rejects_bad_extension(project_env, tmp_path):
    fake = tmp_path / "malicious.exe"
    fake.write_bytes(b"RIFFxxxxx")
    with pytest.raises(SecurityError):
        audio_io.validate_audio_file(fake)


def test_validate_audio_file_rejects_oversize(project_env, tmp_path):
    big = tmp_path / "big.wav"
    big.write_bytes(b"RIFF" + b"\x00" * 1024)  # tiny file, tiny limit triggers size rule
    with pytest.raises(SecurityError):
        audio_io.validate_audio_file(big, max_mb=0)  # max_mb=0 rejects everything


def test_validate_audio_file_accepts_real_wav(project_env, speech_wav_file):
    info = audio_io.validate_audio_file(speech_wav_file)
    assert info["sample_rate"] == 16_000
    assert info["duration_s"] >= 3.5


def test_wav_metadata_stdlib(project_env, speech_wav_file):
    meta = audio_io.wav_metadata(speech_wav_file)
    assert meta["sample_rate"] == 16_000
    assert meta["channels"] == 1
    assert meta["duration_s"] >= 3.5


def test_load_audio_ffmpeg_missing_clear_error(project_env, tmp_path):
    path = tmp_path / "song.mp3"
    path.write_bytes(b"ID3" + b"\x00" * 64)
    with pytest.raises(ConversionError):
        audio_io.load_audio(path, 16_000, ffmpeg=False)
