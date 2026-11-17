"""Dataset check: silence ratio, recursive scan, cap reporting."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from rvc_platform.training.preprocess import DatasetChecker


def _write(path, seconds: float, silence_fraction: float = 0.0, sr: int = 16_000):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    sig = 0.4 * np.sin(2 * np.pi * 180 * t).astype(np.float32)
    silent = int(n * silence_fraction)
    sig[:silent] = 1e-6
    sf.write(str(path), sig, sr, subtype="PCM_16")
    return path


def test_silence_ratio_computed(project_env, tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    _write(ds / "clean.wav", 2.0, silence_fraction=0.0)
    _write(ds / "half_silent.wav", 2.0, silence_fraction=0.5)

    report = DatasetChecker().check(ds)
    assert report.mean_silence_ratio is not None
    assert 0.2 < report.mean_silence_ratio < 0.35, "mean of {0.0, ~0.5} expected"


def test_recursive_scan_finds_subfolder_audio(project_env, tmp_path):
    ds = tmp_path / "ds2"
    (ds / "speaker_a").mkdir(parents=True)
    (ds / "speaker_b").mkdir(parents=True)
    _write(ds / "speaker_a" / "a.wav", 1.0)
    _write(ds / "speaker_b" / "b.wav", 1.0)
    (ds / "speaker_b" / "notes.txt").write_text("not audio")

    report = DatasetChecker().check(ds)
    assert report.usable_files == 2
    assert "notes.txt" in report.unsupported


def test_cap_is_reported_not_silent(project_env, tmp_path, caplog):
    ds = tmp_path / "ds3"
    ds.mkdir()
    for i in range(6):
        _write(ds / f"f{i}.wav", 0.5)
    report = DatasetChecker().check(ds, max_files=3)
    assert report.capped_at == 3, "truncation must be reported in the report"
    assert report.usable_files == 3


def test_duplicates_and_corruption(project_env, tmp_path):
    ds = tmp_path / "ds4"
    ds.mkdir()
    good = _write(ds / "one.wav", 1.0)
    _write(ds / "copy.wav", 1.0)
    # exact duplicate content: same samples -> rewrite identical bytes
    sf.write(str(ds / "copy.wav"), sf.read(str(good))[0], 16_000, subtype="PCM_16")
    (ds / "broken.wav").write_bytes(b"RIFF-not-really-audio")

    report = DatasetChecker().check(ds)
    assert "broken.wav" in report.corrupt
    assert report.duplicates, "content hash must detect the copy"
    assert report.to_dict()["mean_silence_ratio"] is not None
