"""Preprocessing pipeline + dataset quality check tests (real slicing)."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from rvc_platform.training.preprocess import DatasetChecker, DatasetPreprocessor


def test_dataset_check_flags_issues(tmp_path):
    good = tmp_path / "ds"
    good.mkdir()
    from tests.conftest import speech_like

    sf.write(str(good / "a.wav"), speech_like(16_000, 3.0), 16_000, subtype="PCM_16")
    sf.write(str(good / "b.wav"), speech_like(16_000, 3.0, seed=9), 16_000, subtype="PCM_16")
    (good / "dup.wav").write_bytes((good / "a.wav").read_bytes())  # exact duplicate
    (good / "broken.wav").write_bytes(b"RIFF-broken-junk")
    (good / "readme.txt").write_text("not audio", encoding="utf-8")

    report = DatasetChecker().check(good)
    assert report.total_files == 5
    assert report.usable_files == 3
    assert report.unsupported == ["readme.txt"]
    assert report.corrupt == ["broken.wav"]
    assert len(report.duplicates) == 1  # dup.wav == a.wav
    assert report.total_duration_s >= 8.5


def test_preprocess_slices_and_writes_dual_rate(tmp_path):
    from tests.conftest import speech_like

    ds = tmp_path / "ds"
    ds.mkdir()
    # long enough (>4s) to exercise multiple slicer cuts + chunk overlap
    sf.write(str(ds / "long.wav"), speech_like(16_000, 12.0), 16_000, subtype="PCM_16")

    run_dir = tmp_path / "run"
    pp = DatasetPreprocessor(40_000, run_dir)
    report = pp.run(ds)
    assert report["files"] == 1
    assert report["chunks"] >= 2, "12 s of speech must produce several chunks"
    gt = list((run_dir / "0_gt_wavs").glob("*.wav"))
    w16 = list((run_dir / "1_16k_wavs").glob("*.wav"))
    assert len(gt) == len(w16) == report["chunks"]
    # verify the dual-rate twins are time-aligned & normalised
    for g in gt[:3]:
        twin = run_dir / "1_16k_wavs" / g.name
        a, sr_a = sf.read(str(g))
        b, sr_b = sf.read(str(twin))
        assert sr_a == 40_000 and sr_b == 16_000
        assert abs(len(a) / 40_000 - len(b) / 16_000) < 0.01
        # upstream alpha-blended normalisation: 0.675*norm + 0.25*raw — for hot
        # chunks (peak ~1.4) the result is ~1.02; upstream ships these floats
        assert float(np.max(np.abs(a))) <= 1.05


def test_preprocess_idempotent_skip_is_upstream_behaviour(tmp_path):
    from tests.conftest import speech_like

    ds = tmp_path / "ds"
    ds.mkdir()
    sf.write(str(ds / "x.wav"), speech_like(16_000, 5.0), 16_000, subtype="PCM_16")
    run_dir = tmp_path / "run"
    pp = DatasetPreprocessor(40_000, run_dir)
    first = pp.run(ds)
    second = pp.run(ds)
    assert first["chunks"] == second["chunks"]
