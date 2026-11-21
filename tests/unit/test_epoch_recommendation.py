"""Tests for the official FAQ-Q9 epoch recommendation engine (v1.4.1).

RVC FAQ Q9 (docs/en/faq_en.md): noisy/high-floor datasets need only 20-30
epochs ("setting it too high won't improve the audio quality"); clean sets
allow up to 200. Q10: 10-50 min recommended. The engine measures per-file
SNR (80th vs 10th percentile frame RMS) and derives the range.
"""
from __future__ import annotations

import numpy as np
import pytest

from rvc_platform.training.preprocess import (
    DatasetChecker,
    _noise_floor_db,
    _silence_ratio,
    _snr_db,
    recommend_epochs,
)


def _tone(seconds: float = 3.0, freq: float = 220.0, amp: float = 0.5) -> np.ndarray:
    sr = 16_000
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _clean_file() -> np.ndarray:
    """Loud speech-like bursts + very quiet floor."""
    return np.concatenate([_tone(), _tone(amp=0.001)])

def _noisy_file() -> np.ndarray:
    """Same speech + strong broadband noise."""
    rng = np.random.default_rng(7)
    return _clean_file() + 0.15 * rng.standard_normal(_clean_file().size).astype(np.float32)


class TestSnrMeasurement:
    def test_clean_beats_noisy_by_margin(self):
        clean, noisy = _snr_db(_clean_file()), _snr_db(_noisy_file())
        assert clean > 35.0, f"clean synthetic should read >35 dB, got {clean}"
        assert noisy < 20.0, f"noisy synthetic should read <20 dB, got {noisy}"
        assert clean - noisy > 15.0

    def test_silence_is_finite_not_nan(self):
        assert _snr_db(np.zeros(16_000, dtype=np.float32)) == 0.0

    def test_too_short_audio_returns_zero(self):
        assert _snr_db(np.zeros(100, dtype=np.float32)) == 0.0

    def test_noise_floor_sane(self):
        floor = _noise_floor_db(_clean_file())
        assert -120.0 <= floor <= 0.0


class TestRecommendationEngine:
    def test_noisy_gets_official_20_30(self):
        rec = recommend_epochs(12.0, 2_400)
        assert (rec["min"], rec["max"]) == (20, 30)
        assert rec["quality"] == "noisy"
        assert "20-30" in rec["reason"]

    def test_good_gets_50_150(self):
        rec = recommend_epochs(25.0, 2_400)
        assert (rec["min"], rec["max"]) == (50, 150)
        assert rec["quality"] == "good"

    def test_clean_gets_up_to_200(self):
        rec = recommend_epochs(41.8, 2_400)
        assert (rec["min"], rec["max"]) == (100, 200)
        assert rec["quality"] == "clean"

    def test_unknown_snr_is_honest(self):
        rec = recommend_epochs(None, 2_400)
        assert rec["quality"] == "unknown"
        assert rec["min"] <= rec["max"]

    def test_boundary_20_and_35(self):
        assert recommend_epochs(20.0, 2_400)["quality"] == "good"
        assert recommend_epochs(19.99, 2_400)["quality"] == "noisy"
        assert recommend_epochs(35.0, 2_400)["quality"] == "clean"


class TestReportIntegration:
    def test_report_carries_snr_and_recommendation(self, tmp_path):
        # two clean files -> the report should classify them 'clean'
        for i in range(2):
            import scipy.io.wavfile as wf

            audio = (np.clip(_clean_file(), -1.0, 1.0) * 32767).astype(np.int16)
            wf.write(tmp_path / f"c{i}.wav", 16_000, audio)

        report = DatasetChecker(audio_loader=lambda p, sr=None: _clean_file()).check(tmp_path)
        d = report.to_dict()
        assert d["mean_snr_db"] is not None and d["mean_snr_db"] > 35
        assert d["epoch_recommendation"]["quality"] == "clean"
        assert d["epoch_recommendation"]["max"] == 200

    def test_noisy_dataset_report(self, tmp_path):
        import scipy.io.wavfile as wf

        audio = (np.clip(_noisy_file(), -1.0, 1.0) * 32767).astype(np.int16)
        wf.write(tmp_path / "n.wav", 16_000, audio)

        report = DatasetChecker(audio_loader=lambda p, sr=None: _noisy_file()).check(tmp_path)
        d = report.to_dict()
        assert d["epoch_recommendation"]["quality"] == "noisy"
        assert d["epoch_recommendation"]["max"] == 30
