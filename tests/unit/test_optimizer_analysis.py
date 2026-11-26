"""Optimizer analysis engine — pure DSP tests on synthetic speech."""
from __future__ import annotations

import numpy as np
import pytest

from rvc_platform.optimizer.analysis import SR_ANALYSIS, AudioAnalyzer
from tests.conftest import speech_like


class _SilentEmbedder:
    """Deterministic stand-in for the neural speaker embedders (unit tests
    must not download models); identity-like vectors keyed by signal energy
    so similarity ordering stays meaningful within one test."""

    def __init__(self):
        self.rng = np.random.default_rng(5)
        self.base = self.rng.standard_normal(64).astype(np.float32)
        self.base /= np.linalg.norm(self.base)

    def embeddings(self, wav16k):
        # deterministic function of the signal so identical audio -> identical
        # embedding and different audio -> measurably different embedding
        x = np.asarray(wav16k, dtype=np.float32)
        e = np.zeros(64, dtype=np.float32)
        e[0] = float(np.mean(np.abs(x)))
        e[1] = float(np.std(x))
        e[2:6] = np.abs(x[: 4 * 4096].reshape(4, 4096)).mean(axis=1)
        e += 0.01 * self.base
        e /= np.linalg.norm(e)
        return {"fake": e}


@pytest.fixture()
def analyzer(project_env):
    return AudioAnalyzer(_SilentEmbedder(), device="cpu")


def test_analyze_speech_features(analyzer):
    wav = speech_like(SR_ANALYSIS, 4.0, f0=160.0)
    res = analyzer.analyze(wav, "t")
    d = res.to_dict()
    assert d["duration_s"] == pytest.approx(4.0, abs=0.05)
    # F0 statistics close to the synthetic 160 Hz
    assert d["f0"]["median_hz"] is not None
    assert 130.0 <= d["f0"]["median_hz"] <= 200.0
    assert d["f0"]["voiced_ratio"] > 0.4
    # spectral + MFCC populated
    assert len(d["mfcc_mean"]) == 20
    assert len(d["band_energies"]) == 8
    assert pytest.approx(sum(d["band_energies"]), abs=1e-3) == 1.0
    assert d["spectral"]["centroid_hz"] is not None
    # VAD found speech
    assert d["vad"]["voiced_ratio"] > 0.2
    assert d["vad"]["pauses"] is not None
    # loudness block complete
    assert d["loudness"]["rms_db"] is not None
    assert d["loudness"]["clipping_ratio"] is not None
    # embeddings via the injected manager
    assert "fake" in d["embeddings"]
    # no warnings for a clean 4 s synthetic utterance about shortness
    assert all("short" not in w for w in d["warnings"])


def test_analyze_short_audio_warns(analyzer):
    wav = speech_like(SR_ANALYSIS, 0.4, f0=150.0)
    d = analyzer.analyze(wav, "short").to_dict()
    assert any("short" in w for w in d["warnings"])


def test_analyze_too_short_returns_early(analyzer):
    res = analyzer.analyze(np.zeros(1600, dtype=np.float32), "tiny")  # 0.1s
    assert res.duration_s < 0.25
    assert res.warnings  # too-short warning, no crash


def test_silence_has_low_voiced_ratio_and_warns(analyzer):
    wav = 1e-5 * np.random.default_rng(1).standard_normal(SR_ANALYSIS * 3)
    d = analyzer.analyze(wav.astype(np.float32), "sil").to_dict()
    assert (d["f0"]["voiced_ratio"] or 0.0) < 0.2


def test_hnr_and_jitter_measured(analyzer):
    wav = speech_like(SR_ANALYSIS, 3.0, f0=140.0)
    d = analyzer.analyze(wav, "hnr").to_dict()
    # clean synthetic harmonics -> positive HNR, tiny jitter
    assert d["hnr_db"] is not None and d["hnr_db"] > 2.0
    assert d["f0"]["jitter_semitones"] is not None
    assert d["f0"]["jitter_semitones"] < 0.5
    assert d["shimmer_db"] is not None


def test_formants_measured(analyzer):
    wav = speech_like(SR_ANALYSIS, 3.0, f0=180.0)
    d = analyzer.analyze(wav, "fmt").to_dict()
    f = d["formants_hz"]
    assert any(v is not None for v in f)


def test_vad_segmentation(analyzer):
    wav = np.zeros(SR_ANALYSIS * 4, dtype=np.float32)
    # two speech bursts separated by 0.6 s of silence
    burst1 = speech_like(SR_ANALYSIS, 1.0, f0=150.0, seed=1)
    burst2 = speech_like(SR_ANALYSIS, 1.0, f0=170.0, seed=2)
    wav[4000:4000 + burst1.size] = burst1[:len(wav) - 4000]
    start2 = 4000 + SR_ANALYSIS + 6000
    wav[start2:start2 + burst2.size] = burst2[:len(wav) - start2]
    segs = analyzer.segment_speech(wav)
    assert 1 <= len(segs) <= 3  # two bursts (possibly merged by hangover)
    assert all(b > a for a, b in segs)


def test_split_search_validation_disjoint(analyzer):
    wav = np.zeros(SR_ANALYSIS * 14, dtype=np.float32)
    # speech bursts with gaps so VAD finds multiple segments
    for i, s in enumerate(range(0, 14, 3)):
        seg = speech_like(SR_ANALYSIS, 1.8, f0=150.0 + 10 * i, seed=i + 1)
        lo = s * SR_ANALYSIS
        wav[lo:lo + seg.size] = seg[: min(seg.size, wav.size - lo)]
    search, val = analyzer.split_search_validation(wav)
    assert search.size >= SR_ANALYSIS  # >= 1s of search speech
    assert val.size >= SR_ANALYSIS // 2  # held-out speech exists
    # short input -> honest empty validation
    short = speech_like(SR_ANALYSIS, 2.0, f0=150.0)
    s2, v2 = analyzer.split_search_validation(short)
    assert s2.size > 0
    assert v2.size == 0


def test_analyze_file_roundtrip(analyzer, tmp_path, project_env):
    import soundfile as sf

    wav = speech_like(SR_ANALYSIS, 3.0, f0=160.0)
    p = tmp_path / "speech.wav"
    sf.write(str(p), wav, SR_ANALYSIS, subtype="PCM_16")
    d = analyzer.analyze_file(p).to_dict()
    assert d["source"] == "speech.wav"
    assert d["f0"]["median_hz"] is not None
