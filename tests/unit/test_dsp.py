"""DSP primitive tests — real numerics, no mocks."""
from __future__ import annotations

import numpy as np

from rvc_platform.infrastructure.audio import dsp


def test_resample_changes_length_and_keeps_content():
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(16_000) / 16_000).astype(np.float32)
    out = dsp.resample(tone, 16_000, 32_000)
    assert len(out) == 32_000
    assert np.max(np.abs(out)) <= 0.6
    energy_out = float(np.sqrt(np.mean(out ** 2)))
    assert 0.2 < energy_out < 0.6


def test_normalize_scales_to_peak():
    audio = np.array([0.1, -0.2, 0.05, -0.05], dtype=np.float32)
    out = dsp.normalize(audio, 0.9)
    assert np.isclose(np.max(np.abs(out)), 0.9, atol=1e-6)
    silent = np.zeros(100, dtype=np.float32)
    assert np.allclose(dsp.normalize(silent, 0.9), 0.0)


def test_highpass_removes_dc_offset():
    t = np.arange(16_000) / 16_000
    signal = (1.0 + 0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    filtered = dsp.highpass(signal, 16_000)
    assert abs(float(np.mean(filtered))) < 0.05  # DC largely removed
    assert abs(float(np.mean(signal))) > 0.9


def test_detect_silence_finds_padded_gaps():
    from tests.conftest import speech_like

    audio = speech_like(16_000, 4.0)
    spans = dsp.detect_silence(audio, 16_000)
    assert spans, "leading/trailing silence must be detected"
    assert spans[0][0] == 0 or spans[-1][1] >= len(audio) - 1
    speech_core = audio[16_000 // 2 : -16_000 // 2]
    core_spans = dsp.detect_silence(speech_core, 16_000)
    assert not core_spans, "voiced middle must not be flagged as silence"


def test_trim_silence_keeps_speech_core():
    from tests.conftest import speech_like

    audio = speech_like(16_000, 4.0)
    trimmed = dsp.trim_silence(audio, 16_000)
    assert len(trimmed) < len(audio)
    assert len(trimmed) > 16_000 * 2  # most of the speech kept
    assert float(np.max(np.abs(trimmed))) > 0.1


def test_crossfade_concat_is_continuous():
    a = np.ones(500, dtype=np.float32)
    b = np.zeros(500, dtype=np.float32)
    out = dsp.crossfade_concat(a, b, fade=100)
    assert len(out) == 900
    assert out[0] == 1.0 and out[899] == 0.0
    # midpoint of the linear ramp sits in the middle of the blend region [400..500)
    assert abs(float(out[450]) - 0.5) < 0.02


def test_denoise_attenuates_broadband_noise():
    from tests.conftest import speech_like

    audio = speech_like(16_000, 2.0, seed=5)
    noisy = audio + 0.05 * np.random.default_rng(1).standard_normal(audio.shape).astype(np.float32)
    cleaned = dsp.denoise_spectral_gate(noisy, 16_000, strength=1.5)
    # noise floor (measured on the quiet margins) must drop
    def floor_energy(x):
        seg = x[: int(16_000 * 0.2)]
        return float(np.sqrt(np.mean(seg ** 2)))
    assert floor_energy(cleaned) < floor_energy(noisy)


def test_vad_mask_shape():
    from tests.conftest import speech_like

    audio = speech_like(16_000, 2.0)
    mask = dsp.vad_mask(audio, 16_000)
    assert mask.shape == audio.shape
    assert mask[: int(16_000 * 0.3)].mean() < 0.1   # leading silence unvoiced
    assert mask[16_000 : 16_000 + 5000].mean() > 0.5  # voiced speech flagged
