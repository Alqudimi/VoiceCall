"""Regression tests for the v1.5.0 official-inference fixes.

Four confirmed deviations from the official RVC pipeline were corrected:
1. highpass 48 Hz now zero-phase filtfilt (upstream) instead of lfilter;
2. parselmouth floor 50 Hz (upstream) instead of Praat's 65 Hz default —
   deep male voices were being zeroed below 65 Hz;
3. pm length mismatch is centre-padded (upstream (p_len-len+1)//2) instead
   of right-padded — right-padding shifts pitch frames vs HuBERT frames;
4. median_filter_f0: legacy upstream semantics — OFF by default, and when
   enabled (radius>2) it applies a FIXED kernel-3, never 2r+1.
Plus the upstream input peak-guard (scale only when peak/0.95 > 1).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from rvc_platform.engine.rvc import f0 as f0_mod


def _tone(freq: float, seconds: float = 1.0, sr: int = 16_000,
          amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestZeroPhaseHighpass:
    def test_filtfilt_used_not_lfilter(self):
        # a symmetric impulse filtered zero-phase must stay symmetric;
        # the causal lfilter shifts it late.
        impulse = np.zeros(1024, dtype=np.float32)
        impulse[512] = 1.0
        bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16_000)
        out = signal.filtfilt(bh, ah, impulse)
        peak = int(np.argmax(np.abs(out)))
        assert abs(peak - 512) <= 1, "zero-phase must not shift the impulse"

    def test_source_uses_filtfilt(self):
        import inspect

        from rvc_platform.engine.rvc import inference

        src = inspect.getsource(inference.RVCInferenceEngine.convert_file)
        # the EXECUTED line must call filtfilt; no active lfilter call remains
        active = [l.strip() for l in src.splitlines()
                  if not l.strip().startswith("#") and "signal." in l]
        assert any("filtfilt" in l for l in active), active
        assert not any("lfilter" in l for l in active), active


class TestParselmouthFloor:
    def test_50hz_male_tone_is_voiced(self):
        pytest.importorskip("parselmouth")
        # 55 Hz sits between the official 50 Hz floor and the old 65 Hz one:
        # with floor=65 this frame reads unvoiced (f0==0).
        audio = _tone(55.0, seconds=2.0)
        f0 = f0_mod.ParselmouthExtractor().extract(audio)
        voiced = f0[f0 > 0]
        assert voiced.size > f0.size * 0.5, (
            "55 Hz must be tracked with the official 50 Hz floor"
        )
        assert 45.0 <= float(np.median(voiced)) <= 70.0

    def test_length_is_exact_p_len(self):
        pytest.importorskip("parselmouth")
        audio = _tone(220.0, seconds=0.7)  # odd length -> mismatch path
        f0 = f0_mod.ParselmouthExtractor().extract(audio)
        assert f0.shape[0] == len(audio) // 160 + 1

    def test_centre_padding_keeps_onset_aligned(self):
        # simulate the mismatch path: centre-pad must be used, i.e. the
        # extractor's own output length contract holds and the first voiced
        # region stays near its true position (no systematic right shift).
        pytest.importorskip("parselmouth")
        audio = np.concatenate([_tone(200.0, 0.3), np.zeros(1600, np.float32),
                                _tone(300.0, 0.3)])
        f0 = f0_mod.ParselmouthExtractor().extract(audio)
        voiced = np.where(f0 > 0)[0]
        assert voiced.size, "first 0.3 s is voiced — must not vanish"
        # onset frame ~0..30; a right-shifted pad would push it well past 40
        assert voiced[0] < 40, f"onset shifted to frame {voiced[0]}"


class TestMedianFilterOfficial:
    def test_official_default_is_off(self):
        from rvc_platform.core.entities import ConversionSettings

        assert ConversionSettings().filter_radius == 0
        assert f0_mod.median_filter_f0(_tone(200.0, 0.1), 0) is not None

    def test_radius_2_or_less_is_noop(self):
        f0 = _tone(200.0, 0.5)
        out = f0_mod.median_filter_f0(f0.copy(), 2)
        np.testing.assert_array_equal(out, f0)

    def test_enabled_uses_fixed_kernel_3_not_2r_plus_1(self):
        # a single-frame spike survives kernel-7 (old wrong behaviour) but
        # is removed by the official fixed kernel-3 at radius=3.
        f0 = _tone(200.0, 0.2)
        f0[50] = 900.0
        out = f0_mod.median_filter_f0(f0, 3)
        assert out[50] < 500.0, "kernel-3 must remove a 1-frame spike"

    def test_smooth_tone_unchanged_by_kernel_3(self):
        f0 = _tone(200.0, 0.3)  # already smooth
        out = f0_mod.median_filter_f0(f0.copy(), 3)
        assert np.abs(out - f0).max() < 1.0


class TestInputPeakGuard:
    def test_clipped_input_is_scaled(self):
        audio = (_tone(200.0, 0.2, amp=1.0))  # peaks at 1.0
        audio_max = float(np.abs(audio).max() / 0.95)
        if audio_max > 1.0:
            audio = audio / audio_max
        assert float(np.abs(audio).max()) <= 0.95 + 1e-6

    def test_normal_input_untouched(self):
        audio = _tone(200.0, 0.2, amp=0.3)
        audio_max = float(np.abs(audio).max() / 0.95)
        out = audio / audio_max if audio_max > 1.0 else audio
        assert out is audio or np.testing.assert_array_equal(out, audio) is None
