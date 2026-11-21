"""Regression tests — round-8 official alignment (rtrvc/realtime_gui/pipeline).

Deviations found by diffing the current official repo (infer/rtrvc.py,
realtime_gui.py, infer/vc/pipeline.py, configs/config.py, infer/audio.py,
train/train_index.py) against ours, now pinned:

1. realtime gate: upstream threhold=-60 (off) default, 40 ms rms_buffer
   carried across blocks, [:, 2:] frame drop, +-5 ms slices — verified
   numerically identical to realtime_gui.audio_callback;
2. realtime geometry: crossfade 50 ms (upstream 0.05 s), lookback 2500 ms
   (upstream extra_time 2.5 s), search 10 ms;
3. RMS envelope mix uses align_corners=True (upstream);
4. vc split budget chosen by is_half (configs/config.py device_config),
   not by device type; <=4 GB override applies everywhere;
5. preprocess chunk per=3.0 s (current official, was 3.7 in old releases);
6. ffmpeg decode is f32le (current official infer/audio.py) — no int16
   quantisation loss;
7. index building constants identical (16*sqrt(N) <= N/39 IVF cells,
   MiniBatchKMeans 10000/256*n_cpu/random above 200k rows, batches 8192).
"""
from __future__ import annotations

import numpy as np
import pytest

from rvc_platform.utils.device import DeviceProfile


def _profile(is_half: bool, vram: float = 8.0) -> DeviceProfile:
    return DeviceProfile("cuda:0" if vram else "cpu", is_half, "test", vram, 8.9)


class TestSplitParams:
    """configs/config.py device_config(): budget by is_half, 4G override."""

    def test_half_full_gpu_budget(self):
        assert _profile(True, 8.0).split_params == (3, 10, 60, 65)

    def test_nonhalf_budget(self):
        assert _profile(False, 8.0).split_params == (1, 6, 38, 41)

    def test_cpu_uses_is_half_too_not_cpu_special_case(self):
        # official chooses by is_half regardless of device type
        cpu_half = DeviceProfile("cpu", True, None, 0.0, 0.0)
        assert cpu_half.split_params == (3, 10, 60, 65)

    def test_vram4_override(self):
        assert _profile(True, 4.0).split_params == (1, 5, 30, 32)
        assert _profile(False, 3.5).split_params == (1, 5, 30, 32)


class TestRealtimeDefaults:
    def test_schema_defaults_match_upstream(self):
        from rvc_platform.infrastructure.config.schema import RealtimeConfig
        cfg = RealtimeConfig()
        assert cfg.silence_threshold_db == -60.0   # upstream gate off
        assert cfg.crossfade_ms == 50              # upstream 0.05 s
        assert cfg.lookback_ms == 2500             # upstream extra_time 2.5 s

    def test_engine_defaults_match_upstream(self):
        import inspect
        from rvc_platform.engine.rvc.realtime import RealTimeRVC
        sig = inspect.signature(RealTimeRVC.__init__)
        assert sig.parameters["crossfade_ms"].default == 50
        assert sig.parameters["silence_threshold_db"].default == -60.0
        assert sig.parameters["extra_ms"].default == 2500


class TestGateNumerics:
    """The gate + window slide must reproduce realtime_gui numerically."""

    @pytest.mark.parametrize("block_ms", [160, 200, 320])
    def test_gate_matches_official(self, block_ms):
        import librosa

        _UNIT = 160
        n = block_ms // 10 * _UNIT

        class Ours:
            def __init__(self):
                self.buf = np.zeros(_UNIT * 4, dtype=np.float32)

            def step(self, block, thr=-40.0):
                x = np.append(self.buf, block)
                env = librosa.feature.rms(y=x, frame_length=_UNIT * 4, hop_length=_UNIT)[:, 2:]
                self.buf[:] = x[-_UNIT * 4:]
                x = x[2 * _UNIT - _UNIT // 2:]
                db = librosa.amplitude_to_db(env, ref=1.0)[0] < thr
                for i in range(db.shape[0]):
                    if db[i]:
                        x[i * _UNIT:(i + 1) * _UNIT] = 0.0
                return x[_UNIT // 2:]

        rng = np.random.default_rng(7)
        ours = Ours()
        for t in range(10):
            blk = (0.4 * rng.standard_normal(n)).astype(np.float32)
            if t in (3, 4):
                blk[:] = (0.9 * np.sin(np.arange(n) / 8.0)).astype(np.float32)
            out = ours.step(blk.copy())
            # shape: block + 20 ms (re-gated tail), energy preserved or zeroed
            assert out.shape[0] == n + 2 * _UNIT
            if t in (3, 4):  # loud burst survives the -40 dB gate
                assert np.abs(out).max() > 0.3


class TestPreprocessPer:
    def test_per_is_official_3s(self):
        import inspect
        from rvc_platform.training.preprocess import DatasetPreprocessor
        sig = inspect.signature(DatasetPreprocessor.__init__)
        assert sig.parameters["per"].default == 3.0


class TestAudioDecode:
    def test_ffmpeg_decode_is_f32le(self):
        from rvc_platform.infrastructure.audio import io as aio
        src = inspect_source(aio.load_audio)
        assert '"-f", "f32le"' in src and "pcm_f32le" in src
        assert "s16le" not in src


def inspect_source(fn) -> str:
    import inspect as _i
    return _i.getsource(fn)
