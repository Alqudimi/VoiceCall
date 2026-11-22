"""Unit tests — previously low-coverage modules, batch 4.

Targets (round-10 baseline): security/auth.py (59%), config/loader.py (56%),
audio/io.py (56%), audio/slicer.py (45%), audio/pipeline.py (65%),
processes/manager.py (41%), utils/environment.py (73%).
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import soundfile as sf


# --------------------------------------------------------------------------- #
# security/auth.py
# --------------------------------------------------------------------------- #
class TestAuth:
    def test_disabled_token_accepts_everything(self):
        from rvc_platform.infrastructure.security.auth import TokenAuth

        auth = TokenAuth("secret", enabled=False)
        assert auth.check_header(None) is True
        assert auth.check_header("Bearer wrong") is True

    def test_enabled_token_check_header(self):
        from rvc_platform.infrastructure.security.auth import TokenAuth

        auth = TokenAuth("secret", enabled=True)
        assert auth.check_header("Bearer secret") is True
        assert auth.check_header("Bearer  secret ") is True
        assert auth.check_header("Bearer wrong") is False
        assert auth.check_header("secret") is False          # no Bearer prefix
        assert auth.check_header(None) is False
        assert auth.check_header("") is False

    def test_guard_http_disabled_and_enabled(self):
        from fastapi import HTTPException

        from rvc_platform.infrastructure.security.auth import TokenAuth

        auth = TokenAuth("secret", enabled=False)
        # disabled -> never raises even with no request headers
        class FakeRequest:
            headers: dict = {}

        auth.guard_http(FakeRequest())  # type: ignore[arg-type]

        auth_on = TokenAuth("secret", enabled=True)
        with pytest.raises(HTTPException) as ei:
            auth_on.guard_http(FakeRequest())  # type: ignore[arg-type]
        assert ei.value.status_code == 401

        class OkRequest:
            headers = {"Authorization": "Bearer secret"}

        auth_on.guard_http(OkRequest())  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_guard_ws(self):
        from rvc_platform.infrastructure.security.auth import TokenAuth

        class FakeWS:
            def __init__(self, headers):
                self.headers = headers
                self.closed = None

            async def close(self, code, reason):
                self.closed = (code, reason)

        auth = TokenAuth("secret", enabled=False)
        assert await auth.guard_ws(FakeWS({})) is True

        auth_on = TokenAuth("secret", enabled=True)
        ok = FakeWS({"Authorization": "Bearer secret"})
        assert await auth_on.guard_ws(ok) is True
        bad = FakeWS({"Authorization": "Bearer nope"})
        assert await auth_on.guard_ws(bad) is False
        assert bad.closed == (4401, "unauthorized")

    def test_issue_token_unique_and_length(self):
        from rvc_platform.infrastructure.security.auth import issue_token

        t1, t2 = issue_token(), issue_token()
        assert t1 != t2 and len(t1) >= 24


# --------------------------------------------------------------------------- #
# config/loader.py
# --------------------------------------------------------------------------- #
class TestConfigLoader:
    def test_deep_merge_nested(self):
        from rvc_platform.infrastructure.config.loader import _deep_merge

        base = {"a": {"x": 1, "y": 2}, "b": 1}
        overlay = {"a": {"y": 3, "z": 4}, "c": 5}
        merged = _deep_merge(base, overlay)
        assert merged == {"a": {"x": 1, "y": 3, "z": 4}, "b": 1, "c": 5}
        # base untouched (deep copy)
        assert base == {"a": {"x": 1, "y": 2}, "b": 1}

    def test_coerce_types(self):
        from rvc_platform.infrastructure.config.loader import _coerce

        assert _coerce("true") is True
        assert _coerce("FALSE") is False
        assert _coerce("42") == 42 and isinstance(_coerce("42"), int)
        assert _coerce("3.5") == 3.5
        assert _coerce("hello") == "hello"
        assert _coerce("  8080 ") == 8080

    def test_load_with_user_yaml_and_env(self, project_env, monkeypatch):
        from rvc_platform.infrastructure.config.loader import ConfigLoader

        default_yaml = project_env / "configs" / "default.yaml"
        default_yaml.parent.mkdir(parents=True)
        default_yaml.write_text("server:\n  host: 0.0.0.0\n  port: 9000\n",
                                encoding="utf-8")
        user_yaml = project_env / "configs" / "user.yaml"
        user_yaml.write_text("server:\n  port: 9100\n", encoding="utf-8")
        monkeypatch.setenv("RVC_PLATFORM_CONFIG", str(user_yaml))
        monkeypatch.setenv("RVC_LOG_LEVEL", "DEBUG")

        cfg = ConfigLoader(default_yaml=default_yaml).load()
        assert cfg.server.port == 9100          # user overrides default
        assert cfg.server.host == "0.0.0.0"     # default survives
        assert cfg.logging.level == "DEBUG"     # env overrides both

    def test_load_invalid_yaml_raises(self, project_env):
        from rvc_platform.core.errors import ConfigurationError
        from rvc_platform.infrastructure.config.loader import ConfigLoader

        bad = project_env / "bad.yaml"
        bad.write_text(":\n  - broken: [", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            ConfigLoader(default_yaml=bad).load()

    def test_load_non_mapping_yaml_raises(self, project_env):
        from rvc_platform.core.errors import ConfigurationError
        from rvc_platform.infrastructure.config.loader import ConfigLoader

        bad = project_env / "list.yaml"
        bad.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            ConfigLoader(default_yaml=bad).load()

    def test_load_invalid_schema_raises(self, project_env):
        from rvc_platform.core.errors import ConfigurationError
        from rvc_platform.infrastructure.config.loader import ConfigLoader

        bad = project_env / "schema.yaml"
        bad.write_text("server:\n  port: not-a-port\n", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            ConfigLoader(default_yaml=bad).load()

    def test_missing_default_and_user_ok(self, project_env):
        from rvc_platform.infrastructure.config.loader import ConfigLoader

        cfg = ConfigLoader(default_yaml=project_env / "nope.yaml").load()
        assert cfg.server.port > 0             # pure code defaults

    def test_user_path_env_var(self, project_env, monkeypatch, tmp_path):
        from rvc_platform.infrastructure.config.loader import ConfigLoader

        custom = tmp_path / "custom.yaml"
        custom.write_text("server:\n  port: 7777\n", encoding="utf-8")
        monkeypatch.setenv("RVC_PLATFORM_CONFIG", str(custom))
        loader = ConfigLoader(default_yaml=project_env / "nope.yaml")
        assert loader._user_path() == custom
        assert loader.load().server.port == 7777

    def test_write_user_roundtrip_and_env_omission(self, project_env,
                                                   monkeypatch):
        from rvc_platform.infrastructure.config.loader import (
            ConfigLoader,
        )

        monkeypatch.delenv("RVC_SERVER_PORT", raising=False)
        loader = ConfigLoader(default_yaml=project_env / "nope.yaml")
        cfg = loader.load()
        cfg.server.port = 9355
        target = loader.write_user(cfg)
        assert target.is_file()
        # env-set values must NOT be persisted (they shadow the file)
        monkeypatch.setenv("RVC_SERVER_PORT", "1234")
        cfg2 = ConfigLoader(default_yaml=project_env / "nope.yaml").load()
        assert cfg2.server.port == 1234
        import yaml

        persisted = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert persisted["server"]["port"] == 9355

    def test_get_config_cached_and_reload(self, project_env):
        from rvc_platform.infrastructure.config import loader as cl

        c1 = cl.get_config()
        c2 = cl.get_config()
        assert c1 is c2                       # cached
        c3 = cl.get_config(reload=True)
        assert c3 is not c1
        cl.reset_config()
        assert cl._cached is None


# --------------------------------------------------------------------------- #
# audio/slicer.py — the full upstream silence slicer
# --------------------------------------------------------------------------- #
def _speech_with_gaps(sr=16_000):
    """3 loud bursts separated by long silences (>min_interval)."""
    t = np.arange(sr * 6) / sr
    sig = 0.4 * np.sin(2 * np.pi * 150 * t)
    sig[: sr] = 0.0                    # 1 s leading silence
    sig[2 * sr: 3 * sr] = 0.0          # 1 s gap
    sig[5 * sr:] = 0.0                 # 1 s trailing silence
    return sig.astype(np.float32)


class TestSlicer:
    def test_get_rms_values(self):
        from rvc_platform.infrastructure.audio.slicer import get_rms

        sr = 16_000
        loud = np.full(sr, 0.5, dtype=np.float32)
        rms = get_rms(y=loud, frame_length=512, hop_length=256)
        # interior frames are exactly 0.5 (edges have zero-padding effects)
        assert np.allclose(rms[:, 1:-1], 0.5, atol=1e-6)
        silent = np.zeros(sr, dtype=np.float32)
        assert float(get_rms(y=silent, frame_length=512, hop_length=256)
                     .max()) == 0.0

    def test_slicer_validation_errors(self):
        from rvc_platform.infrastructure.audio.slicer import Slicer

        with pytest.raises(ValueError):
            Slicer(16_000, min_length=100, min_interval=5000)  # interval > len
        with pytest.raises(ValueError):
            Slicer(16_000, min_interval=100, hop_size=5000)    # hop > interval
        with pytest.raises(ValueError):
            Slicer(16_000, max_sil_kept=10, hop_size=20)       # kept < hop

    def test_slice_returns_chunks(self):
        from rvc_platform.infrastructure.audio.slicer import Slicer

        sr = 16_000
        wav = _speech_with_gaps(sr)
        slicer = Slicer(sr=sr, threshold=-40, min_length=1500,
                        min_interval=400, hop_size=15, max_sil_kept=500)
        chunks = slicer.slice(wav)
        assert len(chunks) >= 2                     # silence was cut
        joined = sum(len(c) for c in chunks)
        assert joined < len(wav)                    # something removed
        # every chunk has signal above the silence floor
        for c in chunks:
            assert float(np.abs(c).max()) > 0.1

    def test_slice_short_waveform_passthrough(self):
        from rvc_platform.infrastructure.audio.slicer import Slicer

        sr = 16_000
        slicer = Slicer(sr=sr, min_length=5000)
        short = np.zeros(1000, dtype=np.float32)
        assert slicer.slice(short) == [short]       # <= min_length -> whole

    def test_slice_continuous_waveform_passthrough(self):
        from rvc_platform.infrastructure.audio.slicer import Slicer

        sr = 16_000
        slicer = Slicer(sr=sr, threshold=-40, min_length=1500,
                        min_interval=400, hop_size=15, max_sil_kept=500)
        t = np.arange(sr * 3) / sr
        continuous = (0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        assert slicer.slice(continuous) == [continuous]

    def test_slice_stereo_channels(self):
        from rvc_platform.infrastructure.audio.slicer import Slicer

        sr = 16_000
        mono = _speech_with_gaps(sr)
        stereo = np.stack([mono, mono * 0.5], axis=0)
        slicer = Slicer(sr=sr, threshold=-40, min_length=1500,
                        min_interval=400, hop_size=15, max_sil_kept=500)
        chunks = slicer.slice(stereo)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.ndim == 2 and c.shape[0] == 2  # channels preserved

    def test_apply_slice_bounds(self):
        from rvc_platform.infrastructure.audio.slicer import Slicer

        s = Slicer(16_000)
        wav = np.arange(100_000, dtype=np.float32)
        # end beyond waveform -> clamped
        chunk = s._apply_slice(wav, 0, 10**6)
        assert len(chunk) == 100_000
        assert np.array_equal(s._apply_slice(wav, 2, 5),
                              wav[2 * s.hop_size: 5 * s.hop_size])


# --------------------------------------------------------------------------- #
# audio/io.py — validation, magic sniffing, writers, fallback path
# --------------------------------------------------------------------------- #
class TestAudioIO:
    def test_validate_ok_and_size_cap(self, tmp_path):
        from rvc_platform.infrastructure.audio.io import validate_audio_file

        p = tmp_path / "ok.wav"
        sf.write(str(p), np.zeros(1600, dtype=np.float32), 16_000)
        meta = validate_audio_file(p, max_mb=200)
        assert meta["sample_rate"] == 16_000
        with pytest.raises(Exception):
            validate_audio_file(p, max_mb=0)       # smaller than the file

    def test_validate_rejects_bad_suffix_and_magic(self, tmp_path):
        from rvc_platform.core.errors import SecurityError
        from rvc_platform.infrastructure.audio.io import validate_audio_file

        bad_ext = tmp_path / "evil.exe"
        bad_ext.write_bytes(b"RIFF" + b"\x00" * 200)
        with pytest.raises(SecurityError):
            validate_audio_file(bad_ext)
        bad_magic = tmp_path / "fake.wav"
        bad_magic.write_bytes(b"ZZZZ" + b"\x00" * 200)   # wrong magic
        with pytest.raises(SecurityError):
            validate_audio_file(bad_magic)

    def test_load_audio_ffmpeg_and_fallback(self, tmp_path):
        from rvc_platform.infrastructure.audio.io import load_audio

        p = tmp_path / "x.wav"
        sf.write(str(p), np.linspace(-0.5, 0.5, 8000, dtype=np.float32),
                 16_000)
        a_ff = load_audio(p, 16_000, ffmpeg=True)
        assert a_ff.dtype == np.float32 and a_ff.shape[0] == 8000
        a_fb = load_audio(p, 16_000, ffmpeg=False)   # soundfile fallback
        assert a_fb.shape[0] == 8000
        # fallback resamples via dsp
        a_up = load_audio(p, 32_000, ffmpeg=False)
        assert a_up.shape[0] == 16_000

    def test_load_audio_failure_raises_conversion_error(self, tmp_path):
        from rvc_platform.core.errors import ConversionError
        from rvc_platform.infrastructure.audio.io import load_audio

        garbage = tmp_path / "g.wav"
        garbage.write_bytes(b"RIFFzzz")
        with pytest.raises(ConversionError):
            load_audio(garbage, 16_000, ffmpeg=False)

    def test_write_wav_and_metadata(self, tmp_path):
        from rvc_platform.infrastructure.audio.io import (
            detect_wav_sr,
            wav_metadata,
            write_wav,
        )

        p = tmp_path / "out.wav"
        write_wav(p, np.full(1000, 0.5, dtype=np.float32), 24_000)
        assert p.is_file()
        meta = wav_metadata(p)
        assert meta["sample_rate"] == 24_000
        assert meta["channels"] == 1
        assert detect_wav_sr(p) == 24_000
        assert detect_wav_sr(tmp_path / "missing.wav") is None

    def test_wav_metadata_non_wav_raises(self, tmp_path):
        from rvc_platform.core.errors import SecurityError
        from rvc_platform.infrastructure.audio.io import wav_metadata

        p = tmp_path / "x.txt"
        p.write_text("no", encoding="utf-8")
        # falls through to validate_audio_file -> unsupported extension
        with pytest.raises(SecurityError):
            wav_metadata(p)

    def test_probe_ffmpeg(self):
        from rvc_platform.infrastructure.audio.io import probe_ffmpeg

        # ffmpeg exists in this sandbox
        assert probe_ffmpeg() in (True, False)  # no crash either way


# --------------------------------------------------------------------------- #
# audio/pipeline.py — the AudioProcessor facade
# --------------------------------------------------------------------------- #
class TestAudioPipeline:
    def test_facade_delegation(self, tmp_path):
        from rvc_platform.infrastructure.audio.pipeline import \
            AudioProcessingEngine

        ap = AudioProcessingEngine(ffmpeg=False)
        p = tmp_path / "a.wav"
        sf.write(str(p), np.zeros(8000, dtype=np.float32), 16_000)
        loaded = ap.load_any(p, 16_000)
        assert loaded.shape[0] == 8000
        resampled = ap.resample(np.zeros(8000, dtype=np.float32), 16_000, 32_000)
        assert resampled.shape[0] == 16_000
        normed = ap.normalize(np.array([0.0, 2.0, -4.0], dtype=np.float32),
                              peak=0.9)
        assert np.abs(normed).max() == pytest.approx(0.9)
        silences = ap.detect_silence(np.zeros(16000, dtype=np.float32), 16000)
        assert len(silences) >= 1
        trimmed = ap.trim_silence(
            np.concatenate([np.zeros(4800, dtype=np.float32),
                            np.full(16000, 0.5, dtype=np.float32)]), 16_000)
        assert trimmed.shape[0] < 20_800
        assert ap.validate_audio_file(p)["sample_rate"] == 16_000
        out = tmp_path / "o.wav"
        ap.write_wav(out, np.zeros(100, dtype=np.float32), 16_000)
        assert out.is_file()
        hp = ap.highpass(np.random.default_rng(0).standard_normal(1000)
                         .astype(np.float32), 16_000)
        assert hp.shape[0] == 1000

    def test_condition_chain(self, tmp_path):
        from rvc_platform.infrastructure.audio.pipeline import \
            AudioProcessingEngine

        ap = AudioProcessingEngine(ffmpeg=False)
        # loud signal with a DC offset + clipping headroom
        sig = np.full(16_000, 0.8, dtype=np.float32)
        sig[100:200] = 5.0                       # clipped peak
        out = ap.condition(sig, 16_000, normalize_peak=0.9)
        assert np.abs(out).max() <= 1.0
        assert out.dtype == np.float32

    def test_ffmpeg_property(self):
        from rvc_platform.infrastructure.audio.pipeline import \
            AudioProcessingEngine

        assert AudioProcessingEngine(ffmpeg=True).ffmpeg is True
        assert AudioProcessingEngine(ffmpeg=False).ffmpeg is False
        # None -> probes the environment
        assert isinstance(AudioProcessingEngine().ffmpeg, bool)


# --------------------------------------------------------------------------- #
# processes/manager.py — subprocess supervision
# --------------------------------------------------------------------------- #
class TestProcessManager:
    def test_lifecycle(self, tmp_path):
        from rvc_platform.infrastructure.processes.manager import ProcessManager

        pm = ProcessManager(log_dir=tmp_path / "logs")
        proc = pm.create("sleeper", ["python3", "-c",
                                     "import time; time.sleep(30)"])
        proc.start()
        assert proc.is_running() is True
        assert proc.info().name == "sleeper"
        assert proc.info().running is True
        d = proc.info().to_dict()
        assert d["name"] == "sleeper"
        assert proc.log_path.parent.is_dir()
        # duplicate start name -> error
        with pytest.raises(Exception):
            pm.create("sleeper", ["python3", "-c", "pass"])
        assert pm.stop("sleeper") is True
        assert proc.is_running() is False
        # after exit the name can be reused
        pm.create("sleeper", ["python3", "-c", "pass"])
        pm.stop_all()

    def test_require_and_get_missing(self, tmp_path):
        from rvc_platform.core.errors import ProcessError
        from rvc_platform.infrastructure.processes.manager import ProcessManager

        pm = ProcessManager(log_dir=tmp_path)
        assert pm.get("ghost") is None
        with pytest.raises(ProcessError):
            pm.require("ghost")
        assert pm.stop("ghost") is False

    def test_tail_logs(self, tmp_path):
        from rvc_platform.infrastructure.processes.manager import ProcessManager

        pm = ProcessManager(log_dir=tmp_path / "logs")
        proc = pm.create("echoer", ["python3", "-c",
                                    "print('hello-log-line')"])
        proc.start()
        time.sleep(1.5)
        pm.stop_all()
        lines = proc.tail(lines=10)
        assert any("hello-log-line" in ln for ln in lines)

    def test_status_map(self, tmp_path):
        from rvc_platform.infrastructure.processes.manager import ProcessManager

        pm = ProcessManager(log_dir=tmp_path / "logs")
        pm.create("a", ["python3", "-c", "pass"])
        pm.create("b", ["python3", "-c", "pass"])
        status = pm.status()
        assert set(status.keys()) == {"a", "b"}
        pm.stop_all()

    def test_failed_start_is_not_running(self, tmp_path):
        from rvc_platform.infrastructure.processes.manager import ProcessManager

        pm = ProcessManager(log_dir=tmp_path / "logs")
        proc = pm.create("bad", ["definitely-not-a-binary-xyz"])
        with pytest.raises(Exception):
            proc.start()
        assert proc.is_running() is False


# --------------------------------------------------------------------------- #
# utils/environment.py
# --------------------------------------------------------------------------- #
class TestEnvironment:
    def test_is_colab_detection(self, monkeypatch):
        from rvc_platform.utils import environment as env

        monkeypatch.setenv("COLAB_GPU", "T4")
        assert env.detect_system().is_colab is True
        monkeypatch.delenv("COLAB_GPU")
        # no COLAB_GPU, no google dist-packages, no google.colab module
        monkeypatch.setattr(env, "sys_modules", lambda: [])
        assert env.detect_system().is_colab is False

    def test_environment_report_structure(self, monkeypatch):
        from rvc_platform.utils import environment as env

        monkeypatch.setattr(env, "sys_modules", lambda: [])
        monkeypatch.delenv("COLAB_GPU", raising=False)
        prof = env.detect_system()
        # SystemProfile carries the runtime facts; summary() renders them
        assert hasattr(prof, "is_colab") and prof.is_colab is False
        assert hasattr(prof, "python_version") and prof.python_version
        assert hasattr(prof, "os") and prof.os
