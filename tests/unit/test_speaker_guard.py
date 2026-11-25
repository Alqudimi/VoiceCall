"""Unit & API tests for strict real-time speaker isolation (SpeakerGuard)."""
from __future__ import annotations

import base64

import numpy as np
import pytest
from scipy import signal


def _synth_voice(
    f0: float,
    formants: list[tuple[float, float]],
    dur: float = 0.16,
    amp: float = 0.25,
    sr: int = 16_000,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    f0_t = f0 * (1.0 + 0.06 * np.sin(2 * np.pi * 2.5 * t))
    phase = 2 * np.pi * np.cumsum(f0_t) / sr
    src = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 15))
    src += 0.015 * rng.standard_normal(len(t))
    y = src.copy()
    for freq, bw in formants:
        b, a = signal.iirpeak(freq / (sr / 2), Q=freq / bw)
        y = signal.lfilter(b, a, y)
    return (y / (np.max(np.abs(y)) + 1e-9) * amp).astype(np.float32)


def _enrolled_speaker_a() -> np.ndarray:
    return np.concatenate([
        _synth_voice(130.0, [(700, 80), (1150, 100), (2450, 140), (3400, 180)], 0.5, seed=1),
        _synth_voice(135.0, [(500, 70), (1400, 100), (2500, 140), (3400, 180)], 0.5, seed=2),
        _synth_voice(140.0, [(320, 60), (2100, 110), (2900, 150), (3450, 180)], 0.5, seed=3),
    ])


class TestSpeakerGuard:
    def test_rejects_noise_claps_and_hum(self):
        from rvc_platform.engine.rvc.speaker_guard import SpeakerGuard

        guard = SpeakerGuard()
        rng = np.random.default_rng(42)
        white = (0.25 * rng.standard_normal(2560)).astype(np.float32)
        clap = np.zeros(2560, dtype=np.float32)
        clap[400:480] = (0.8 * rng.standard_normal(80)).astype(np.float32)
        hum = (0.25 * np.sin(2 * np.pi * 60 * np.arange(2560) / 16_000)).astype(np.float32)

        for sig in (white, clap, hum):
            ok, reason = guard.evaluate(sig, threshold_db=-36.0, strict=True)
            assert not ok
            assert reason in {"noise", "silence"}

    def test_rejects_distant_voice_via_proximity_gate(self):
        from rvc_platform.engine.rvc.speaker_guard import SpeakerGuard, build_speaker_profile

        guard = SpeakerGuard()
        guard.set_profile(build_speaker_profile(_enrolled_speaker_a()))

        close = _synth_voice(132.0, [(500, 70), (1400, 100), (2500, 140), (3400, 180)], 0.16, amp=0.25, seed=10)
        ok, reason = guard.evaluate(close, threshold_db=-36.0, strict=True)
        assert ok and reason == "ok"

        # Reset hold (simulate speaker paused), then a distant quiet voice arrives
        guard._hold_blocks = 0
        distant = _synth_voice(132.0, [(500, 70), (1400, 100), (2500, 140), (3400, 180)], 0.16, amp=0.015, seed=11)
        ok, reason = guard.evaluate(distant, threshold_db=-36.0, strict=True)
        assert not ok
        assert reason in {"far_voice", "silence"}

    def test_enrolled_speaker_accepted_other_speaker_rejected(self):
        from rvc_platform.engine.rvc.speaker_guard import SpeakerGuard, build_speaker_profile

        guard = SpeakerGuard()
        guard.set_profile(build_speaker_profile(_enrolled_speaker_a()))

        same_diff_vowel = _synth_voice(
            138.0, [(330, 60), (2050, 110), (2880, 150), (3450, 180)], 0.16, amp=0.25, seed=20
        )
        ok, reason = guard.evaluate(same_diff_vowel, threshold_db=-36.0, strict=True)
        assert ok and reason == "ok"

        # Simulate pause then another speaker (higher pitch + shifted formants)
        guard._hold_blocks = 0
        guard._recent_mfccs.clear()
        other_speaker = _synth_voice(
            245.0, [(600, 80), (1950, 130), (3300, 160), (4100, 200)], 0.16, amp=0.25, seed=21
        )
        ok, reason = guard.evaluate(other_speaker, threshold_db=-36.0, strict=True)
        assert not ok
        assert reason == "other_speaker"

    def test_auto_locks_and_blocks_self_echo(self):
        from rvc_platform.engine.rvc.speaker_guard import SpeakerGuard

        guard = SpeakerGuard()
        blk = _synth_voice(130.0, [(500, 70), (1400, 100), (2500, 140), (3400, 180)], 0.16, amp=0.25, seed=30)
        for _ in range(5):
            ok, _ = guard.evaluate(blk, threshold_db=-36.0, strict=True)
            assert ok
        assert guard.profile is not None

        # Record a distinctly different cloned voice output, then feed it back as mic input
        cloned = _synth_voice(210.0, [(650, 90), (1850, 120), (3150, 170)], 0.16, amp=0.25, seed=31)
        guard.record_output(cloned, 16_000)
        guard._hold_blocks = 0
        guard._recent_mfccs.clear()
        ok, reason = guard.evaluate(cloned, threshold_db=-36.0, strict=True)
        assert not ok
        assert reason in {"self_echo", "other_speaker"}

    def test_enroll_endpoint(self, project_env):
        from fastapi.testclient import TestClient
        from rvc_platform.api.app import create_app

        app = create_app()
        with TestClient(app) as client:
            pcm = (_enrolled_speaker_a() * 32767.0).astype(np.int16).tobytes()
            r = client.post("/api/realtime/enroll", json={"pcm_b64": base64.b64encode(pcm).decode()})
            assert r.status_code == 200
            body = r.json()
            assert body["ok"] is True
            assert len(body["profile"]["centroids"]) > 0

            # Noise-only audio must be rejected with 400
            noise = (0.01 * np.random.default_rng(1).standard_normal(16_000) * 32767.0).astype(np.int16).tobytes()
            bad = client.post("/api/realtime/enroll", json={"pcm_b64": base64.b64encode(noise).decode()})
            assert bad.status_code == 400
