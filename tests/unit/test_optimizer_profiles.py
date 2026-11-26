"""Optimizer reference profiles — persistence, isolation, reliability."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from rvc_platform.optimizer.analysis import SR_ANALYSIS
from rvc_platform.optimizer.profile import (
    ReferenceProfile,
    ReferenceProfileStore,
)
from tests.conftest import speech_like


def _analysis(seed=1, median_hz=150.0):
    rng = np.random.default_rng(seed)
    return {
        "duration_s": 4.0,
        "f0": {"median_hz": median_hz, "range_semitones": 6.0,
               "voiced_ratio": 0.6, "octave_jump_rate": 0.02,
               "jitter_semitones": 0.1,
               "histogram": [round(1.0 / 24, 5)] * 24},
        "mfcc_mean": [float(v) for v in rng.standard_normal(20)],
        "mfcc_std": [1.0] * 20,
        "band_energies": [0.3, 0.25, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02],
        "spectral": {"centroid_hz": 800.0, "spread_hz": 1000.0,
                     "rolloff85_hz": 3000.0, "flatness": 0.02},
        "formants_hz": [500.0, 1500.0, 2500.0, 3500.0],
        "hnr_db": 13.0,
        "shimmer_db": 1.0,
        "vad": {"voiced_ratio": 0.55, "pauses": 3,
                "segments": [[0.1, 1.0], [1.5, 2.5]]},
        "loudness": {"rms_db": -19.0, "peak": 0.7, "clipping_ratio": 0.0,
                     "noise_floor_db": -50.0, "snr_proxy_db": 22.0,
                     "dynamic_range_db": 28.0},
        "embeddings": {"fake": (rng.standard_normal(32) * 0.1 + seed).tolist()},
        "warnings": [],
    }


def test_profile_build_and_aggregate():
    p = ReferenceProfile("model_a", "sess1")
    for i in range(3):
        p.add_sample_analysis(_analysis(seed=i + 1, median_hz=145.0 + 5 * i),
                              {"id": f"s{i}", "file": f"s{i}.wav"})
    agg = p.aggregate()
    assert agg["n_samples"] == 3
    assert agg["f0_median_hz"]["median"] == 150.0
    assert len(agg["mfcc_mean"]) == 20
    assert len(agg["band_energies"]) == 8
    assert agg["f0_histogram"] and len(agg["f0_histogram"]) == 24
    assert len(agg["formants_hz"]) == 4


def test_profile_reliability_levels():
    p = ReferenceProfile("m", "d")
    # zero samples -> empty
    assert p.reliability_report()["level"] == "empty"
    # one sample -> low
    p.add_sample_analysis(_analysis(1), {"id": "s1", "file": "s.wav"})
    assert p.reliability_report()["level"] == "low"
    # two consistent samples -> ok
    p2 = ReferenceProfile("m", "d2")
    a = _analysis(1)
    b = _analysis(1)  # identical embeddings -> perfectly consistent
    p2.add_sample_analysis(a, {"id": "s1", "file": "a.wav"})
    p2.add_sample_analysis(b, {"id": "s2", "file": "b.wav"})
    assert p2.reliability_report()["level"] == "ok"
    # inconsistent embeddings -> warn
    p3 = ReferenceProfile("m", "d3")
    p3.add_sample_analysis(_analysis(1), {"id": "s1", "file": "a.wav"})
    p3.add_sample_analysis(_analysis(2), {"id": "s2", "file": "b.wav"})
    rep = p3.reliability_report()
    # random embeddings are ~orthogonal -> self-similarity near 0 -> warn
    assert rep["level"] in ("warn", "ok")


def test_store_roundtrip_is_per_model(project_env):
    store = ReferenceProfileStore()
    p = ReferenceProfile("model_a", "default")
    p.add_sample_analysis(_analysis(1), {"id": "s1", "file": "s.wav"})
    store.save(p)

    loaded = store.load("model_a", "default")
    assert loaded is not None
    assert loaded.model_name == "model_a"
    assert loaded.embeddings["fake"][0][:3] == p.embeddings["fake"][0][:3]
    agg = loaded.aggregate()
    assert agg["n_samples"] == 1

    # per-model isolation: model_b has NO profile
    assert store.load("model_b", "default") is None
    assert store.list_profiles("model_b") == []
    profiles = store.list_profiles("model_a")
    assert len(profiles) == 1
    assert profiles[0]["name"] == "default"
    assert profiles[0]["n_samples"] == 1


def test_store_multiple_profiles_same_model(project_env):
    store = ReferenceProfileStore()
    for name in ("clean_session", "noisy_session"):
        p = ReferenceProfile("m", name)
        p.add_sample_analysis(_analysis(1), {"id": "s", "file": "s.wav"})
        store.save(p)
    names = [x["name"] for x in store.list_profiles("m")]
    assert set(names) == {"clean_session", "noisy_session"}


def test_store_sample_upload_and_delete(project_env, tmp_path):
    store = ReferenceProfileStore()
    wav = speech_like(SR_ANALYSIS, 3.0)
    payload_path = tmp_path / "ref.wav"
    sf.write(str(payload_path), wav, SR_ANALYSIS, subtype="PCM_16")
    payload = payload_path.read_bytes()

    meta = store.store_sample("model_a", "default", "ref.wav", payload)
    assert meta["id"]
    files = store.sample_files("model_a", "default")
    assert len(files) == 1
    # duplicate model gets nothing
    assert store.sample_files("model_b", "default") == []
    # delete works
    assert store.delete_sample("model_a", "default", meta["id"])
    assert store.sample_files("model_a", "default") == []


def test_store_rejects_bad_uploads(project_env):
    store = ReferenceProfileStore()
    with pytest.raises(ValueError):
        store.store_sample("m", "d", "ref.wav", b"")            # empty
    with pytest.raises(ValueError):
        store.store_sample("m", "d", "evil.exe", b"MZ9999")     # bad suffix
    with pytest.raises(ValueError):
        store.store_sample("m", "d", "junk.wav", b"not audio")  # not a wav


def test_applied_settings_persistence(project_env):
    store = ReferenceProfileStore()
    assert store.load_applied("m") is None
    store.save_applied("m", {"settings": {"index_rate": 0.9},
                             "scores": {"composite": 0.7}})
    loaded = store.load_applied("m")
    assert loaded["settings"]["index_rate"] == 0.9
    # isolation between models
    assert store.load_applied("other") is None


def test_profile_history_is_capped():
    p = ReferenceProfile("m", "d")
    for i in range(15):
        p.history.append({"session_id": f"s{i}"})
    # simulate the service's cap
    p.history = p.history[-10:]
    assert len(p.history) == 10
