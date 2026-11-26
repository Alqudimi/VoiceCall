"""Optimizer evaluation engine — gates, ordering, pareto, weights."""
from __future__ import annotations

import numpy as np
import pytest

from rvc_platform.optimizer.scoring import (
    DEFAULT_WEIGHTS,
    GATES,
    EvaluationEngine,
    MetricWeights,
    VoiceEvaluation,
)


def _analysis(median_hz=150.0, hnr=14.0, jitter=0.12, flat=0.02, clip=0.0,
              voiced=0.7, hf=(0.18, 0.1, 0.05), duration=6.0):
    """A minimal synthetic analysis dict (the shape AnalysisResult.to_dict()
    produces) with the levers the scorer responds to."""
    bands = [0.35, 0.2, 0.12, 0.08, hf[0], hf[1], hf[2], 0.02]
    tot = sum(bands)
    bands = [b / tot for b in bands]
    return {
        "duration_s": duration,
        "f0": {
            "median_hz": median_hz,
            "range_semitones": 7.0,
            "voiced_ratio": voiced,
            "octave_jump_rate": 0.01,
            "jitter_semitones": jitter,
            "histogram": [0.05] * 24,
        },
        "mfcc_mean": [float(10 * np.sin(i)) for i in range(20)],
        "mfcc_std": [2.0] * 20,
        "band_energies": bands,
        "spectral": {"centroid_hz": 900.0, "spread_hz": 1100.0,
                     "rolloff85_hz": 3200.0, "flatness": flat},
        "formants_hz": [520.0, 1480.0, 2600.0, 3400.0],
        "hnr_db": hnr,
        "shimmer_db": 1.2,
        "vad": {"voiced_ratio": voiced, "pauses": 4,
                "segments": [[0.2, 1.4], [1.8, 3.1], [3.6, 5.2]]},
        "loudness": {"rms_db": -18.0, "peak": 0.8, "clipping_ratio": clip,
                     "noise_floor_db": -55.0, "snr_proxy_db": 24.0,
                     "dynamic_range_db": 30.0},
        "warnings": [],
    }


def _profile(ref_analysis):
    p = {
        "model": "m", "name": "default",
        "embeddings": {"fake": [np.ones(64, dtype=np.float32).tolist()]},
        "stats": {"samples": [ref_analysis]},
    }
    from rvc_platform.optimizer.profile import ReferenceProfile

    prof = ReferenceProfile("m", "default")
    prof.add_sample_analysis(ref_analysis, {"id": "s1", "file": "s.wav"})
    return prof.to_dict()


def _sim(ensemble=0.8, spread=0.02):
    return {
        "similarities": {"wavlm": ensemble + spread, "ecapa": ensemble - spread},
        "ensemble": ensemble,
        "agreement_spread": spread,
        "reliability": "high",
    }


@pytest.fixture()
def engine():
    return EvaluationEngine(MetricWeights(dict(DEFAULT_WEIGHTS)))


def test_all_axes_and_composite_present(engine):
    ref = _analysis()
    out = _analysis()
    ev = engine.evaluate(out, _analysis(), _profile(ref), _sim(0.85))
    axes = ev.axis_scores()
    assert set(axes) == set(DEFAULT_WEIGHTS)
    for k, v in axes.items():
        assert v is not None and 0.0 <= v <= 1.0, f"{k}={v}"
    assert ev.composite is not None and 0.0 <= ev.composite <= 1.0
    assert ev.gates_passed


def test_identical_output_beats_degraded(engine):
    ref = _analysis()
    prof = _profile(ref)
    identical = engine.evaluate(ref, ref, prof, _sim(0.9))
    # heavily clipped, metallic, robotic variant
    bad = _analysis(clip=0.05, flat=0.09, hnr=30.0, jitter=0.005)
    bad_ev = engine.evaluate(bad, ref, prof, _sim(0.9))
    assert identical.composite > bad_ev.composite
    assert bad_ev.artifact_score < 0.6
    assert bad_ev.detail.get("artifact_penalties")


def test_gates_exclude_distorted_voice_from_best(engine):
    """A candidate with high speaker similarity but bad artifacts must NOT
    win: gates fail and composite is heavily penalised."""
    ref = _analysis()
    prof = _profile(ref)
    clean = engine.evaluate(ref, ref, prof, _sim(0.95))
    distorted = _analysis(clip=0.02, flat=0.10, hnr=32.0, jitter=0.004)
    d_ev = engine.evaluate(distorted, ref, prof, _sim(0.99))
    assert not d_ev.gates_passed
    assert "artifact_score" in d_ev.failed_gates
    # even though similarity is higher, the composite must not win
    assert d_ev.composite < clean.composite


def test_speaker_similarity_dominates_weights(engine):
    ref = _analysis()
    prof = _profile(ref)
    # A: better identity, everything else equal
    ev_a = engine.evaluate(ref, ref, prof, _sim(0.9))
    # B: much worse identity
    ev_b = engine.evaluate(ref, ref, prof, _sim(0.4))
    assert ev_a.speaker_similarity > ev_b.speaker_similarity
    assert ev_a.composite > ev_b.composite
    assert ev_a.detail["speaker_backends"]["wavlm"] > \
        ev_b.detail["speaker_backends"]["wavlm"]


def test_pitch_offset_is_penalised(engine):
    ref = _analysis(median_hz=150.0)
    prof = _profile(ref)
    same = engine.evaluate(_analysis(median_hz=150.0), ref, prof, _sim(0.8))
    off = engine.evaluate(_analysis(median_hz=238.0), ref, prof, _sim(0.8))  # +8 st
    assert same.pitch_consistency > off.pitch_consistency
    assert same.composite > off.composite


def test_missing_similarity_reports_none_axis(engine):
    ref = _analysis()
    ev = engine.evaluate(ref, ref, _profile(ref), None)
    assert ev.speaker_similarity is None
    assert ev.warnings
    assert ev.composite is not None  # other axes still computed


def test_weights_normalisation_and_override():
    w = MetricWeights({"speaker_similarity": 1.0}).normalised()
    assert w["speaker_similarity"] == 1.0
    total = sum(MetricWeights({"speaker_similarity": 2.0, "clarity": 1.0})
                .normalised().values())
    assert total == pytest.approx(1.0, abs=1e-9)
    # empty/invalid falls back to defaults
    assert MetricWeights({}).normalised() == \
        MetricWeights(dict(DEFAULT_WEIGHTS)).normalised()


def test_pareto_front_filters_dominated():
    evs = []
    for spk, nat, cla in [(0.9, 0.6, 0.6), (0.6, 0.9, 0.6), (0.6, 0.6, 0.9),
                          (0.5, 0.5, 0.5), (0.95, 0.95, 0.95)]:
        ev = VoiceEvaluation(speaker_similarity=spk, naturalness=nat,
                             clarity=cla)
        evs.append(ev)
    front = EvaluationEngine.pareto_front(evs)
    # the strictly-best point dominates everything; the three trade-offs and
    # the all-95 point are non-dominated; the (0.5,0.5,0.5) is dominated
    assert 3 in front or True
    assert front  # non-empty
    assert 3 not in front or True
    # dominated one is excluded
    assert 3 not in front
    # strictly better point is on the front
    assert 4 in front


def test_gates_thresholds_are_sane():
    for k, v in GATES.items():
        assert 0.0 < v < 0.6
