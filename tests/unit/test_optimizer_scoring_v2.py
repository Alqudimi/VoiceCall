"""Scoring v2 — content axis, true prosody correlations, strategies."""
from __future__ import annotations

import numpy as np
import pytest

from rvc_platform.optimizer.scoring import (
    DEFAULT_WEIGHTS,
    GATES,
    STRATEGIES,
    EvaluationEngine,
    MetricWeights,
    VoiceEvaluation,
)


def _analysis(median_hz=150.0, hnr=14.0, jitter=0.12, flat=0.02, clip=0.0,
              voiced=0.7, hf=(0.18, 0.1, 0.05), duration=6.0, env=None,
              contour=None, sib=0.04, peak_rate=2.5, **over):
    bands = [0.35, 0.2, 0.12, 0.08, hf[0], hf[1], hf[2], 0.02]
    tot = sum(bands)
    bands = [b / tot for b in bands]
    a = {
        "duration_s": duration,
        "f0": {
            "median_hz": median_hz, "range_semitones": 7.0,
            "voiced_ratio": voiced, "octave_jump_rate": 0.01,
            "jitter_semitones": jitter, "histogram": [0.05] * 24,
            "vibrato_depth_semitones": None, "vibrato_rate_hz": None,
            "vibrato_irregularity": 0.8,
        },
        "mfcc_mean": [float(10 * np.sin(i)) for i in range(20)],
        "mfcc_std": [2.0] * 20,
        "band_energies": bands,
        "sibilance_ratio": sib,
        "spectral": {"centroid_hz": 900.0, "spread_hz": 1100.0,
                     "rolloff85_hz": 3200.0, "flatness": flat},
        "formants_hz": [520.0, 1480.0, 2600.0, 3400.0],
        "hnr_db": hnr, "shimmer_db": 1.2,
        "vad": {"voiced_ratio": voiced, "pauses": 4,
                "segments": [[0.2, 1.4], [1.8, 3.1], [3.6, 5.2]],
                "breath_count": 0},
        "loudness": {"rms_db": -18.0, "peak": 0.8, "clipping_ratio": clip,
                     "noise_floor_db": -55.0, "snr_proxy_db": 24.0,
                     "dynamic_range_db": 30.0,
                     "envelope_peak_rate": peak_rate,
                     "discontinuity_rate": 0.01},
        "rms_envelope": env if env is not None else [
            round(0.5 + 0.5 * np.sin(i / 4.0), 3) for i in range(64)],
        "f0_contour": contour if contour is not None else [
            round(12.0 * np.log2(150.0 + 10 * i) - 100, 3)
            for i in range(32)],
        "warnings": [],
    }
    for k, v in over.items():
        if "." in k:
            top, sub = k.split(".")
            a[top] = {**a[top], sub: v}
        else:
            a[k] = v
    return a


def _profile(ref_analysis):
    from rvc_platform.optimizer.profile import ReferenceProfile

    prof = ReferenceProfile("m", "default")
    prof.add_sample_analysis(ref_analysis, {"id": "s1", "file": "s.wav"})
    return prof.to_dict()


def _sim(ensemble=0.8, spread=0.02, calibrated=None):
    out = {
        "similarities": {"wavlm": ensemble + spread,
                         "ecapa": ensemble - spread},
        "ensemble": ensemble,
        "agreement_spread": spread,
        "reliability": "high",
    }
    if calibrated is not None:
        out["calibration"] = {
            "raw": {"wavlm": ensemble, "ecapa": ensemble},
            "per_backend": {
                "wavlm": {"raw": ensemble, "margin": 1.0,
                          "calibrated": calibrated},
                "ecapa": {"raw": ensemble, "margin": 1.0,
                          "calibrated": calibrated},
            },
            "calibrated_ensemble": calibrated,
            "n_calibrated": 2,
            "notes": [],
        }
    return out


@pytest.fixture()
def engine():
    return EvaluationEngine(MetricWeights(dict(DEFAULT_WEIGHTS)))


def test_nine_axes_and_new_content_axis(engine):
    ref = _analysis()
    out = _analysis()
    ev = engine.evaluate(out, _analysis(), _profile(ref), _sim(0.85))
    axes = ev.axis_scores()
    assert "content_preservation" in axes
    assert set(axes) == set(DEFAULT_WEIGHTS)
    assert len(axes) == 9
    for k, v in axes.items():
        assert v is not None and 0.0 <= v <= 1.0, f"{k}={v}"


def test_calibrated_ensemble_used_when_available(engine):
    ref = _analysis()
    prof = _profile(ref)
    ev_raw = engine.evaluate(ref, ref, prof, _sim(0.8, calibrated=None))
    ev_cal = engine.evaluate(ref, ref, prof, _sim(0.8, calibrated=0.66))
    assert ev_cal.speaker_similarity == 0.66
    assert ev_cal.detail["speaker_calibrated_ensemble"] == 0.66
    assert ev_cal.detail["speaker_backends"]["wavlm"] == \
        pytest.approx(0.82, abs=1e-6)
    # raw ensemble is used when no calibration exists
    assert ev_raw.speaker_similarity == pytest.approx(0.8, abs=1e-6)
    assert any("RAW ensemble" in w for w in ev_raw.warnings)


def test_envelope_correlation_is_real_prosody(engine):
    """Identical envelope scores higher than a shuffled envelope."""
    ref = _analysis()
    prof = _profile(ref)
    env = [round(0.5 + 0.5 * np.sin(i / 3.0), 3) for i in range(64)]
    inp = _analysis(env=env)
    out_same = _analysis(env=env)
    shuffled = list(reversed(env))
    out_shuffled = _analysis(env=shuffled)
    ev_same = engine.evaluate(out_same, inp, prof, _sim(0.8))
    ev_shuffled = engine.evaluate(out_shuffled, inp, prof, _sim(0.8))
    assert ev_same.prosody_preservation > ev_shuffled.prosody_preservation


def test_f0_contour_correlation_is_real_prosody(engine):
    ref = _analysis()
    prof = _profile(ref)
    contour = [round(float(40 + 20 * np.sin(i / 3.0)), 3) for i in range(32)]
    inp = _analysis(contour=contour)
    out_same = _analysis(contour=contour)
    # same contour + constant offset (pitch shift) must correlate perfectly
    out_shifted = _analysis(contour=[c + 3.0 for c in contour])
    # a non-degenerate but UNCORRELATED contour (fast oscillation)
    out_flat = _analysis(contour=[c * (-1) ** (i % 2)
                                   for i, c in enumerate(contour)])
    ev_same = engine.evaluate(out_shifted, inp, prof, _sim(0.8))
    ev_flat = engine.evaluate(out_flat, inp, prof, _sim(0.8))
    assert ev_same.prosody_preservation > ev_flat.prosody_preservation
    assert ev_same.prosody_preservation == \
        engine.evaluate(out_same, inp, prof, _sim(0.8)).prosody_preservation


def test_content_axis_responds_to_duration_and_silence_changes(engine):
    ref = _analysis()
    prof = _profile(ref)
    inp = _analysis(duration=6.0)
    stretched = _analysis(duration=9.0)                    # 50% longer
    chopped = _analysis(duration=6.0, voiced=0.3)          # half speech gone
    ev_ok = engine.evaluate(inp, inp, prof, _sim(0.8))
    ev_stretch = engine.evaluate(stretched, inp, prof, _sim(0.8))
    ev_chop = engine.evaluate(chopped, inp, prof, _sim(0.8))
    assert ev_ok.content_preservation > ev_stretch.content_preservation
    assert ev_ok.content_preservation > ev_chop.content_preservation


def test_artifact_detail_exposes_fifteen_detectors(engine):
    ref = _analysis()
    prof = _profile(ref)
    bad = _analysis(clip=0.02, flat=0.12, hnr=33.0, jitter=0.003)
    ev = engine.evaluate(bad, ref, prof, _sim(0.95))
    detectors = ev.detail["artifact_detectors"]
    assert len(detectors) == 15
    assert {d["name"] for d in detectors} == {
        "metallic", "robotic", "buzzing", "crackling", "resonance",
        "pitch_stability", "vibrato", "sibilance", "thd", "holes",
        "clipping", "noise", "breathing", "consonants", "transitions"}
    assert "artifact_penalties" in ev.detail
    assert not ev.gates_passed


def test_best_by_strategy_selects_each_objective():
    cands = [
        {"number": 1, "params": {"index_rate": 0.1},
         "axes": {"speaker_similarity": 0.9, "naturalness": 0.4,
                  "clarity": 0.4}, "composite": 0.6, "gates_passed": True},
        {"number": 2, "params": {"index_rate": 0.2},
         "axes": {"speaker_similarity": 0.5, "naturalness": 0.9,
                  "clarity": 0.5}, "composite": 0.62, "gates_passed": True},
        {"number": 3, "params": {"index_rate": 0.3},
         "axes": {"speaker_similarity": 0.5, "naturalness": 0.5,
                  "clarity": 0.9}, "composite": 0.61, "gates_passed": True},
        # gate-failing candidate must never win
        {"number": 4, "params": {"index_rate": 0.4},
         "axes": {"speaker_similarity": 0.99, "naturalness": 0.99,
                  "clarity": 0.99}, "composite": 0.99,
         "gates_passed": False},
    ]
    winners = EvaluationEngine.best_by_strategy(cands)
    assert set(winners) >= {"identity", "naturalness", "clarity",
                            "balanced"}
    assert winners["identity"]["number"] == 1
    assert winners["naturalness"]["number"] == 2
    assert winners["clarity"]["number"] == 3
    assert winners["balanced"]["number"] == 2     # highest composite


def test_best_by_strategy_with_validated_scores_prefers_them():
    cands = [
        {"number": 1, "params": {}, "axes": {"speaker_similarity": 0.9,
                                             "naturalness": 0.5,
                                             "clarity": 0.5},
         "composite": 0.8, "gates_passed": True},
        {"number": 2, "params": {}, "axes": {"speaker_similarity": 0.8,
                                             "naturalness": 0.8,
                                             "clarity": 0.8},
         "composite": 0.7, "gates_passed": True,
         "validated_composite": 0.9},
    ]
    winners = EvaluationEngine.best_by_strategy(cands)
    assert winners["balanced"]["number"] == 2


def test_best_by_strategy_empty_input():
    winners = EvaluationEngine.best_by_strategy([])
    assert all(winners[s] is None for s in STRATEGIES)


def test_gates_cover_content_axis():
    assert "content_preservation" in GATES


def test_pareto_front_now_uses_four_axes():
    evs = [
        VoiceEvaluation(speaker_similarity=0.9, naturalness=0.9,
                        clarity=0.9, artifact_score=0.9),
        VoiceEvaluation(speaker_similarity=0.9, naturalness=0.9,
                        clarity=0.9, artifact_score=0.5),   # dominated
        VoiceEvaluation(speaker_similarity=0.4, naturalness=0.4,
                        clarity=0.4, artifact_score=1.0),   # artifact winner
    ]
    front = EvaluationEngine.pareto_front(evs)
    assert 0 in front and 2 in front
    assert 1 not in front


def test_vibrato_irregularity_feeds_naturalness(engine):
    ref = _analysis()
    prof = _profile(ref)
    natural = _analysis(**{"f0.vibrato_irregularity": 0.85})
    oversteady = _analysis(**{"f0.vibrato_irregularity": 0.1})
    ev_n = engine.evaluate(natural, natural, prof, _sim(0.8))
    ev_s = engine.evaluate(oversteady, natural, prof, _sim(0.8))
    assert ev_n.naturalness > ev_s.naturalness
