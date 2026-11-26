"""Artifact detector bank — 15 real detectors (Phase 02 W6)."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from rvc_platform.optimizer.quality import (
    artifact_axis_score,
    detect_artifacts,
    recurrence_analysis,
)


def _analysis(**over):
    """Analysis-dict with the levers the detectors respond to."""
    bands = [0.30, 0.22, 0.14, 0.10, 0.10, 0.07, 0.05, 0.02]
    a = {
        "duration_s": 6.0,
        "f0": {"median_hz": 150.0, "range_semitones": 7.0,
               "voiced_ratio": 0.7, "octave_jump_rate": 0.01,
               "jitter_semitones": 0.12,
               "vibrato_depth_semitones": None, "vibrato_rate_hz": None,
               "vibrato_irregularity": 0.8},
        "band_energies": [b / sum(bands) for b in bands],
        "sibilance_ratio": 0.04,
        "spectral": {"centroid_hz": 900.0, "spread_hz": 1100.0,
                     "rolloff85_hz": 3200.0, "flatness": 0.02},
        "formants_hz": [520.0, 1480.0, 2600.0, 3400.0],
        "hnr_db": 14.0,
        "shimmer_db": 1.2,
        "vad": {"voiced_ratio": 0.7, "pauses": 4, "breath_count": 1,
                 "segments": [[0.2, 1.4], [1.8, 3.1], [3.6, 5.2]]},
        "loudness": {"rms_db": -18.0, "peak": 0.8,
                     "clipping_ratio": 0.0, "noise_floor_db": -55.0,
                     "snr_proxy_db": 24.0, "dynamic_range_db": 30.0,
                     "envelope_peak_rate": 2.5,
                     "discontinuity_rate": 0.01},
    }
    for k, v in over.items():
        if "." in k:
            top, sub = k.split(".")
            a[top] = {**a[top], sub: v}
        else:
            a[k] = v
    return a


@pytest.fixture()
def ref_agg():
    return {
        "spectral_flatness": {"median": 0.02},
        "f0_jitter_semitones": {"median": 0.12},
        "hnr_db": {"median": 14.0},
        "band_energies": [b / 1.0 for b in
                          [0.30, 0.22, 0.14, 0.10, 0.10, 0.07, 0.05, 0.02]],
        "formants_hz": [520.0, 1480.0, 2600.0, 3400.0],
    }


def test_all_fifteen_detectors_present():
    results = detect_artifacts(_analysis(), _analysis(), {})
    names = {r.name for r in results}
    assert len(results) == 15
    expected = {"metallic", "robotic", "buzzing", "crackling", "resonance",
                "pitch_stability", "vibrato", "sibilance", "thd", "holes",
                "clipping", "noise", "breathing", "consonants", "transitions"}
    assert names == expected


def test_clean_output_scores_high_artifact_axis(ref_agg):
    results = detect_artifacts(_analysis(), _analysis(), ref_agg)
    score, penalties = artifact_axis_score(results)
    assert score > 0.85
    # a clean output may carry only negligible (< 5%) detector penalties
    assert sum(penalties.values()) < 0.05


def test_metallic_detector_fires_on_flatness_excess(ref_agg):
    bad = _analysis(**{"spectral.flatness": 0.11})
    r = next(d for d in detect_artifacts(bad, _analysis(), ref_agg)
             if d.name == "metallic")
    assert r.present and r.severity > 0.5
    assert r.evidence["spectral_flatness"] == 0.11


def test_robotic_detector_fires_on_jitter_collapse_and_hnr_excess(ref_agg):
    bad = _analysis(**{"f0.jitter_semitones": 0.004, "hnr_db": 32.0})
    r = next(d for d in detect_artifacts(bad, _analysis(), ref_agg)
             if d.name == "robotic")
    assert r.present
    assert "jitter collapse" in r.evidence["reasons"]


def test_clipping_severity_scales_with_ratio(ref_agg):
    r1 = next(d for d in detect_artifacts(
        _analysis(**{"loudness.clipping_ratio": 0.001}), _analysis(),
        ref_agg) if d.name == "clipping")
    r2 = next(d for d in detect_artifacts(
        _analysis(**{"loudness.clipping_ratio": 0.02}), _analysis(),
        ref_agg) if d.name == "clipping")
    assert r2.severity > r1.severity
    assert r2.present and r1.present


def test_vibrato_detector_uses_depth(ref_agg):
    bad = _analysis(**{"f0.vibrato_depth_semitones": 2.0,
                       "f0.vibrato_rate_hz": 5.5})
    r = next(d for d in detect_artifacts(bad, _analysis(), ref_agg)
             if d.name == "vibrato")
    assert r.present and r.severity > 0.5


def test_sibilance_detector(ref_agg):
    bad = _analysis(sibilance_ratio=0.20)
    r = next(d for d in detect_artifacts(bad, _analysis(), ref_agg)
             if d.name == "sibilance")
    assert r.present


def test_consonant_loss_detector_vs_input():
    inp = _analysis()
    out = _analysis(**{"band_energies": [0.45, 0.30, 0.15, 0.07, 0.015,
                                        0.01, 0.005, 0.0]})
    out["band_energies"] = [x / sum(out["band_energies"])
                            for x in out["band_energies"]]
    r = next(d for d in detect_artifacts(out, inp, None)
             if d.name == "consonants")
    assert r.present and r.severity > 0.3


def test_noise_floor_rise_detector():
    inp = _analysis(**{"loudness.noise_floor_db": -60.0})
    out = _analysis(**{"loudness.noise_floor_db": -30.0})
    r = next(d for d in detect_artifacts(out, inp, None)
             if d.name == "noise")
    assert r.present
    assert r.evidence["floor_rise_db"] == 30.0


def test_crackling_detector_uses_discontinuity_rate():
    bad = _analysis(**{"loudness.discontinuity_rate": 1.5})
    r = next(d for d in detect_artifacts(bad, _analysis(), None)
             if d.name == "crackling")
    assert r.present and r.severity > 0.5


def test_detectors_report_locations_and_evidence_shape(ref_agg):
    bad = _analysis(**{"spectral.flatness": 0.15, "sibilance_ratio": 0.25})
    results = detect_artifacts(bad, _analysis(), ref_agg)
    for r in results:
        d = r.to_dict()
        assert set(d) == {"name", "present", "severity", "locations",
                          "evidence", "confidence"}
        assert 0.0 <= d["severity"] <= 1.0
        assert 0.0 <= d["confidence"] <= 1.0


def test_multiple_severe_artifacts_fail_the_gate_line(ref_agg):
    bad = _analysis(**{"spectral.flatness": 0.12, "hnr_db": 33.0,
                       "f0.jitter_semitones": 0.003,
                       "loudness.clipping_ratio": 0.02})
    score, penalties = artifact_axis_score(detect_artifacts(
        bad, _analysis(), ref_agg))
    assert score < 0.45      # below the artifact gate — must not win


def test_recurrence_analysis_links_parameters():
    trials = []
    rng = np.random.default_rng(3)
    for i in range(10):
        ir = 0.2 + 0.6 * (i % 3) / 2.0
        comp = 0.4 + 0.4 * ir + 0.02 * rng.standard_normal()
        trials.append({
            "params": {"index_rate": round(ir, 2), "protect": 0.33,
                       "rms_mix_rate": 0.25, "pitch_shift": 0,
                       "f0_method": "pm", "filter_radius": 0},
            "composite": float(comp),
            "axes": {"speaker_similarity": comp},
            "gates_passed": True, "error": None,
            "artifact_detail": {"metallic": 0.4 - 0.3 * ir},
        })
    rep = recurrence_analysis(trials)
    assert rep["n_trials"] == 10
    rec = {r["artifact"] for r in rep["recurrence"]}
    assert "metallic" in rec
    links = rep["parameter_links"].get("metallic") or []
    assert links
    # higher index_rate -> lower metallic severity: negative rho
    assert all(l["parameter"] == "index_rate" for l in links)
    assert links[0]["spearman_rho"] < 0
    assert "not a causal claim" in links[0]["note"]


def test_recurrence_never_links_with_too_few_trials():
    trials = [{
        "params": {"index_rate": 0.5}, "composite": 0.5,
        "artifact_detail": {"metallic": 0.3}, "error": None,
    }] * 4
    rep = recurrence_analysis(trials)
    assert rep["parameter_links"] == {}
