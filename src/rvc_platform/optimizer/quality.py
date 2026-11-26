"""Voice-quality artifact intelligence — 15 real detectors (Phase 02 W6).

A converted voice can be close to the target speaker's identity yet sound
wrong: metallic, robotic, buzzy, crackly... Phase 01 folded six coarse
penalties into one number. Phase 02 replaces that with a real detector
bank over the analysis features:

===========  ==================================================================
detector     evidence (all measured, all reported)
===========  ==================================================================
metallic     spectral flatness excess vs the speaker's natural anchor +
             formant cluster collapse (F1/F2 squeezed)
robotic      F0 jitter collapse + HNR excess + vibrato absence — the
             "unnaturally periodic machine voice" signature
buzzing      mid-band (250–2k Hz) energy concentration vs the reference
crackling   sample-discontinuity rate (clicks) — direct from the waveform
resonance    formant offsets vs the speaker's anchor (unnatural resonances)
pitch_stab   octave-jump rate + jitter excess vs the speaker's anchor
vibrato      dominant 4–8 Hz F0 modulation depth (unwanted vibrato)
sibilance    5–8 kHz share excess vs the INPUT utterance
thd          harmonic-distortion proxy: odd-harmonic excess via HNR /
             flatness anomaly combination
holes        high-band collapse vs input (vocoder cutoff / spectral holes)
clipping     clipped-sample ratio + peak
noise        noise-floor rise vs input + SNR-proxy drop
breathing    breath-proxy events added vs input's own pauses
consonants   2–6 kHz consonant-band retention vs input (broken consonants)
transitions  VAD structure mismatch + speech-rate change vs input
             (unnatural speech transitions)
===========  ==================================================================

Each detector returns ``present`` (bool), ``severity`` (0..1),
``locations`` (time windows where the evidence concentrated, best-effort),
``evidence`` (the raw measured numbers) and ``confidence`` (how trustworthy
this detector is with the data at hand — e.g. resonance is low-confidence
without a reference anchor).

Cross-trial intelligence:

* :func:`recurrence_analysis` — which problems repeat across trials and
  how they correlate with the parameters (rank correlation between each
  parameter value and each detector severity), so the system can say
  "buzzing drops when index_rate rises: rho = -0.62 over 14 trials".
* No single detector is ever treated as proof of badness — the overall
  ``artifact_score`` axis is a bounded combination, and every contributing
  severity stays visible in the evaluation detail.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ArtifactDetectorResult",
    "detect_artifacts",
    "artifact_axis_score",
    "recurrence_analysis",
]

# weights of the overall artifact severity (bounded combination; a single
# severe detector can push the axis down, two or more always do)
_SEVERITY_WEIGHTS = {
    "clipping": 0.22,
    "metallic": 0.18,
    "robotic": 0.16,
    "pitch_stability": 0.12,
    "consonants": 0.10,
    "crackling": 0.08,
    "buzzing": 0.06,
    "holes": 0.06,
    "noise": 0.05,
    "resonance": 0.04,
    "vibrato": 0.03,
    "sibilance": 0.03,
    "thd": 0.03,
    "breathing": 0.02,
    "transitions": 0.02,
}
_GATE_SCALE = 0.55          # severity needed alone to reach the gate line


@dataclass
class ArtifactDetectorResult:
    """One detector's verdict on one candidate output."""

    name: str
    present: bool
    severity: float                  # 0..1
    locations: list[tuple[float, float]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0          # 0..1 — data availability for this detector

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "present": bool(self.present),
            "severity": round(float(self.severity), 4),
            "locations": [[round(a, 2), round(b, 2)]
                          for a, b in self.locations[:20]],
            "evidence": self.evidence,
            "confidence": round(float(self.confidence), 3),
        }


def _num(x: Any, default: float | None = None) -> float | None:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _agg_num(node: dict[str, Any] | None, key: str,
             sub: str = "median") -> float | None:
    """Read a reference aggregate value ({median: v} shape)."""
    entry = (node or {}).get(key)
    if isinstance(entry, dict):
        return _num(entry.get(sub))
    return _num(entry)


# --------------------------------------------------------------------------- #
# detectors — each returns ArtifactDetectorResult
# --------------------------------------------------------------------------- #
def _d_metallic(out: dict[str, Any], inp: dict[str, Any] | None,
                agg: dict[str, Any] | None) -> ArtifactDetectorResult:
    flat = _num((out.get("spectral") or {}).get("flatness"))
    ref_flat = _agg_num(agg, "spectral_flatness")
    ev: dict[str, Any] = {"spectral_flatness": flat, "ref_flatness": ref_flat}
    if flat is None or ref_flat is None:
        # no anchor to compare against — "not detected" with LOW confidence
        return ArtifactDetectorResult(
            "metallic", False, 0.0, evidence=ev, confidence=0.3)
    excess = max(0.0, flat - (ref_flat + 0.012))
    sev = float(np.clip(excess / 0.05, 0.0, 1.0))
    # formant cluster collapse adds evidence (F2/F1 squeezed)
    f = [v for v in (out.get("formants_hz") or []) if v is not None]
    if len(f) >= 2 and f[1] > 0:
        ev["f2_f1_ratio"] = round(f[1] / f[0], 2)
        if f[1] / f[0] < 2.0:                      # squeezed vowel space
            sev = float(min(1.0, sev + 0.15))
            ev["f2_f1_ratio_flag"] = True
    # locations: high-flatness speech segments (best-effort)
    locs: list[tuple[float, float]] = []
    if sev > 0.2:
        locs = [(a, b) for a, b in (out.get("vad") or {}).get(
            "segments", [])[:8]]
    return ArtifactDetectorResult(
        "metallic", sev >= 0.2, sev, locs, ev)


def _d_robotic(out: dict[str, Any], inp: dict[str, Any] | None,
               agg: dict[str, Any] | None) -> ArtifactDetectorResult:
    jit = _num((out.get("f0") or {}).get("jitter_semitones"))
    ref_jit = _agg_num(agg, "f0_jitter_semitones")
    hnr = _num(out.get("hnr_db"))
    ref_hnr = _agg_num(agg, "hnr_db")
    vib = _num((out.get("f0") or {}).get("vibrato_depth_semitones"))
    ev = {"jitter_semitones": jit, "ref_jitter": ref_jit,
          "hnr_db": hnr, "ref_hnr": ref_hnr}
    score = 0.0
    reasons: list[str] = []
    if ref_jit is not None and jit is not None and ref_jit > 0.01:
        if jit < 0.35 * ref_jit:
            score += 0.45
            reasons.append("jitter collapse")
        ev["jitter_ratio"] = round(jit / ref_jit, 3)
    if ref_hnr is not None and hnr is not None and hnr > ref_hnr + 7.0:
        score += 0.4
        reasons.append("HNR excess")
        ev["hnr_excess_db"] = round(hnr - ref_hnr, 2)
    if vib is not None and jit is not None and jit < 0.02:
        # unnaturally steady + vibrato-free
        score += 0.15
        reasons.append("vibrato-free steady pitch")
    sev = float(np.clip(score, 0.0, 1.0))
    ev["reasons"] = reasons
    conf = 1.0 if (ref_jit is not None and ref_hnr is not None) else 0.4
    return ArtifactDetectorResult("robotic", sev >= 0.2, sev, [], ev, conf)


def _d_buzzing(out: dict[str, Any], inp: dict[str, Any] | None,
               agg: dict[str, Any] | None) -> ArtifactDetectorResult:
    ob = out.get("band_energies") or []
    rb = (agg or {}).get("band_energies") or []
    if len(ob) != 8 or len(rb) != 8:
        return ArtifactDetectorResult("buzzing", False, 0.0, evidence={},
                                       confidence=0.3)
    out_mid = float(np.mean(ob[1:4]))
    ref_mid = float(np.mean(rb[1:4]))
    if ref_mid <= 1e-9:
        return ArtifactDetectorResult("buzzing", False, 0.0, evidence={},
                                       confidence=0.3)
    d = abs(np.log2(max(out_mid, 1e-6) / ref_mid))
    sev = float(np.clip(d / 1.2, 0.0, 1.0))
    ev = {"mid_band_output": round(out_mid, 4),
          "mid_band_reference": round(ref_mid, 4),
          "log2_ratio": round(float(d), 3)}
    locs = [(a, b) for a, b in (out.get("vad") or {}).get(
        "segments", [])[:6]] if sev >= 0.3 else []
    return ArtifactDetectorResult("buzzing", sev >= 0.25, sev, locs, ev)


def _d_crackling(out: dict[str, Any], inp: dict[str, Any] | None,
                 ) -> ArtifactDetectorResult:
    disc = _num((out.get("loudness") or {}).get("discontinuity_rate"))
    ev = {"discontinuities_per_second": disc}
    if disc is None:
        return ArtifactDetectorResult("crackling", False, 0.0, evidence=ev,
                                       confidence=0.2)
    sev = float(np.clip((disc - 0.05) / 2.0, 0.0, 1.0))
    return ArtifactDetectorResult("crackling", disc > 0.15, sev, [], ev)


def _d_resonance(out: dict[str, Any], inp: dict[str, Any] | None,
                 agg: dict[str, Any] | None) -> ArtifactDetectorResult:
    of = out.get("formants_hz") or []
    rf = (agg or {}).get("formants_hz") or []
    pairs = [(a, b) for a, b in zip(of[:4], rf[:4])
             if a is not None and b is not None]
    if len(pairs) < 2:
        return ArtifactDetectorResult("resonance", False, 0.0, evidence={},
                                       confidence=0.3)
    errs = [abs(float(a) - float(b)) / float(b) for a, b in pairs]
    mean_err = float(np.mean(errs))
    sev = float(np.clip((mean_err - 0.10) / 0.25, 0.0, 1.0))
    ev = {"formant_relative_errors": [round(e, 3) for e in errs],
          "mean_relative_error": round(mean_err, 3),
          "pairs": [[round(a, 1), round(b, 1)] for a, b in pairs]}
    return ArtifactDetectorResult(
        "resonance", sev >= 0.25, sev, [], ev,
        confidence=0.7)       # Praat formants are estimates, not ground truth


def _d_pitch_stability(out: dict[str, Any], inp: dict[str, Any] | None,
                       agg: dict[str, Any] | None) -> ArtifactDetectorResult:
    jumps = _num((out.get("f0") or {}).get("octave_jump_rate"))
    jit = _num((out.get("f0") or {}).get("jitter_semitones"))
    ref_jit = _agg_num(agg, "f0_jitter_semitones")
    ev = {"octave_jump_rate": jumps, "jitter_semitones": jit,
          "ref_jitter": ref_jit}
    score = 0.0
    if jumps is not None:
        score += float(np.clip(jumps / 0.08, 0.0, 0.6))
    if ref_jit is not None and jit is not None:
        if jit > 2.0 * ref_jit and jit > 0.15:
            score += float(np.clip((jit / max(ref_jit, 1e-9) - 2.0) / 6.0,
                                  0.0, 0.4))
            ev["jitter_ratio"] = round(jit / max(ref_jit, 1e-9), 2)
    sev = float(np.clip(score, 0.0, 1.0))
    conf = 1.0 if ref_jit is not None else 0.6
    return ArtifactDetectorResult("pitch_stability", sev >= 0.25, sev,
                                  [], ev, conf)


def _d_vibrato(out: dict[str, Any], inp: dict[str, Any] | None,
               ) -> ArtifactDetectorResult:
    depth = _num((out.get("f0") or {}).get("vibrato_depth_semitones"))
    rate = _num((out.get("f0") or {}).get("vibrato_rate_hz"))
    ev = {"vibrato_depth_semitones": depth, "vibrato_rate_hz": rate}
    if depth is None or rate is None:
        return ArtifactDetectorResult("vibrato", False, 0.0, evidence=ev,
                                       confidence=0.6)
    # natural vibrato depth is usually < 0.6 st; > 1 st is clearly audible
    sev = float(np.clip((depth - 0.5) / 1.5, 0.0, 1.0))
    ev["audible"] = sev >= 0.25
    return ArtifactDetectorResult("vibrato", sev >= 0.25, sev, [], ev)


def _d_sibilance(out: dict[str, Any], inp: dict[str, Any] | None,
                 agg: dict[str, Any] | None) -> ArtifactDetectorResult:
    sib = _num(out.get("sibilance_ratio"))
    ref_flat = _agg_num(agg, "spectral_flatness")
    if sib is None or ref_flat is None:
        return ArtifactDetectorResult("sibilance", False, 0.0,
                                      evidence={"sibilance_ratio": sib},
                                      confidence=0.3)
    ev = {"sibilance_ratio": sib}
    sev = float(np.clip((sib - 0.06) / 0.18, 0.0, 1.0))
    ev["anchor_note"] = "5-8kHz share; >12% is hissy"
    locs: list[tuple[float, float]] = []
    if sev >= 0.3:
        locs = [(a, b) for a, b in (out.get("vad") or {}).get(
            "segments", [])[:8]]
    return ArtifactDetectorResult("sibilance", sev >= 0.3, sev, locs, ev)


def _d_thd(out: dict[str, Any], inp: dict[str, Any] | None,
           agg: dict[str, Any] | None) -> ArtifactDetectorResult:
    """Harmonic-distortion proxy: unnaturally strong periodicity with a
    harmonically-locked spectrum (flatness LOW + HNR high + jitter low —
    the saturated-generator signature)."""
    flat = _num((out.get("spectral") or {}).get("flatness"))
    hnr = _num(out.get("hnr_db"))
    ref_hnr = _agg_num(agg, "hnr_db")
    ref_flat = _agg_num(agg, "spectral_flatness")
    ev: dict[str, Any] = {"flatness": flat, "hnr_db": hnr}
    if None in (flat, hnr, ref_hnr, ref_flat):
        return ArtifactDetectorResult("thd", False, 0.0, evidence=ev,
                                       confidence=0.4)
    score = 0.0
    if hnr - ref_hnr > 8.0 and flat < ref_flat * 0.75:
        score += 0.6                          # over-locked spectrum
        ev["hnr_excess_db"] = round(hnr - ref_hnr, 2)
        ev["flatness_ratio"] = round(flat / max(ref_flat, 1e-9), 3)
    if (out.get("f0") or {}).get("jitter_semitones") is not None:
        jit = float((out["f0"])["jitter_semitones"])
        if jit < 0.01:
            score += 0.3                      # machine-steady F0
            ev["jitter_semitones"] = round(jit, 4)
    sev = float(np.clip(score, 0.0, 1.0))
    return ArtifactDetectorResult("thd", sev >= 0.3, sev, [], ev)


def _d_holes(out: dict[str, Any], inp: dict[str, Any] | None,
             ) -> ArtifactDetectorResult:
    ib = (inp or {}).get("band_energies") or []
    ob = out.get("band_energies") or []
    if len(ib) != 8 or len(ob) != 8 or float(np.sum(ib[6:8])) <= 1e-4:
        return ArtifactDetectorResult("holes", False, 0.0, evidence={},
                                       confidence=0.3)
    kept = float(np.sum(ob[6:8])) / float(np.sum(ib[6:8]))
    sev = float(np.clip((0.6 - kept) / 0.6, 0.0, 1.0)) if kept < 0.6 else 0.0
    ev = {"hf_retention": round(kept, 3)}
    return ArtifactDetectorResult(
        "holes", kept < 0.5, sev, [], ev)


def _d_clipping(out: dict[str, Any], inp: dict[str, Any] | None,
                ) -> ArtifactDetectorResult:
    lum = out.get("loudness") or {}
    clip = _num(lum.get("clipping_ratio"))
    peak = _num(lum.get("peak"))
    ev = {"clipping_ratio": clip, "peak": peak}
    if clip is None:
        return ArtifactDetectorResult("clipping", False, 0.0, evidence=ev,
                                       confidence=0.2)
    sev = float(np.clip(clip / 0.004, 0.0, 1.0))
    return ArtifactDetectorResult("clipping", clip >= 0.001, sev, [], ev)


def _d_noise(out: dict[str, Any], inp: dict[str, Any] | None,
             agg: dict[str, Any] | None) -> ArtifactDetectorResult:
    in_floor = _num(((inp or {}).get("loudness") or {}).get("noise_floor_db"))
    out_floor = _num((out.get("loudness") or {}).get("noise_floor_db"))
    snr = _num((out.get("loudness") or {}).get("snr_proxy_db"))
    ev = {"noise_floor_db": out_floor, "input_floor_db": in_floor,
          "snr_proxy_db": snr}
    score = 0.0
    if in_floor is not None and out_floor is not None:
        # Phase 02 bug fix (caught by the new tests): a noise-floor RISE means
        # the output floor went UP (toward 0 dB / louder) vs the input —
        # ``out_floor - in_floor``. The Phase 01 formula (in - out) fired on
        # the opposite condition and never saw real floor rises.
        rise = max(0.0, out_floor - in_floor)
        ev["floor_rise_db"] = round(rise, 2)
        score += float(np.clip(rise / 18.0, 0.0, 0.6))
    if snr is not None:
        score += float(np.clip((10.0 - snr) / 10.0, 0.0, 0.4))
    sev = float(np.clip(score, 0.0, 1.0))
    conf = 1.0 if in_floor is not None else 0.5
    return ArtifactDetectorResult("noise", sev >= 0.3, sev, [], ev, conf)


def _d_breathing(out: dict[str, Any], inp: dict[str, Any] | None,
                 ) -> ArtifactDetectorResult:
    breaths = _num((out.get("vad") or {}).get("breath_count"))
    in_breaths = _num(((inp or {}).get("vad") or {}).get("breath_count"))
    in_pauses = _num(((inp or {}).get("vad") or {}).get("pauses"))
    ev = {"breath_events": breaths, "input_breath_events": in_breaths}
    if breaths is None:
        return ArtifactDetectorResult("breathing", False, 0.0, evidence={},
                                       confidence=0.2)
    if in_breaths is not None:
        added = max(0, int(breaths - max(in_breaths, 0)))
        ev["added_breaths"] = added
        sev = float(np.clip(added / 4.0, 0.0, 1.0))
    else:
        # no input to compare: many breaths in a short clip is still odd
        sev = float(np.clip((breaths - 6) / 8.0, 0.0, 1.0)) \
            if (in_pauses or 0) < 4 else 0.0
    return ArtifactDetectorResult("breathing", sev >= 0.25, sev, [], ev,
                                   confidence=0.6)


def _d_consonants(out: dict[str, Any], inp: dict[str, Any] | None,
                  ) -> ArtifactDetectorResult:
    ib = (inp or {}).get("band_energies") or []
    ob = out.get("band_energies") or []
    if len(ib) != 8 or len(ob) != 8:
        return ArtifactDetectorResult("consonants", False, 0.0, evidence={},
                                       confidence=0.3)
    in_hf = float(np.sum(ib[4:7]))            # 2..6 kHz
    out_hf = float(np.sum(ob[4:7]))
    if in_hf <= 1e-6:
        return ArtifactDetectorResult("consonants", False, 0.0, evidence={},
                                       confidence=0.3)
    ratio = out_hf / in_hf
    ev = {"consonant_band_retention": round(ratio, 3)}
    loss = max(0.0, 1.0 - ratio)              # ratio < 1 loses consonants
    excess = max(0.0, ratio - 2.0) / 2.0       # blown-up = another artifact
    sev = float(np.clip(max(loss, min(excess, 1.0)), 0.0, 1.0))
    locs = [(a, b) for a, b in (out.get("vad") or {}).get(
        "segments", [])[:8]] if sev >= 0.3 else []
    return ArtifactDetectorResult("consonants", sev >= 0.25, sev, locs, ev)


def _d_transitions(out: dict[str, Any], inp: dict[str, Any] | None,
                   ) -> ArtifactDetectorResult:
    v_in = (inp or {}).get("vad") or {}
    v_out = out.get("vad") or {}
    ev: dict[str, Any] = {}
    if not v_in or not v_out:
        return ArtifactDetectorResult("transitions", False, 0.0, evidence={},
                                       confidence=0.4)
    score = 0.0
    if v_in.get("voiced_ratio") is not None and \
            v_out.get("voiced_ratio") is not None:
        d = abs(float(v_in["voiced_ratio"]) - float(v_out["voiced_ratio"]))
        ev["voiced_ratio_input"] = round(float(v_in["voiced_ratio"]), 3)
        ev["voiced_ratio_output"] = round(float(v_out["voiced_ratio"]), 3)
        if d > 0.15:
            score += float(np.clip((d - 0.15) / 0.35, 0.0, 0.5))
            ev["voiced_ratio_gap"] = round(d, 3)
    p_in = v_in.get("pauses")
    p_out = v_out.get("pauses")
    if p_in is not None and p_out is not None:
        denom = max(3, int(p_in) + 1, int(p_out) + 1)
        d = abs(int(p_in) - int(p_out))
        if d / denom > 0.4:
            score += 0.3
            ev["pause_count_gap"] = d
    # speech-rate change (envelope peak rate)
    r_in = _num(((inp or {}).get("loudness") or {}).get("envelope_peak_rate"))
    r_out = _num((out.get("loudness") or {}).get("envelope_peak_rate"))
    if r_in is not None and r_out is not None and r_in > 0.05:
        rel = abs(r_out - r_in) / r_in
        ev["peak_rate_input"] = round(r_in, 2)
        ev["peak_rate_output"] = round(r_out, 2)
        if rel > 0.35:
            score += float(np.clip((rel - 0.35) / 0.8, 0.0, 0.3))
            ev["peak_rate_change"] = round(rel, 3)
    sev = float(np.clip(score, 0.0, 1.0))
    return ArtifactDetectorResult("transitions", sev >= 0.3, sev, [], ev)


# --------------------------------------------------------------------------- #
# detector bank
# --------------------------------------------------------------------------- #
def detect_artifacts(out: dict[str, Any], inp: dict[str, Any] | None = None,
                    agg: dict[str, Any] | None = None,
                    ) -> list[ArtifactDetectorResult]:
    """Run all 15 detectors on one candidate output analysis."""
    results = [
        _d_metallic(out, inp, agg),
        _d_robotic(out, inp, agg),
        _d_buzzing(out, inp, agg),
        _d_crackling(out, inp),
        _d_resonance(out, inp, agg),
        _d_pitch_stability(out, inp, agg),
        _d_vibrato(out, inp),
        _d_sibilance(out, inp, agg),
        _d_thd(out, inp, agg),
        _d_holes(out, inp),
        _d_clipping(out, inp),
        _d_noise(out, inp, agg),
        _d_breathing(out, inp),
        _d_consonants(out, inp),
        _d_transitions(out, inp),
    ]
    return results


def artifact_axis_score(results: list[ArtifactDetectorResult],
                        ) -> tuple[float, dict[str, float]]:
    """Overall artifact axis (1 = clean) + per-detector penalty view.

    Bounded weighted combination: ``severity_total = Σ w_i × s_i`` with
    weights normalised to reach 1.0 only when several detectors fire — one
    severe detector alone reaches ~0.45 (just below the old 0.55 gate
    line), two or more always cross it.
    """
    penalties: dict[str, float] = {}
    total = 0.0
    for r in results:
        w = _SEVERITY_WEIGHTS.get(r.name, 0.0)
        if r.severity > 0.02:
            # scale single-detector max to _GATE_SCALE, multi-detector sums
            contrib = r.severity * (w / _GATE_SCALE)
            penalties[r.name] = round(contrib, 4)
            total += contrib
    severity = float(np.clip(total, 0.0, 1.0))
    return float(1.0 - severity), penalties


# --------------------------------------------------------------------------- #
# cross-trial recurrence + parameter linkage
# --------------------------------------------------------------------------- #
def _rankdata(x: list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, arr.size + 1)
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 4:
        return 0.0
    rx, ry = _rankdata(x), _rankdata(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def recurrence_analysis(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Which artifacts repeat across trials and how parameters relate.

    ``trials`` — session trial dicts with ``params`` + ``axes`` +
    ``artifact_detail`` (the detector penalties stored by the service).

    Output:

    * ``recurrence`` — per artifact: trial count, mean/max severity,
      stability (share of trials with the artifact present);
    * ``parameter_links`` — per artifact: Spearman rank correlation between
      each numeric parameter value and the artifact severity (|rho| >= 0.4
      with n >= 6 is reported as a link; never a causal claim).
    """
    rec: dict[str, dict[str, Any]] = {}
    ok_trials = [t for t in trials if not t.get("error")]
    for t in ok_trials:
        detail = t.get("artifact_detail") or {}
        for name, sev in detail.items():
            slot = rec.setdefault(name, {
                "n": 0, "sum": 0.0, "max": 0.0,
                "params": [], "sevs": []})
            sev_f = float(sev)
            slot["n"] += 1
            slot["sum"] += sev_f
            slot["max"] = max(slot["max"], sev_f)
            slot["params"].append(dict(t.get("params") or {}))
            slot["sevs"].append(sev_f)
    n_trials = len(ok_trials)
    recurrence: list[dict[str, Any]] = []
    links: dict[str, list[dict[str, Any]]] = {}
    for name, slot in sorted(rec.items()):
        n = slot["n"]
        mean = slot["sum"] / n if n else 0.0
        recurrence.append({
            "artifact": name,
            "trials_affected": n,
            "share_of_trials": round(n / max(1, n_trials), 3),
            "mean_severity": round(mean, 4),
            "max_severity": round(slot["max"], 4),
        })
        # parameter linkage (rank correlation) — only with enough evidence
        if n >= 6:
            sevs = slot["sevs"]
            param_sets = slot["params"]
            numeric_keys: list[str] = []
            for k, v in (param_sets[0] or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numeric_keys.append(k)
            for key in numeric_keys:
                xs = [float((p or {}).get(key, 0.0)) for p in param_sets]
                rho = _spearman(xs, sevs)
                if abs(rho) >= 0.4:
                    links.setdefault(name, []).append({
                        "parameter": key,
                        "spearman_rho": round(rho, 3),
                        "n": n,
                        "direction": ("severity_rises_with" if rho > 0
                                      else "severity_falls_with"),
                        "note": "statistical association over this "
                                "session's trials — not a causal claim",
                    })
    return {
        "n_trials": n_trials,
        "recurrence": recurrence,
        "parameter_links": links,
    }
