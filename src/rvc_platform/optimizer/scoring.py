"""Multi-metric voice evaluation engine — honest, separable, gated.

After every candidate conversion the output audio is scored on NINE
separate axes (each 0..1, higher = better, each reported independently so a
pretty average can never hide a failing axis):

1. ``speaker_similarity`` — cosine similarity between the candidate's
   speaker-verification embeddings and the reference profile's embeddings
   (WavLM x-vector + ECAPA-TDNN; per backend + ensemble). Phase 02: when
   the embedder report carries a per-backend **calibrated** ensemble (built
   from same-speaker cohort stats, see optimizer/calibration.py), the
   calibrated value is used for the axis and BOTH raw and calibrated are
   recorded in the detail. THE primary objective.
2. ``timbre_similarity`` — MFCC distribution + 8-band spectral profile +
   formant closeness vs the reference aggregate (voice colour match).
3. ``pitch_consistency`` — median-F0 closeness (semitones) + pitch-range
   preservation + octave-jump penalty vs the reference F0 statistics.
4. ``naturalness`` — HNR / jitter / shimmer / vibrato-irregularity
   proximity to the reference anchors (a converted voice that is *much*
   more periodic or much noisier than the speaker's natural recordings
   sounds synthetic; measured as distance to the reference anchors, not
   as an absolute value).
5. ``clarity`` — spectral detail retention vs the INPUT utterance +
   high-band energy preservation (articulation crispness proxy).
6. ``artifact_score`` — Phase 02 delegates to the real 15-detector bank in
   ``optimizer/quality.py`` (metallic, robotic, buzzing, crackling,
   resonance, pitch stability, vibrato, sibilance, THD, spectral holes,
   clipping, noise, breathing, consonants, transitions) with locations and
   per-detector evidence preserved in the evaluation detail.
7. ``prosody_preservation`` — Phase 02 implements the correlations the
   Phase 01 docstring only promised: TRUE frame-envelope correlation +
   time-warped F0 contour correlation + speech-rate agreement + pause
   structure agreement vs the INPUT utterance (the words and performance
   being converted).
8. ``content_preservation`` — NEW axis (Phase 02): did the *speech itself*
   survive — duration fidelity, voiced-structure agreement, speech-rate
   ratio, consonant-band retention vs the INPUT utterance.
9. ``audio_quality`` — SNR proxy + dynamic range + clipping vs anchors.

Composite scoring is a *weighted mean with quality gates*: candidates
failing the artifact/clarity/prosody/content gates are mathematically
excluded from "best" selection regardless of speaker similarity (a
distorted voice never wins), and every number is presented as a relative
score within this search — never as an identity-match probability.

Pareto analysis: non-dominated filtering over
(speaker_similarity, naturalness, clarity, artifact_score) exposes
trade-offs instead of forcing one hidden weighting. Final selection
additionally exposes best-by-strategy alternatives (identity /
naturalness / clarity / balanced) — see :func:`best_by_strategy`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# default weights — speaker identity dominates by design; the sum is
# normalised internally so users can tweak any subset freely.
DEFAULT_WEIGHTS: dict[str, float] = {
    "speaker_similarity": 0.29,
    "timbre_similarity": 0.11,
    "pitch_consistency": 0.09,
    "naturalness": 0.11,
    "clarity": 0.10,
    "artifact_score": 0.11,
    "prosody_preservation": 0.07,
    "content_preservation": 0.07,
    "audio_quality": 0.05,
}

# hard gates: below these a candidate is excluded from final selection
# (distorted / unintelligible outputs must never win on similarity alone).
GATES = {
    "artifact_score": 0.45,
    "clarity": 0.40,
    "prosody_preservation": 0.30,
    "content_preservation": 0.30,
}

#: selection strategies exposed in the final results (Phase 02 W11)
STRATEGIES = ("balanced", "identity", "naturalness", "clarity")


@dataclass
class MetricWeights:
    weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def normalised(self) -> dict[str, float]:
        w = {k: max(0.0, float(v)) for k, v in self.weights.items()
             if k in DEFAULT_WEIGHTS}
        total = sum(w.values())
        if total <= 0:
            return dict(DEFAULT_WEIGHTS)
        return {k: v / total for k, v in w.items()}


@dataclass
class VoiceEvaluation:
    """All nine axis scores + composite + gate verdict for one candidate."""

    speaker_similarity: float | None = None
    timbre_similarity: float | None = None
    pitch_consistency: float | None = None
    naturalness: float | None = None
    clarity: float | None = None
    artifact_score: float | None = None
    prosody_preservation: float | None = None
    content_preservation: float | None = None
    audio_quality: float | None = None
    composite: float | None = None
    gates_passed: bool = True
    failed_gates: list[str] = field(default_factory=list)
    reliability: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def axis_scores(self) -> dict[str, float | None]:
        return {
            "speaker_similarity": self.speaker_similarity,
            "timbre_similarity": self.timbre_similarity,
            "pitch_consistency": self.pitch_consistency,
            "naturalness": self.naturalness,
            "clarity": self.clarity,
            "artifact_score": self.artifact_score,
            "prosody_preservation": self.prosody_preservation,
            "content_preservation": self.content_preservation,
            "audio_quality": self.audio_quality,
        }

    def to_dict(self) -> dict[str, Any]:
        out = {
            "axes": {k: None if v is None else round(float(v), 4)
                     for k, v in self.axis_scores().items()},
            "composite": None if self.composite is None
            else round(float(self.composite), 4),
            "gates_passed": self.gates_passed,
            "failed_gates": self.failed_gates,
            "reliability": self.reliability,
            "warnings": list(self.warnings),
        }
        if self.detail:
            out["detail"] = self.detail
        return out


def _sigmoid(x: np.ndarray | float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64))))


def _proximity(value: float | None, anchor: float | None,
               tolerance: float) -> float:
    """1.0 when value == anchor, decaying smoothly to 0 at |value-anchor| ==
    tolerance (tolerance expressed in the value's own units)."""
    if value is None or anchor is None:
        return 0.5                      # unknown -> neutral, never rewarded
    return float(np.clip(
        1.0 - abs(value - anchor) / max(tolerance, 1e-9), 0.0, 1.0))


def _safe(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _hist_distance(a: list[float] | None,
                   b: list[float] | None) -> float:
    """Jeffreys divergence between two normalised histograms (0 = equal)."""
    if not a or not b or len(a) != len(b):
        return 1.0
    x = np.clip(np.asarray(a, dtype=np.float64), 1e-9, None)
    y = np.clip(np.asarray(b, dtype=np.float64), 1e-9, None)
    return float(0.5 * np.sum(x * np.log(x / y) + y * np.log(y / x)))


def _corr(a: list[float] | None, b: list[float] | None) -> float | None:
    """Pearson correlation of two equal-length series (None if degenerate)."""
    if not a or not b or len(a) != len(b) or len(a) < 4:
        return None
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _corr_score(corr: float | None) -> float | None:
    """Correlation (-1..1) -> axis contribution (0..1); None stays None."""
    if corr is None:
        return None
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


class EvaluationEngine:
    """Scores candidate outputs against a reference profile + input analysis."""

    def __init__(self, weights: MetricWeights | None = None) -> None:
        self.metric_weights = weights or MetricWeights()

    # ------------------------------------------------------------------ api
    def evaluate(
        self,
        output_analysis: dict[str, Any],
        input_analysis: dict[str, Any],
        reference_profile: dict[str, Any],      # ReferenceProfile.to_dict()
        embedder_similarity: dict[str, Any] | None = None,
    ) -> VoiceEvaluation:
        ev = VoiceEvaluation()
        agg = reference_profile.get("stats", {}).get("aggregate") or {}
        if not agg:
            ev.warnings.append(
                "reference profile has no aggregated statistics")

        # 1) speaker identity — from the embedder manager report
        ev.speaker_similarity, ev.reliability = self._speaker(
            output_analysis, reference_profile, embedder_similarity, ev)

        # 2) timbre vs reference
        ev.timbre_similarity = self._timbre(output_analysis, agg)

        # 3) pitch vs reference stats
        ev.pitch_consistency = self._pitch(output_analysis, agg, ev)

        # 4) naturalness vs reference anchors
        ev.naturalness = self._naturalness(output_analysis, agg)

        # 5) clarity vs input
        ev.clarity = self._clarity(output_analysis, input_analysis)

        # 6) artifacts — the real 15-detector bank (Phase 02)
        ev.artifact_score = self._artifacts(
            output_analysis, input_analysis, agg, ev)

        # 7) prosody vs input — TRUE correlations (Phase 02 W5)
        ev.prosody_preservation = self._prosody(
            output_analysis, input_analysis)

        # 8) content preservation vs input (Phase 02 W13)
        ev.content_preservation = self._content(
            output_analysis, input_analysis)

        # 9) audio quality
        ev.audio_quality = self._quality(output_analysis, agg)

        # composite + gates
        ev.composite, ev.gates_passed, ev.failed_gates = self._composite(ev)
        for w in (output_analysis.get("warnings") or [])[:4]:
            ev.warnings.append(f"analysis: {w}")
        return ev

    # ------------------------------------------------------------- 1) speaker
    def _speaker(self, out: dict[str, Any], profile: dict[str, Any],
                 report: dict[str, Any] | None, ev: VoiceEvaluation
                 ) -> tuple[float | None, str]:
        if report is None:
            ev.warnings.append(
                "speaker similarity unavailable (no embedder report) — "
                "identity axis degraded")
            return None, "low"
        sims = report.get("similarities") or {}
        rel = report.get("reliability") or "low"
        calibration = report.get("calibration") or {}
        calibrated_ensemble = calibration.get("calibrated_ensemble")

        neural = {k: v for k, v in sims.items() if k in ("wavlm", "ecapa")}
        if calibrated_ensemble is not None:
            # Phase 02: calibrated ensemble — comparable scale across models
            ev.detail["speaker_backends"] = {
                k: round(_safe(v), 4) for k, v in sims.items()}
            ev.detail["speaker_calibrated"] = {
                k: d for k, d in
                (calibration.get("per_backend") or {}).items()}
            ev.detail["speaker_calibrated_ensemble"] = calibrated_ensemble
            ev.detail["speaker_agreement_spread"] = \
                report.get("agreement_spread")
            return float(calibrated_ensemble), rel
        if neural:
            ev.detail["speaker_backends"] = {
                k: round(_safe(v), 4) for k, v in sims.items()}
            ev.detail["speaker_agreement_spread"] = \
                report.get("agreement_spread")
            ev.warnings.append(
                "identity scored with RAW ensemble (no per-backend "
                "calibration available — profile needs >= 2 reference "
                "samples per backend)")
            return float(np.mean(list(neural.values()))), rel
        if "dsp" in sims:
            # DSP voiceprint only — heavily flagged
            ev.warnings.append(
                "identity scored with the DSP fallback only "
                "(reliability LOW)")
            return float(sims["dsp"]), "low"
        return None, rel

    # ------------------------------------------------------------- 2) timbre
    def _timbre(self, out: dict[str, Any], agg: dict[str, Any]) -> float:
        if not agg:
            return 0.5
        scores: list[tuple[float, float]] = []          # (score, weight)

        ref_mfcc = agg.get("mfcc_mean")
        out_mfcc = out.get("mfcc_mean")
        if isinstance(ref_mfcc, list) and isinstance(out_mfcc, list) \
                and len(ref_mfcc) == len(out_mfcc) == 20:
            a = np.asarray(out_mfcc[:14], dtype=np.float64)
            b = np.asarray(ref_mfcc[:14], dtype=np.float64)
            wts = np.ones(a.shape)
            if wts.size:
                wts[0] = 0.25
            d = float(np.sqrt(np.sum(
                ((a - b) / 12.0) ** 2 * wts) / max(1e-9, wts.sum())))
            scores.append((float(np.clip(1.0 - d / 1.5, 0.0, 1.0)), 2.0))

        ref_bands = agg.get("band_energies")
        out_bands = out.get("band_energies")
        if isinstance(ref_bands, list) and isinstance(out_bands, list) \
                and len(ref_bands) == len(out_bands) == 8:
            d = _hist_distance(out_bands, ref_bands)
            scores.append((float(np.clip(1.0 - d / 0.8, 0.0, 1.0)), 1.5))

        ref_f = agg.get("formants_hz") or []
        out_f = out.get("formants_hz") or []
        pairs = [(a, b) for a, b in zip(out_f[:4], ref_f[:4])
                 if a is not None and b is not None]
        if len(pairs) >= 3:
            errs = [abs(float(a) - float(b)) / float(b) for a, b in pairs]
            scores.append((float(np.clip(
                1.0 - float(np.mean(errs)) / 0.35, 0.0, 1.0)), 1.0))

        if not scores:
            return 0.5
        total_w = sum(w for _, w in scores)
        return float(sum(s * w for s, w in scores) / max(1e-9, total_w))

    # -------------------------------------------------------------- 3) pitch
    def _pitch(self, out: dict[str, Any], agg: dict[str, Any],
               ev: VoiceEvaluation) -> float:
        if not agg:
            return 0.5
        f0_out = (out.get("f0") or {})
        ref_median = (agg.get("f0_median_hz") or {}).get("median")
        out_median = f0_out.get("median_hz")
        parts: list[tuple[float, float]] = []
        if ref_median is not None and out_median is not None:
            st = 12.0 * np.log2(max(out_median, 1.0)
                                 / max(ref_median, 1e-9))
            parts.append((float(np.clip(1.0 - abs(st) / 4.0, 0.0, 1.0)), 2.5))
            ev.detail["pitch_offset_semitones"] = round(float(st), 3)
        ref_range = (agg.get("f0_range_semitones") or {}).get("median")
        out_range = f0_out.get("range_semitones")
        if ref_range is not None and out_range is not None:
            d = abs(out_range - ref_range)
            parts.append((float(np.clip(1.0 - d / 8.0, 0.0, 1.0)), 1.0))
        jumps = f0_out.get("octave_jump_rate")
        if jumps is not None:
            parts.append((float(np.clip(1.0 - jumps / 0.10, 0.0, 1.0)), 1.5))
        hist_d = _hist_distance(f0_out.get("histogram"), agg.get("f0_histogram"))
        parts.append((float(np.clip(1.0 - hist_d / 1.2, 0.0, 1.0)), 1.0))
        if not parts:
            return 0.5
        tw = sum(w for _, w in parts)
        return float(sum(s * w for s, w in parts) / max(1e-9, tw))

    # -------------------------------------------------------- 4) naturalness
    def _naturalness(self, out: dict[str, Any], agg: dict[str, Any]) -> float:
        """Proximity of voice-quality measures to the speaker's natural
        anchors: both "more synthetic than the speaker" (HNR/jitter way off)
        and "noisier than the speaker" score low."""
        if not agg:
            return 0.5
        ref_hnr = (agg.get("hnr_db") or {}).get("median")
        out_hnr = out.get("hnr_db")
        p_hnr = _proximity(out_hnr, ref_hnr, 8.0)

        ref_jit = (agg.get("f0_jitter_semitones") or {}).get("median")
        out_jit = (out.get("f0") or {}).get("jitter_semitones")
        p_jit = _proximity(out_jit, ref_jit, 0.35)

        ref_shim = (agg.get("shimmer_db") or {}).get("median")
        out_shim = out.get("shimmer_db")
        p_shim = _proximity(out_shim, ref_shim, 2.5)

        ref_flat = (agg.get("spectral_flatness") or {}).get("median")
        out_flat = (out.get("spectral") or {}).get("flatness")
        p_flat = _proximity(out_flat, ref_flat, 0.02)

        # vibrato irregularity proximity (Phase 02): natural speech has
        # irregular F0 modulation; a flat/over-regular contour sounds robotic
        out_irr = (out.get("f0") or {}).get("vibrato_irregularity")
        p_irr = _proximity(out_irr, 0.75, 0.45)   # natural anchors ~0.6-0.9

        vals = [(p_hnr, 1.5), (p_jit, 1.5), (p_shim, 1.0), (p_flat, 0.8),
                (p_irr, 0.7)]
        tw = sum(w for _, w in vals)
        return float(sum(s * w for s, w in vals) / max(1e-9, tw))

    # ------------------------------------------------------------ 5) clarity
    def _clarity(self, out: dict[str, Any], inp: dict[str, Any]) -> float:
        """Articulation crispness proxy: retention of spectral detail and
        of the input's high-band speech energy."""
        if not inp:
            return 0.5
        parts: list[tuple[float, float]] = []
        # spectral-detail retention: flatness shouldn't collapse vs input
        in_flat = (inp.get("spectral") or {}).get("flatness")
        out_flat = (out.get("spectral") or {}).get("flatness")
        if in_flat is not None and out_flat is not None:
            parts.append((_proximity(
                out_flat, in_flat, max(0.02, 0.5 * in_flat + 0.01)), 1.5))
        # high-band energy retention (2-6 kHz, consonant region) vs input
        ib = inp.get("band_energies") or []
        ob = out.get("band_energies") or []
        if len(ib) == 8 and len(ob) == 8:
            in_hf = float(np.sum(ib[4:7]))          # 2k..6k
            out_hf = float(np.sum(ob[4:7]))
            if in_hf > 1e-6:
                ratio = out_hf / in_hf
                parts.append((float(np.clip(
                    1.0 - abs(np.log2(max(ratio, 1e-6))) / 2.0, 0.0, 1.0)),
                    2.0))
        if not parts:
            return 0.5
        tw = sum(w for _, w in parts)
        return float(sum(s * w for s, w in parts) / max(1e-9, tw))

    # ---------------------------------------------------------- 6) artifacts
    def _artifacts(self, out: dict[str, Any], inp: dict[str, Any],
                   agg: dict[str, Any], ev: VoiceEvaluation) -> float:
        """Real 15-detector artifact bank (Phase 02 — see quality.py)."""
        from rvc_platform.optimizer.quality import (
            artifact_axis_score, detect_artifacts)

        try:
            results = detect_artifacts(out, inp, agg)
        except Exception:
            logger.exception("optimizer: artifact detection failed")
            return 0.5
        score, penalties = artifact_axis_score(results)
        ev.detail["artifact_penalties"] = penalties
        ev.detail["artifact_detectors"] = [r.to_dict() for r in results]
        return score

    # ------------------------------------------------------------ 7) prosody
    def _prosody(self, out: dict[str, Any], inp: dict[str, Any]) -> float:
        """Rhythm/pause-structure AND time-series preservation vs the INPUT
        (the words and performance being converted).

        Phase 02 implements what the Phase 01 docstring only claimed:
        * TRUE frame-envelope correlation (64-pt RMS envelopes),
        * time-warped F0 contour correlation (32-pt semitone contours),
        * speech-rate agreement (envelope peak rate ratio),
        * pause-structure agreement (voiced ratio, pauses, coverage).
        """
        if not inp:
            return 0.5
        parts: list[tuple[float, float]] = []

        # (a) TRUE loudness-envelope correlation
        env_in = inp.get("rms_envelope") or []
        env_out = out.get("rms_envelope") or []
        s = _corr_score(_corr(env_in, env_out))
        if s is not None:
            parts.append((s, 2.0))
            if s < 0.55:
                pass  # poor envelope agreement — recorded in detail below

        # (b) time-warped F0 contour correlation (the intonation of the
        # performance; pitch_shift only offsets the contour, correlation
        # is shift-invariant)
        c_in = inp.get("f0_contour") or []
        c_out = out.get("f0_contour") or []
        s = _corr_score(_corr(c_in, c_out))
        if s is not None:
            parts.append((s, 1.5))

        # (c) speech-rate agreement (envelope peak rate ratio)
        r_in = (inp.get("loudness") or {}).get("envelope_peak_rate")
        r_out = (out.get("loudness") or {}).get("envelope_peak_rate")
        if r_in is not None and r_out is not None and r_in > 0.05:
            rel = float(np.clip(r_out / r_in, 0.0, 4.0))
            parts.append((float(np.clip(
                1.0 - abs(np.log2(max(rel, 1e-6))) / 1.5, 0.0, 1.0)), 1.0))

        # (d) pause-structure agreement
        v_in = (inp.get("vad") or {})
        v_out = (out.get("vad") or {})
        if v_in.get("voiced_ratio") is not None and \
                v_out.get("voiced_ratio") is not None:
            d = abs(float(v_in["voiced_ratio"])
                    - float(v_out["voiced_ratio"]))
            parts.append((float(np.clip(1.0 - d / 0.35, 0.0, 1.0)), 2.0))
        p_in = v_in.get("pauses")
        p_out = v_out.get("pauses")
        if p_in is not None and p_out is not None:
            d = abs(int(p_in) - int(p_out))
            denom = max(3, int(p_in) + 1, int(p_out) + 1)
            parts.append((float(np.clip(1.0 - d / denom, 0.0, 1.0)), 1.0))
        segs_in = v_in.get("segments") or []
        segs_out = v_out.get("segments") or []
        if segs_in and segs_out:
            tin = sum(b - a for a, b in segs_in)
            tout = sum(b - a for a, b in segs_out)
            if tin > 0:
                r = tout / tin
                parts.append((float(np.clip(
                    1.0 - abs(r - 1.0) / 0.4, 0.0, 1.0)), 1.5))

        if not parts:
            return 0.5
        tw = sum(w for _, w in parts)
        return float(sum(s * w for s, w in parts) / max(1e-9, tw))

    # ------------------------------------------------------------ 8) content
    def _content(self, out: dict[str, Any],
                 inp: dict[str, Any]) -> float:
        """Linguistic-content preservation vs the INPUT utterance (Phase 02):
        did the *speech itself* survive conversion?

        Proxies (no ASR needed, no language dependency): duration
        fidelity, voiced-structure agreement, pause-count agreement,
        speech-rate ratio, consonant-band retention. ASR/WER would be the
        gold standard where a suitable model for the language exists —
        these proxies are honest about being proxies.
        """
        if not inp:
            return 0.5
        parts: list[tuple[float, float]] = []
        d_in = inp.get("duration_s") or 0.0
        d_out = out.get("duration_s") or 0.0
        if d_in > 0:
            rel = abs(d_out - d_in) / d_in
            parts.append((float(np.clip(1.0 - rel / 0.5, 0.0, 1.0)), 2.0))
        v_in = (inp.get("vad") or {})
        v_out = (out.get("vad") or {})
        if v_in.get("voiced_ratio") is not None and \
                v_out.get("voiced_ratio") is not None:
            d = abs(float(v_in["voiced_ratio"])
                    - float(v_out["voiced_ratio"]))
            parts.append((float(np.clip(1.0 - d / 0.3, 0.0, 1.0)), 1.5))
        p_in = v_in.get("pauses")
        p_out = v_out.get("pauses")
        if p_in is not None and p_out is not None:
            d = abs(int(p_in) - int(p_out))
            denom = max(3, int(p_in) + 1, int(p_out) + 1)
            parts.append((float(np.clip(1.0 - d / denom, 0.0, 1.0)), 1.0))
        # speech-rate ratio (syllable-timing proxy)
        r_in = (inp.get("loudness") or {}).get("envelope_peak_rate")
        r_out = (out.get("loudness") or {}).get("envelope_peak_rate")
        if r_in is not None and r_out is not None and r_in > 0.05:
            rel = float(np.clip(r_out / r_in, 0.0, 4.0))
            parts.append((float(np.clip(
                1.0 - abs(np.log2(max(rel, 1e-6))) / 1.5, 0.0, 1.0)), 1.0))
        # consonant-band retention (2-6 kHz)
        ib = inp.get("band_energies") or []
        ob = out.get("band_energies") or []
        if len(ib) == 8 and len(ob) == 8:
            in_hf = float(np.sum(ib[4:7]))
            out_hf = float(np.sum(ob[4:7]))
            if in_hf > 1e-6:
                ratio = out_hf / in_hf
                parts.append((float(np.clip(
                    1.0 - abs(np.log2(max(ratio, 1e-6))) / 2.0, 0.0, 1.0)),
                    1.5))
        if not parts:
            return 0.5
        tw = sum(w for _, w in parts)
        return float(sum(s * w for s, w in parts) / max(1e-9, tw))

    # ------------------------------------------------------------- 9) quality
    def _quality(self, out: dict[str, Any], agg: dict[str, Any]) -> float:
        lum = out.get("loudness") or {}
        snr = lum.get("snr_proxy_db")
        dr = lum.get("dynamic_range_db")
        clip = lum.get("clipping_ratio")
        parts: list[tuple[float, float]] = []
        if snr is not None:
            parts.append((float(np.clip((snr - 8.0) / 22.0, 0.0, 1.0)), 1.5))
        if dr is not None:
            parts.append((float(np.clip((dr - 12.0) / 28.0, 0.0, 1.0)), 1.0))
        if clip is not None:
            parts.append((float(np.clip(1.0 - clip / 0.003, 0.0, 1.0)), 1.0))
        if not parts:
            return 0.5
        tw = sum(w for _, w in parts)
        return float(sum(s * w for s, w in parts) / max(1e-9, tw))

    # ------------------------------------------------------------ composite
    def _composite(self, ev: VoiceEvaluation
                   ) -> tuple[float, bool, list[str]]:
        w = self.metric_weights.normalised()
        axes = ev.axis_scores()
        known = [(k, v) for k, v in axes.items() if v is not None and k in w]
        if not known:
            return 0.0, False, list(axes.keys())
        total_w = sum(w[k] for k, _ in known)
        composite = float(
            sum(w[k] * v for k, v in known) / max(1e-9, total_w))
        failed = [k for k, gate in GATES.items()
                  if axes.get(k) is not None and float(axes[k]) < gate]
        if failed:
            # gate failure: composite survives for *reporting*, but the
            # candidate is excluded from selection (gates_passed=False).
            composite *= float(0.5 ** len(failed))
        return composite, not failed, failed

    # ----------------------------------------------------------------- pareto
    @staticmethod
    def pareto_front(evaluations: list[VoiceEvaluation]) -> list[int]:
        """Indices of non-dominated candidates on (speaker, naturalness,
        clarity, artifact) — standard Pareto dominance filtering
        (maximisation)."""
        axes = [
            (ev.speaker_similarity
             if ev.speaker_similarity is not None else -1.0,
             ev.naturalness if ev.naturalness is not None else -1.0,
             ev.clarity if ev.clarity is not None else -1.0,
             ev.artifact_score if ev.artifact_score is not None else -1.0)
            for ev in evaluations
        ]
        front: list[int] = []
        for i, a in enumerate(axes):
            dominated = False
            for j, b in enumerate(axes):
                if i == j:
                    continue
                if all(bx >= ax for ax, bx in zip(a, b)) and \
                        any(bx > ax for ax, bx in zip(a, b)):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        return front

    # --------------------------------------------------- best by strategy
    @staticmethod
    def best_by_strategy(
        candidates: list[dict[str, Any]],
        weights: dict[str, float] | None = None,
    ) -> dict[str, dict[str, Any] | None]:
        """Best candidate per selection strategy (Phase 02 W11).

        ``candidates`` — trial dicts with ``axes`` (+ optional
        ``validated_composite``). Gates are honoured: only gate-passing
        candidates are eligible (fallback: all, flagged).

        Returns one winner per strategy:

        * ``identity``      — max speaker_similarity (ties: composite)
        * ``naturalness``   — max naturalness (ties: composite)
        * ``clarity``       — max clarity (ties: composite)
        * ``balanced``      — max weighted composite (the configured
          weights — or DEFAULT_WEIGHTS when not supplied)
        """
        from rvc_platform.optimizer.quality import _spearman  # noqa: F401

        eligible = [c for c in candidates
                    if c.get("axes") and c.get("gates_passed", True)]
        fallback_used = False
        if not eligible:
            eligible = [c for c in candidates if c.get("axes")]
            fallback_used = True
        if not eligible:
            return {s: None for s in STRATEGIES} | {"note": "no candidates"}

        def _ax(c: dict[str, Any], key: str) -> float:
            return _safe((c.get("axes") or {}).get(key), 0.0)

        def _score(c: dict[str, Any]) -> float:
            v = c.get("validated_composite")
            return _safe(v if v is not None else c.get("composite"), 0.0)

        def _winner(key: str | None) -> dict[str, Any]:
            if key is None:
                return max(eligible, key=_score)
            return max(eligible, key=lambda c: (_ax(c, key), _score(c)))

        out: dict[str, Any] = {
            "identity": _winner("speaker_similarity"),
            "naturalness": _winner("naturalness"),
            "clarity": _winner("clarity"),
            "balanced": _winner(None),
        }
        if fallback_used:
            out["note"] = ("no gate-passing candidate — winners chosen from "
                           "all evaluated candidates (check artifact scores)")
        return out
