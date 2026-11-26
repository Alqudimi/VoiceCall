"""Advanced audio analysis engine — the optimizer's sensory system.

Extracts, for ANY 16 kHz mono utterance, the feature set the evaluation
engine needs (all JSON-serialisable via ``AnalysisResult.to_dict``):

* Speaker identity — neural speaker-verification embeddings (WavLM x-vector
  + ECAPA-TDNN when available) via :class:`SpeakerEmbedderManager`,
  computed on **VAD-selected speech only** (silence never contributes to the
  voiceprint) and, for utterances longer than ~6 s, as windowed embeddings
  with a per-backend stability estimate (mean/std of pairwise similarity).
* F0 — contour statistics via the engine's own F0 registry (RMVPE, falling
  back to Praat ``pm``): median/mean/range (semitones), voiced ratio,
  octave-jump rate, jitter proxy (local |Δsemitones|), 24-bin pitch
  distribution, plus Phase 02 vibrato measures (rate, depth, irregularity).
* Formants — median F1..F4 (Praat Burg on voiced frames; guarded).
* Spectral envelope — MFCC(20) mean/std + 8-band energy profile +
  centroid / spread / rolloff / flatness + sibilance ratio (5–8 kHz share).
* HNR — harmonics-to-noise ratio in dB (Praat harmonicity CC, with an
  autocorrelation fallback that reuses the already-extracted F0 contour —
  no double extraction).
* VAD & segmentation — adaptive energy-based voice-activity detection with
  hangover, pause count, speech segments, breath-proxy detection in pauses.
* Loudness & quality — RMS dB, peak, clipping ratio, noise-floor estimate,
  SNR proxy, dynamic range, envelope peak rate (speech-rate proxy),
  discontinuity rate (click/crackle proxy).
* Stability — duration, voiced-ratio stability across thirds.

Phase 02 changes (audited W2/W17/W20): vectorised jitter/octave-jump
computation (was O(frames) pure Python), single F0 extraction per analysis
(HNR fallback reuses the contour), vibrato/sibilance/discontinuity/breath/
envelope-peak features required by the 15 artifact detectors, and
speech-only embeddings with windowed stability.

Everything is computed at the canonical 16 kHz analysis rate so outputs at
any model sample rate compare against the reference profile consistently.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SR_ANALYSIS = 16_000
_HOP = 256            # 16 ms frames for spectral measures
_FRAME = 1024         # 64 ms FFT window
_FLOOR_DB = -80.0

# 8-band energy profile (Hz edges at 16 kHz analysis rate)
_BAND_EDGES = (0, 250, 500, 1000, 2000, 3000, 4000, 6000, 8000)

# windowed-embedding stability: window length / hop in seconds
_EMB_WIN_S = 4.0
_EMB_WIN_HOP_S = 2.0
_EMB_MAX_WINDOWS = 8          # cap the cost for very long utterances


def _f(x: Any, nd: int = 4) -> float | None:
    """JSON-safe float (None for NaN/inf)."""
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return round(v, nd)
    except (TypeError, ValueError):
        return None


@dataclass
class AnalysisResult:
    """Everything the evaluation engine needs about one utterance."""

    source: str = ""                       # file name / "trial" label
    duration_s: float = 0.0
    # ---- F0 / pitch ----
    f0_median_hz: float | None = None
    f0_mean_hz: float | None = None
    f0_range_semitones: float | None = None      # p95 - p5 of voiced
    f0_voiced_ratio: float | None = None
    f0_octave_jump_rate: float | None = None     # jumps >= 8 st / voiced frame
    f0_jitter_semitones: float | None = None      # mean local |Δst|
    f0_histogram: list[float] = field(default_factory=list)  # 24 bins, sums to 1
    # ---- vibrato (Phase 02) ----
    f0_vibrato_rate_hz: float | None = None       # dominant 4-8 Hz modulation
    f0_vibrato_depth_semitones: float | None = None
    f0_vibrato_irregularity: float | None = None  # 0 regular .. 1 irregular
    # ---- timbre / spectral ----
    mfcc_mean: list[float] = field(default_factory=list)      # 20 coeffs
    mfcc_std: list[float] = field(default_factory=list)
    band_energies: list[float] = field(default_factory=list)  # 8 bands, sum 1
    spectral_centroid_hz: float | None = None
    spectral_spread_hz: float | None = None
    spectral_rolloff85_hz: float | None = None
    spectral_flatness: float | None = None
    sibilance_ratio: float | None = None          # 5-8 kHz share of energy
    # ---- formants ----
    formants_hz: list[float | None] = field(default_factory=list)  # F1..F4
    # ---- voice quality ----
    hnr_db: float | None = None
    shimmer_db: float | None = None
    # ---- VAD / segmentation ----
    vad_voiced_ratio: float | None = None
    vad_segments: list[list[float]] = field(default_factory=list)  # [start,end] s
    vad_pauses: int | None = None
    breath_count: int | None = None              # breath-like pause events
    # ---- loudness / quality ----
    rms_db: float | None = None
    peak: float | None = None
    clipping_ratio: float | None = None
    noise_floor_db: float | None = None
    snr_proxy_db: float | None = None
    dynamic_range_db: float | None = None
    envelope_peak_rate: float | None = None       # envelope peaks / speech sec
    discontinuity_rate: float | None = None       # sample jumps / second
    # ---- prosody time series (Phase 02: true correlation metrics) ----
    rms_envelope: list[float] = field(default_factory=list)   # 64 pts, 0..1
    f0_contour: list[float] = field(default_factory=list)     # 32 pts, semitones
    # ---- speaker identity ----
    embeddings: dict[str, list[float]] = field(default_factory=dict)
    # per-backend windowed embedding stability (Phase 02)
    embedding_stability: dict[str, dict[str, float | None]] = field(
        default_factory=dict)
    # ---- reliability ----
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "duration_s": _f(self.duration_s, 3),
            "f0": {
                "median_hz": _f(self.f0_median_hz, 2),
                "mean_hz": _f(self.f0_mean_hz, 2),
                "range_semitones": _f(self.f0_range_semitones, 3),
                "voiced_ratio": _f(self.f0_voiced_ratio, 4),
                "octave_jump_rate": _f(self.f0_octave_jump_rate, 5),
                "jitter_semitones": _f(self.f0_jitter_semitones, 4),
                "histogram": [_f(x, 5) for x in self.f0_histogram],
                "vibrato_rate_hz": _f(self.f0_vibrato_rate_hz, 2),
                "vibrato_depth_semitones": _f(
                    self.f0_vibrato_depth_semitones, 3),
                "vibrato_irregularity": _f(self.f0_vibrato_irregularity, 3),
            },
            "mfcc_mean": [_f(x, 3) for x in self.mfcc_mean],
            "mfcc_std": [_f(x, 3) for x in self.mfcc_std],
            "band_energies": [_f(x, 5) for x in self.band_energies],
            "spectral": {
                "centroid_hz": _f(self.spectral_centroid_hz, 1),
                "spread_hz": _f(self.spectral_spread_hz, 1),
                "rolloff85_hz": _f(self.spectral_rolloff85_hz, 1),
                "flatness": _f(self.spectral_flatness, 6),
            },
            "sibilance_ratio": _f(self.sibilance_ratio, 5),
            "formants_hz": [_f(x, 1) for x in self.formants_hz],
            "hnr_db": _f(self.hnr_db, 2),
            "shimmer_db": _f(self.shimmer_db, 3),
            "vad": {
                "voiced_ratio": _f(self.vad_voiced_ratio, 4),
                "pauses": self.vad_pauses,
                "segments": [[_f(a, 3), _f(b, 3)]
                             for a, b in self.vad_segments[:200]],
                "breath_count": self.breath_count,
            },
            "loudness": {
                "rms_db": _f(self.rms_db, 2),
                "peak": _f(self.peak, 4),
                "clipping_ratio": _f(self.clipping_ratio, 6),
                "noise_floor_db": _f(self.noise_floor_db, 2),
                "snr_proxy_db": _f(self.snr_proxy_db, 2),
                "dynamic_range_db": _f(self.dynamic_range_db, 2),
                "envelope_peak_rate": _f(self.envelope_peak_rate, 3),
                "discontinuity_rate": _f(self.discontinuity_rate, 3),
            },
            "rms_envelope": [_f(x, 4) for x in self.rms_envelope],
            "f0_contour": [_f(x, 3) for x in self.f0_contour],
            "embeddings": {
                name: [_f(x, 5) for x in vec]
                for name, vec in self.embeddings.items()
            },
            "embedding_stability": {
                name: {
                    "mean_pairwise_similarity": _f(v.get(
                        "mean_pairwise_similarity"), 4),
                    "std_pairwise_similarity": _f(v.get(
                        "std_pairwise_similarity"), 4),
                    "n_windows": int(v.get("n_windows", 0)),
                } for name, v in self.embedding_stability.items()
            },
            "warnings": list(self.warnings),
        }


class AudioAnalyzer:
    """Stateful analyser (shares embedders + F0 extractor across calls)."""

    def __init__(self, embedder: Any | None = None, device: str = "cpu") -> None:
        self.embedder = embedder
        self.device = device
        self._f0_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------ F0
    def _f0_extractor(self):
        """RMVPE when the asset exists (matches the engine's default),
        else the dependency-free Praat autocorrelation ('pm')."""
        if "active" in self._f0_cache:
            return self._f0_cache["active"]
        try:
            from rvc_platform.engine.rvc import f0 as f0_mod

            ext = f0_mod.get_f0_extractor("rmvpe", self.device, False)
            self._f0_cache["active"] = ext
            return ext
        except Exception:
            try:
                from rvc_platform.engine.rvc import f0 as f0_mod

                ext = f0_mod.get_f0_extractor("pm", "cpu", False)
                self._f0_cache["active"] = ext
                return ext
            except Exception as exc:
                logger.warning("optimizer: no F0 extractor available (%s)", exc)
                self._f0_cache["active"] = None
                return None

    def extract_f0(self, wav16k: np.ndarray) -> np.ndarray:
        ext = self._f0_extractor()
        if ext is None:
            return np.zeros(0, dtype=np.float32)
        try:
            f0 = np.asarray(ext.extract(np.ascontiguousarray(wav16k)),
                             dtype=np.float32)
        except Exception:
            logger.exception("optimizer: F0 extraction failed")
            return np.zeros(0, dtype=np.float32)
        return f0

    # ------------------------------------------------------------------ VAD
    def _frame_rms_db(self, wav: np.ndarray, hop: int = _HOP) -> np.ndarray:
        n = wav.size // hop
        if n <= 0:
            return np.zeros(0)
        frames = wav[: n * hop].reshape(n, hop)
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
        return 20.0 * np.log10(rms + 1e-10)

    def voice_activity(self, wav: np.ndarray) -> np.ndarray:
        """Adaptive energy VAD (bool per 16 ms frame): threshold = max(noise
        floor + 8 dB, p95 - 32 dB) with 1-frame hangover — no external
        dependency, robust to moderate recording levels."""
        rms_db = self._frame_rms_db(wav)
        if rms_db.size == 0:
            return np.zeros(0, dtype=bool)
        floor = float(np.percentile(rms_db, 10))
        p95 = float(np.percentile(rms_db, 95))
        thresh = max(floor + 8.0, p95 - 32.0, _FLOOR_DB + 8.0)
        vad = rms_db > thresh
        # hangover: extend each voiced run by one frame both ways
        if vad.size > 2:
            ext = vad.copy()
            ext[1:] |= vad[:-1]
            ext[:-1] |= vad[1:]
            vad = ext
        return vad

    def segment_speech(self, wav: np.ndarray, merge_gap_s: float = 0.25,
                       min_len_s: float = 0.12) -> list[tuple[float, float]]:
        """Speech segments in seconds (merged across short pauses)."""
        vad = self.voice_activity(wav)
        if vad.size == 0:
            return []
        hop_s = _HOP / SR_ANALYSIS
        segments: list[list[float]] = []
        start = None
        for i, v in enumerate(vad):
            if v and start is None:
                start = i * hop_s
            elif not v and start is not None:
                segments.append([start, i * hop_s])
                start = None
        if start is not None:
            segments.append([start, vad.size * hop_s])
        merged: list[list[float]] = []
        for seg in segments:
            if merged and seg[0] - merged[-1][1] <= merge_gap_s:
                merged[-1][1] = seg[1]
            else:
                merged.append(list(seg))
        return [(a, b) for a, b in merged if (b - a) >= min_len_s]

    def speech_only(self, wav: np.ndarray, max_s: float = 30.0) -> np.ndarray:
        """Concatenated VAD-selected speech audio (silence removed).

        Identity embedding input: pauses and silence never contribute to
        the voiceprint; capped for cost on very long utterances.
        """
        segs = self.segment_speech(wav)
        if not segs:
            return np.asarray(wav, dtype=np.float32)
        pieces = []
        budget = int(max_s * SR_ANALYSIS)
        for a, b in segs:
            if budget <= 0:
                break
            piece = wav[int(a * SR_ANALYSIS): int(b * SR_ANALYSIS)]
            pieces.append(piece[:budget])
            budget -= piece.size
        if not pieces:
            return np.asarray(wav, dtype=np.float32)
        out = np.concatenate(pieces)
        return out if out.size >= SR_ANALYSIS // 4 else \
            np.asarray(wav, dtype=np.float32)

    # ------------------------------------------------------------------ core
    def analyze(self, wav16k: np.ndarray, source: str = "",
                with_embeddings: bool = True) -> AnalysisResult:
        wav = np.asarray(wav16k, dtype=np.float32).ravel()
        res = AnalysisResult(source=source)
        res.duration_s = wav.size / SR_ANALYSIS
        if wav.size < SR_ANALYSIS // 4:
            res.warnings.append("audio too short (<0.25s) — metrics unreliable")
            return res

        f0 = self.extract_f0(wav)
        voiced_any = self._analyze_f0(wav, res, f0)
        self._analyze_time_series(wav, res, f0)
        self._analyze_spectral(wav, res)
        if voiced_any:
            self._analyze_formants(wav, res)
        self._analyze_loudness_quality(wav, res, f0)
        self._analyze_vad(wav, res)
        if with_embeddings and self.embedder is not None:
            self._analyze_embeddings(wav, res)
            if not res.embeddings:
                res.warnings.append(
                    "no neural speaker embedder available — identity scoring "
                    "degraded (reliability low)")
        if res.duration_s < 1.5:
            res.warnings.append(
                f"short utterance ({res.duration_s:.1f}s) — embedding similarity "
                "is less stable; prefer >= 3s segments")
        return res

    def analyze_file(self, path: Path | str,
                     with_embeddings: bool = True) -> AnalysisResult:
        import librosa

        path = Path(path)
        wav, _ = librosa.load(str(path), sr=SR_ANALYSIS, mono=True)
        return self.analyze(wav, source=path.name,
                             with_embeddings=with_embeddings)

    # --------------------------------------------------------- embeddings
    def _analyze_embeddings(self, wav: np.ndarray,
                            res: AnalysisResult) -> None:
        """Speech-only embedding + windowed stability (Phase 02 W2)."""
        try:
            speech = self.speech_only(wav)
            embs = self.embedder.embeddings(speech)
            res.embeddings = {k: v.tolist() for k, v in embs.items()}
        except Exception:
            logger.exception("optimizer: embedding extraction failed")
            res.warnings.append("speaker embedding extraction failed")
            return
        if not res.embeddings:
            return
        # windowed stability for longer utterances
        try:
            win = int(_EMB_WIN_S * SR_ANALYSIS)
            hop = int(_EMB_WIN_HOP_S * SR_ANALYSIS)
            if speech.size >= 2 * win:                     # >= ~8 s speech
                windows: list[np.ndarray] = []
                pos = 0
                while pos + win <= speech.size and \
                        len(windows) < _EMB_MAX_WINDOWS:
                    windows.append(speech[pos: pos + win])
                    pos += hop
                if len(windows) >= 3:
                    stability: dict[str, dict[str, float | None]] = {}
                    for name in res.embeddings:
                        vecs = [np.asarray(self.embedder.embeddings(w)[name],
                                           dtype=np.float64)
                                for w in windows]
                        vecs = [v for v in vecs if v.size]
                        if len(vecs) < 3:
                            continue
                        mat = np.vstack(vecs)
                        mat = mat / np.maximum(
                            np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
                        sims = (mat @ mat.T)[
                            np.triu_indices(len(vecs), k=1)]
                        stability[name] = {
                            "mean_pairwise_similarity": float(np.mean(sims)),
                            "std_pairwise_similarity": float(np.std(sims)),
                            "n_windows": len(vecs),
                        }
                    if stability:
                        res.embedding_stability = stability
        except Exception:
            logger.debug("optimizer: windowed stability failed", exc_info=True)

    # ------------------------------------------------------------------ F0
    def _analyze_f0(self, wav: np.ndarray, res: AnalysisResult,
                    f0: np.ndarray | None = None) -> bool:
        if f0 is None:
            f0 = self.extract_f0(wav)
        if f0.size < 3:
            res.warnings.append("F0 extraction produced no frames")
            return False
        voiced = f0[f0 > 20.0]
        res.f0_voiced_ratio = float(voiced.size / f0.size)
        if voiced.size < 3:
            res.warnings.append(
                "no voiced frames detected (unvoiced/silent audio?)")
            return False
        res.f0_median_hz = float(np.median(voiced))
        res.f0_mean_hz = float(np.mean(voiced))
        log_f0 = np.log2(voiced)
        res.f0_range_semitones = float(
            12.0 * (np.percentile(log_f0, 95) - np.percentile(log_f0, 5)))

        # ---- vectorised octave-jump / jitter (Phase 02 W17) ----
        v_mask = f0 > 20.0
        pairs = v_mask[:-1] & v_mask[1:]          # consecutive voiced frames
        n_pairs = int(np.count_nonzero(pairs))
        if n_pairs:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(
                    pairs, f0[1:] / np.maximum(f0[:-1], 1e-9), 1.0)
                dst = np.where(pairs, np.abs(12.0 * np.log2(np.maximum(
                    ratio, 1e-9))), 0.0)
            dst = dst[pairs]
            res.f0_octave_jump_rate = float(
                np.count_nonzero(dst >= 8.0) / n_pairs)
            res.f0_jitter_semitones = float(np.mean(dst))

        self._analyze_vibrato(f0, res)

        # 24-bin pitch distribution over voiced semitones (50..1100 Hz warp)
        lo = np.log2(50.0)
        hi = np.log2(1100.0)
        edges = np.linspace(lo, hi, 25)
        hist, _ = np.histogram(np.log2(voiced), bins=edges)
        total = hist.sum()
        res.f0_histogram = (hist / total).tolist() if total else []
        return True

    @staticmethod
    def _analyze_vibrato(f0: np.ndarray, res: AnalysisResult) -> None:
        """Dominant 4–8 Hz F0 modulation (vibrato) + irregularity.

        Works on the voiced contour in semitones (voiced frames only,
        interpolated across short unvoiced gaps so the modulation spectrum
        is not fragmented by pauses).
        """
        try:
            mask = f0 > 20.0
            if np.count_nonzero(mask) < 32:
                return
            st = 12.0 * np.log2(np.maximum(f0, 1e-9))
            idx = np.where(mask)[0]
            # interpolate unvoiced gaps (contours are ~5.8ms/frame for pm)
            full_idx = np.arange(idx[0], idx[-1] + 1)
            contour = np.interp(full_idx, idx, st[idx])
            # detrend (slow pitch movement) with a moving average ~0.5 s
            frame_hz = 1.0 / 0.0058
            win = max(8, int(frame_hz * 0.5))
            if contour.size < 2 * win:
                return
            kernel = np.ones(win) / win
            trend = np.convolve(contour, kernel, mode="same")
            trend[: win // 2] = trend[win // 2]
            trend[-(win // 2):] = trend[-(win // 2)]
            resid = contour - trend
            spec = np.abs(np.fft.rfft(resid * np.hanning(resid.size)))
            freqs = np.fft.rfftfreq(resid.size, d=1.0 / frame_hz)
            band = (freqs >= 3.5) & (freqs <= 8.5)
            total_band = (freqs >= 0.5) & (freqs <= 12.0)
            total_e = float(np.sum(spec[total_band] ** 2))
            if total_e <= 1e-9:
                return
            band_e = float(np.sum(spec[band] ** 2))
            res.f0_vibrato_irregularity = round(
                float(1.0 - band_e / total_e), 3) if band_e < total_e else 0.0
            if band_e > 0.25 * total_e:
                # a dominant quasi-periodic modulation — vibrato present
                peak = int(np.argmax(np.where(band, spec, 0.0)))
                rate = float(freqs[peak])
                if 3.5 <= rate <= 8.5:
                    # depth ≈ 2 × amplitude at the peak (semitones, windowed)
                    depth = 2.0 * float(spec[peak]) / max(
                        1.0, np.hanning(resid.size).sum() / 2.0)
                    res.f0_vibrato_rate_hz = round(rate, 2)
                    res.f0_vibrato_depth_semitones = round(depth, 3)
        except Exception:
            logger.debug("optimizer: vibrato analysis failed", exc_info=True)

    # -------------------------------------------------------- time series
    def _analyze_time_series(self, wav: np.ndarray, res: AnalysisResult,
                             f0: np.ndarray | None = None) -> None:
        """Downsampled RMS envelope + F0 contour for TRUE cross-signal
        correlation in prosody/content scoring (Phase 02 W5).

        Both series are length-normalised so input/output comparisons are
        content-aligned without DTW; the F0 contour is interpolated across
        unvoiced gaps (performance continuity, not frame-level truth).
        """
        try:
            rms_db = self._frame_rms_db(wav)
            if rms_db.size:
                n = 64
                idx = np.linspace(0, rms_db.size - 1, n)
                env = np.interp(idx, np.arange(rms_db.size), rms_db)
                lo, hi = float(np.min(env)), float(np.max(env))
                env = (env - lo) / max(1e-9, hi - lo) if hi > lo \
                    else np.zeros_like(env)
                res.rms_envelope = [round(float(v), 4) for v in env]
        except Exception:
            logger.debug("optimizer: envelope series failed", exc_info=True)
        try:
            if f0 is None:
                return
            mask = f0 > 20.0
            if np.count_nonzero(mask) < 8:
                return
            st = 12.0 * np.log2(np.maximum(f0, 1e-9))
            idx = np.where(mask)[0]
            full_idx = np.arange(idx[0], idx[-1] + 1)
            contour = np.interp(full_idx, idx, st[idx])
            n = 32
            sel = np.linspace(0, contour.size - 1, n)
            resampled = np.interp(sel, np.arange(contour.size), contour)
            res.f0_contour = [round(float(v), 3) for v in resampled]
        except Exception:
            logger.debug("optimizer: f0 contour series failed", exc_info=True)

    # ------------------------------------------------------------- spectral
    def _analyze_spectral(self, wav: np.ndarray, res: AnalysisResult) -> None:
        import librosa

        # MFCC(20) — mean/std over frames
        mfcc = librosa.feature.mfcc(y=wav, sr=SR_ANALYSIS, n_mfcc=20,
                                    n_fft=_FRAME, hop_length=_HOP)
        res.mfcc_mean = mfcc.mean(axis=1).tolist()
        res.mfcc_std = mfcc.std(axis=1).tolist()
        # band energy profile
        spec = np.abs(librosa.stft(wav, n_fft=_FRAME, hop_length=_HOP)) ** 2
        freqs = librosa.fft_frequencies(sr=SR_ANALYSIS, n_fft=_FRAME)
        band_e = []
        for lo, hi in zip(_BAND_EDGES[:-1], _BAND_EDGES[1:]):
            m = (freqs >= lo) & (freqs < hi)
            band_e.append(float(spec[m].mean()) if m.any() else 0.0)
        tot = float(np.sum(band_e))
        res.band_energies = (np.asarray(band_e) / tot).tolist() \
            if tot > 0 else [0.0] * 8
        # sibilance ratio: 5–8 kHz share (fricative / S-artifact band)
        if tot > 0:
            sib = float(sum(
                e for lo, hi, e in zip(_BAND_EDGES[:-1], _BAND_EDGES[1:],
                                       band_e) if lo >= 5000))
            res.sibilance_ratio = round(sib, 5)
        # centroid / spread / rolloff / flatness on speech-energy frames
        cent = librosa.feature.spectral_centroid(S=np.abs(spec),
                                                 sr=SR_ANALYSIS)[0]
        res.spectral_centroid_hz = float(np.mean(cent))
        res.spectral_spread_hz = float(np.std(cent))
        res.spectral_rolloff85_hz = float(np.mean(
            librosa.feature.spectral_rolloff(S=np.abs(spec), sr=SR_ANALYSIS,
                                             roll_percent=0.85)[0]))
        flat = librosa.feature.spectral_flatness(S=np.abs(spec))[0]
        res.spectral_flatness = float(np.mean(flat))

    def _analyze_formants(self, wav: np.ndarray, res: AnalysisResult) -> None:
        try:
            import parselmouth

            snd = parselmouth.Sound(np.asarray(wav, dtype=np.float64),
                                    sampling_frequency=SR_ANALYSIS)
            formant = snd.to_formant_burg(time_step=0.02,
                                          max_number_of_formants=4)
            vals: list[list[float]] = [[], [], [], []]
            n = formant.get_number_of_frames()
            for i in range(1, n + 1):
                t = formant.get_time_from_frame_number(i)
                for j in range(4):
                    try:
                        v = formant.get_value_at_time(j + 1, t)
                        if v is not None and np.isfinite(v) \
                                and 90.0 < v < 7500.0:
                            vals[j].append(float(v))
                    except Exception:
                        pass
            res.formants_hz = [
                float(np.median(v)) if len(v) >= 5 else None for v in vals
            ]
        except Exception:
            logger.debug("optimizer: formant analysis unavailable", exc_info=True)
            res.formants_hz = [None] * 4

    def _analyze_hnr(self, wav: np.ndarray, res: AnalysisResult,
                     f0: np.ndarray | None = None) -> None:
        try:
            import parselmouth

            snd = parselmouth.Sound(np.asarray(wav, dtype=np.float64),
                                    sampling_frequency=SR_ANALYSIS)
            harm = snd.to_harmonicity_cc(time_step=0.02)
            vals = harm.values[0]
            vals = vals[np.isfinite(vals) & (vals > -10.0)]
            if vals.size >= 5:
                res.hnr_db = float(np.median(vals))
                return
        except Exception:
            logger.debug("optimizer: parselmouth HNR unavailable", exc_info=True)
        # fallback: autocorrelation-based HNR proxy on voiced frames
        # (Phase 02 W17: reuses the already-extracted contour — no re-run)
        try:
            if f0 is None:
                f0 = self.extract_f0(wav)
            frames = []
            hop = int(0.02 * SR_ANALYSIS)
            voiced_idx = np.where(f0 > 20.0)[0]
            if voiced_idx.size:
                for i in voiced_idx[:200]:
                    start = i * 160
                    seg = wav[start:start + hop]
                    if seg.size < hop:
                        continue
                    seg = seg - seg.mean()
                    energy = float(np.sum(seg ** 2))
                    if energy <= 0:
                        continue
                    corr = float(
                        np.correlate(seg, seg, "full")[seg.size - 1 + hop // 2])
                    frames.append(max(0.0, corr / energy))
            if frames:
                r = float(np.mean(frames))
                res.hnr_db = float(10.0 * np.log10(r / max(1e-6, 1.0 - r)))
        except Exception:
            logger.debug("optimizer: HNR fallback failed", exc_info=True)

    def _analyze_loudness_quality(self, wav: np.ndarray,
                                  res: AnalysisResult,
                                  f0: np.ndarray | None = None) -> None:
        self._analyze_hnr(wav, res, f0)
        rms_db = self._frame_rms_db(wav)
        if rms_db.size == 0:
            return
        res.rms_db = float(np.percentile(rms_db, 95))
        res.peak = float(np.max(np.abs(wav)))
        res.clipping_ratio = float(np.mean(np.abs(wav) > 0.98))
        res.noise_floor_db = float(np.percentile(rms_db, 10))
        speech_db = float(np.percentile(
            rms_db[rms_db > res.noise_floor_db + 5.0], 80)) \
            if np.any(rms_db > res.noise_floor_db + 5.0) else res.rms_db
        res.snr_proxy_db = speech_db - res.noise_floor_db
        res.dynamic_range_db = float(np.percentile(rms_db, 95)
                                     - np.percentile(rms_db, 10))
        # shimmer proxy: mean |ΔdB| between consecutive voiced frames
        vad = self.voice_activity(wav)
        idx = np.where(vad)[0]
        if idx.size >= 3:
            d = np.abs(np.diff(rms_db[idx]))
            res.shimmer_db = float(np.mean(d))
        # envelope peak rate (speech-rate proxy): local maxima of the
        # smoothed envelope inside speech, per second of speech
        if idx.size >= 8:
            env = rms_db.copy()
            k = 6                                     # ~100 ms smoothing
            if env.size > 2 * k:
                kernel = np.ones(k) / k
                env = np.convolve(env, kernel, mode="same")
            inner = idx[1:-1]
            peaks = (env[inner] > env[inner - 1]) & \
                    (env[inner] >= env[inner + 1]) & \
                    (env[inner] > float(np.median(rms_db[idx])))
            speech_s = idx.size * (_HOP / SR_ANALYSIS)
            res.envelope_peak_rate = float(np.count_nonzero(peaks)
                                           / max(1e-9, speech_s))
        # discontinuity rate: sample-to-sample jumps (clicks / crackle)
        if wav.size > 2:
            jumps = np.abs(np.diff(wav.astype(np.float64))) > 0.30
            res.discontinuity_rate = float(
                np.count_nonzero(jumps) / (wav.size / SR_ANALYSIS))

    def _analyze_vad(self, wav: np.ndarray, res: AnalysisResult) -> None:
        vad = self.voice_activity(wav)
        if vad.size == 0:
            return
        res.vad_voiced_ratio = float(np.mean(vad))
        segs = self.segment_speech(wav)
        res.vad_segments = [[a, b] for a, b in segs]
        res.vad_pauses = max(0, len(segs) - 1)
        res.breath_count = self._count_breaths(wav, segs)

    def _count_breaths(self, wav: np.ndarray,
                       segs: list[tuple[float, float]]) -> int:
        """Breath-proxy: short low-energy broadband events inside pauses.

        A breath between words is noise-like, sits well above the silence
        floor but below speech, lasts ~0.15–0.6 s and is spectrally flat.
        This is a proxy, not a certified breath detector — its only job is
        to feed the "unnatural breathing" artifact detector with evidence.
        """
        try:
            if len(segs) < 2:
                return 0
            rms_db = self._frame_rms_db(wav)
            if rms_db.size == 0:
                return 0
            floor = float(np.percentile(rms_db, 10))
            speech_levels = []
            hop_s = _HOP / SR_ANALYSIS
            for a, b in segs:
                i0, i1 = int(a / hop_s), int(b / hop_s)
                if i1 > i0:
                    speech_levels.append(float(np.percentile(rms_db[i0:i1], 60)))
            if not speech_levels:
                return 0
            speech_ref = float(np.median(speech_levels))
            lo = floor + 6.0
            hi = speech_ref - 12.0
            if hi <= lo:
                return 0
            count = 0
            for i in range(len(segs) - 1):
                gap0 = segs[i][1] + 0.05
                gap1 = segs[i + 1][0] - 0.05
                if gap1 - gap0 < 0.12:
                    continue
                i0, i1 = int(gap0 / hop_s), int(gap1 / hop_s)
                if i1 <= i0 or i1 > rms_db.size:
                    continue
                gap = rms_db[i0:i1]
                if gap.size == 0:
                    continue
                above = gap[(gap >= lo) & (gap <= hi)]
                dur = above.size * hop_s
                if 0.15 <= dur <= 0.6:
                    count += 1
            return count
        except Exception:
            logger.debug("optimizer: breath proxy failed", exc_info=True)
            return 0

    # -------------------------------------------------------------- helpers
    def split_search_validation(self, wav: np.ndarray,
                                max_search_s: float = 12.0,
                                max_validation_s: float = 12.0
                                ) -> tuple[np.ndarray, np.ndarray]:
        """Split an utterance into search + validation speech audio.

        Speech (VAD) is concatenated into a *search* clip (capped) and a
        distinct *validation* clip (capped), so the validation set never
        reuses the audio the search optimised on (anti-overfitting).
        Short inputs degrade honestly: < 3.5s -> empty validation.

        NOTE (Phase 02): this segment-level split of ONE utterance is the
        last-resort fallback. When the user supplies MULTIPLE input files
        the service assigns whole files to roles instead
        (see optimizer/robustness.py) — adjacent-segment splits of a single
        take share channel/reverb and overestimate generalisation.
        """
        segs = self.segment_speech(wav)
        if not segs:
            wav = np.asarray(wav, dtype=np.float32).ravel()
            if wav.size < SR_ANALYSIS:  # <1s: use it all for search
                return wav, np.zeros(0, dtype=np.float32)
            mid = wav.size // 2
            return wav[:mid], wav[mid:]
        pieces: list[np.ndarray] = []
        for a, b in segs:
            i0, i1 = int(a * SR_ANALYSIS), int(b * SR_ANALYSIS)
            pieces.append(wav[i0:i1])
        total = sum(p.size for p in pieces)
        if total < int(3.5 * SR_ANALYSIS) or len(pieces) < 2:
            # not enough material for a held-out set
            search = np.concatenate(pieces)[: int(max_search_s * SR_ANALYSIS)]
            return search, np.zeros(0, dtype=np.float32)
        # interleave assignment: alternate segments -> search/validation
        search_parts, val_parts = [], []
        for i, p in enumerate(pieces):
            (search_parts if i % 2 == 0 else val_parts).append(p)
        search = np.concatenate(search_parts)[: int(max_search_s * SR_ANALYSIS)]
        validation = np.concatenate(val_parts)[
            : int(max_validation_s * SR_ANALYSIS)]
        return search, validation
