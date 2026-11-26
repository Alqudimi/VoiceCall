"""Advanced audio analysis engine — the optimizer's sensory system.

Extracts, for ANY 16 kHz mono utterance, the feature set the evaluation
engine needs (all JSON-serialisable via ``AnalysisResult.to_dict``):

* Speaker identity — neural speaker-verification embeddings (WavLM x-vector
  + ECAPA-TDNN when available) via :class:`SpeakerEmbedderManager`.
* F0 — contour statistics via the engine's own F0 registry (RMVPE, falling
  back to Praat ``pm``): median/mean/range (semitones), voiced ratio,
  octave-jump rate, jitter proxy (local |Δsemitones|), 24-bin pitch
  distribution (voiced semitones) for distribution matching.
* Formants — median F1..F4 (Praat Burg on voiced frames; guarded).
* Spectral envelope — mel-projection-free 8-band energy profile +
  MFCC(20) mean/std + centroid / spread / rolloff / flatness.
* HNR — harmonics-to-noise ratio in dB (Praat harmonicity CC, with an
  autocorrelation-based fallback).
* VAD & segmentation — adaptive energy-based voice-activity detection with
  hangover, pause count, speech segments.
* Loudness & quality — RMS dB, peak, clipping ratio, noise-floor estimate
  (10th-percentile frame RMS), SNR proxy, dynamic range.
* Stability — duration, voiced-ratio stability across thirds.

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
    # ---- timbre / spectral ----
    mfcc_mean: list[float] = field(default_factory=list)      # 20 coeffs
    mfcc_std: list[float] = field(default_factory=list)
    band_energies: list[float] = field(default_factory=list)  # 8 bands, sum 1
    spectral_centroid_hz: float | None = None
    spectral_spread_hz: float | None = None
    spectral_rolloff85_hz: float | None = None
    spectral_flatness: float | None = None
    # ---- formants ----
    formants_hz: list[float | None] = field(default_factory=list)  # F1..F4
    # ---- voice quality ----
    hnr_db: float | None = None
    shimmer_db: float | None = None
    # ---- VAD / segmentation ----
    vad_voiced_ratio: float | None = None
    vad_segments: list[list[float]] = field(default_factory=list)  # [start,end] s
    vad_pauses: int | None = None
    # ---- loudness / quality ----
    rms_db: float | None = None
    peak: float | None = None
    clipping_ratio: float | None = None
    noise_floor_db: float | None = None
    snr_proxy_db: float | None = None
    dynamic_range_db: float | None = None
    # ---- speaker identity ----
    embeddings: dict[str, list[float]] = field(default_factory=dict)
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
            "formants_hz": [_f(x, 1) for x in self.formants_hz],
            "hnr_db": _f(self.hnr_db, 2),
            "shimmer_db": _f(self.shimmer_db, 3),
            "vad": {
                "voiced_ratio": _f(self.vad_voiced_ratio, 4),
                "pauses": self.vad_pauses,
                "segments": [[_f(a, 3), _f(b, 3)] for a, b in self.vad_segments[:200]],
            },
            "loudness": {
                "rms_db": _f(self.rms_db, 2),
                "peak": _f(self.peak, 4),
                "clipping_ratio": _f(self.clipping_ratio, 6),
                "noise_floor_db": _f(self.noise_floor_db, 2),
                "snr_proxy_db": _f(self.snr_proxy_db, 2),
                "dynamic_range_db": _f(self.dynamic_range_db, 2),
            },
            "embeddings": {
                name: [_f(x, 5) for x in vec] for name, vec in self.embeddings.items()
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
            f0 = np.asarray(ext.extract(np.ascontiguousarray(wav16k)), dtype=np.float32)
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

    # ------------------------------------------------------------------ core
    def analyze(self, wav16k: np.ndarray, source: str = "",
                with_embeddings: bool = True) -> AnalysisResult:
        wav = np.asarray(wav16k, dtype=np.float32).ravel()
        res = AnalysisResult(source=source)
        res.duration_s = wav.size / SR_ANALYSIS
        if wav.size < SR_ANALYSIS // 4:
            res.warnings.append("audio too short (<0.25s) — metrics unreliable")
            return res

        voiced_any = self._analyze_f0(wav, res)
        self._analyze_spectral(wav, res)
        if voiced_any:
            self._analyze_formants(wav, res)
        self._analyze_loudness_quality(wav, res)
        self._analyze_vad(wav, res)
        if with_embeddings and self.embedder is not None:
            try:
                embs = self.embedder.embeddings(wav)
                res.embeddings = {k: v.tolist() for k, v in embs.items()}
            except Exception:
                logger.exception("optimizer: embedding extraction failed")
                res.warnings.append("speaker embedding extraction failed")
            if not res.embeddings:
                res.warnings.append(
                    "no neural speaker embedder available — identity scoring "
                    "degraded (reliability low)")
        if res.duration_s < 1.5:
            res.warnings.append(
                f"short utterance ({res.duration_s:.1f}s) — embedding similarity "
                "is less stable; prefer >= 3s segments")
        return res

    def analyze_file(self, path: Path | str, with_embeddings: bool = True) -> AnalysisResult:
        import librosa

        path = Path(path)
        wav, _ = librosa.load(str(path), sr=SR_ANALYSIS, mono=True)
        return self.analyze(wav, source=path.name, with_embeddings=with_embeddings)

    # ------------------------------------------------------------- blocks
    def _analyze_f0(self, wav: np.ndarray, res: AnalysisResult) -> bool:
        f0 = self.extract_f0(wav)
        if f0.size < 3:
            res.warnings.append("F0 extraction produced no frames")
            return False
        voiced = f0[f0 > 20.0]
        res.f0_voiced_ratio = float(voiced.size / f0.size)
        if voiced.size < 3:
            res.warnings.append("no voiced frames detected (unvoiced/silent audio?)")
            return False
        res.f0_median_hz = float(np.median(voiced))
        res.f0_mean_hz = float(np.mean(voiced))
        log_f0 = np.log2(voiced)
        res.f0_range_semitones = float(
            12.0 * (np.percentile(log_f0, 95) - np.percentile(log_f0, 5)))
        # octave-jump rate on the full contour (voiced neighbours)
        v = (f0 > 20.0).astype(np.float64)
        jumps = 0.0
        pairs = 0
        for i in range(1, f0.size):
            if f0[i - 1] > 20.0 and f0[i] > 20.0:
                pairs += 1
                if abs(12.0 * np.log2(f0[i] / f0[i - 1])) >= 8.0:
                    jumps += 1.0
        res.f0_octave_jump_rate = float(jumps / max(1, pairs))
        # jitter proxy: mean local |Δsemitones| over voiced pairs
        if pairs:
            diffs = []
            for i in range(1, f0.size):
                if f0[i - 1] > 20.0 and f0[i] > 20.0:
                    diffs.append(abs(12.0 * np.log2(f0[i] / f0[i - 1])))
            res.f0_jitter_semitones = float(np.mean(diffs))
        # 24-bin pitch distribution over voiced semitones (50..1100 Hz warp)
        lo = np.log2(50.0)
        hi = np.log2(1100.0)
        edges = np.linspace(lo, hi, 25)
        hist, _ = np.histogram(np.log2(voiced), bins=edges)
        total = hist.sum()
        res.f0_histogram = (hist / total).tolist() if total else []
        return True

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
        res.band_energies = (np.asarray(band_e) / tot).tolist() if tot > 0 else [0.0] * 8
        # centroid / spread / rolloff / flatness on speech-energy frames
        cent = librosa.feature.spectral_centroid(S=np.abs(spec), sr=SR_ANALYSIS)[0]
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
            formant = snd.to_formant_burg(time_step=0.02, max_number_of_formants=4)
            vals: list[list[float]] = [[], [], [], []]
            n = formant.get_number_of_frames()
            for i in range(1, n + 1):
                t = formant.get_time_from_frame_number(i)
                for j in range(4):
                    try:
                        v = formant.get_value_at_time(j + 1, t)
                        if v is not None and np.isfinite(v) and 90.0 < v < 7500.0:
                            vals[j].append(float(v))
                    except Exception:
                        pass
            res.formants_hz = [
                float(np.median(v)) if len(v) >= 5 else None for v in vals
            ]
        except Exception:
            logger.debug("optimizer: formant analysis unavailable", exc_info=True)
            res.formants_hz = [None] * 4

    def _analyze_hnr(self, wav: np.ndarray, res: AnalysisResult) -> None:
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
        try:
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
                    corr = float(np.correlate(seg, seg, "full")[seg.size - 1 + hop // 2])
                    frames.append(max(0.0, corr / energy))
            if frames:
                r = float(np.mean(frames))
                res.hnr_db = float(10.0 * np.log10(r / max(1e-6, 1.0 - r)))
        except Exception:
            logger.debug("optimizer: HNR fallback failed", exc_info=True)

    def _analyze_loudness_quality(self, wav: np.ndarray, res: AnalysisResult) -> None:
        self._analyze_hnr(wav, res)
        rms_db = self._frame_rms_db(wav)
        if rms_db.size == 0:
            return
        res.rms_db = float(np.percentile(rms_db, 95))
        res.peak = float(np.max(np.abs(wav)))
        res.clipping_ratio = float(np.mean(np.abs(wav) > 0.98))
        res.noise_floor_db = float(np.percentile(rms_db, 10))
        speech_db = float(np.percentile(rms_db[rms_db > res.noise_floor_db + 5.0], 80)) \
            if np.any(rms_db > res.noise_floor_db + 5.0) else res.rms_db
        res.snr_proxy_db = speech_db - res.noise_floor_db
        res.dynamic_range_db = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10))
        # shimmer proxy: mean |ΔdB| between consecutive voiced frames
        vad = self.voice_activity(wav)
        idx = np.where(vad)[0]
        if idx.size >= 3:
            d = np.abs(np.diff(rms_db[idx]))
            res.shimmer_db = float(np.mean(d))

    def _analyze_vad(self, wav: np.ndarray, res: AnalysisResult) -> None:
        vad = self.voice_activity(wav)
        if vad.size == 0:
            return
        res.vad_voiced_ratio = float(np.mean(vad))
        segs = self.segment_speech(wav)
        res.vad_segments = [[a, b] for a, b in segs]
        res.vad_pauses = max(0, len(segs) - 1)

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
        validation = np.concatenate(val_parts)[: int(max_validation_s * SR_ANALYSIS)]
        return search, validation
