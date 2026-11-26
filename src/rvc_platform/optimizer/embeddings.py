"""Speaker-embedding backends for identity-aware optimization.

Research-grounded choice of speaker verification models (see profile docs):

* **WavLM x-vector** (``microsoft/wavlm-base-plus-sv``, Chen et al. 2022) —
  a self-supervised speaker-verification encoder; reported to correlate with
  *perceptual* voice similarity better than classic ECAPA on VC-style data
  (Ahn et al., "VoxSim", 2024). Uses ``transformers`` which is already a
  CORE dependency of this platform — no new install.
* **ECAPA-TDNN** (Desplanques et al., Interspeech 2020; SpeechBrain
  ``spkrec-ecapa-voxceleb``) — the metric used to report speaker similarity
  in VC papers (YourTTS, RVC overtraining detectors). Reuses this project's
  existing integration in ``training/eval_metrics.py``. OPTIONAL dependency.
* **DSP fallback** — MFCC-statistics voiceprint: mean+std of 20 MFCCs over
  VAD-speech frames, L2-normalised. Always available, flagged
  ``reliability="low"``; used only when both neural models are unavailable,
  and the UI shows an explicit warning (never a silent downgrade). Phase 02
  made this claim TRUE — the fallback previously documented but never
  implemented is now real code with real tests.

The manager never trusts a single model blindly: when two neural embedders
are available their cosine similarities are reported *separately* and their
agreement is measured — large disagreement degrades the reliability verdict
for that comparison instead of hiding it inside an average.

Phase 02 adds per-backend score **calibration** (audited W1): raw cosines
from different SV models live on different scales, so
``similarity_report`` accepts a :class:`~rvc_platform.optimizer.calibration.
SpeakerScoreCalibrator` and reports calibrated values + a calibrated
ensemble alongside the raw ones (raw is never hidden).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from rvc_platform.optimizer.calibration import SpeakerScoreCalibrator

logger = logging.getLogger(__name__)

_SR = 16_000
_WAVLM_MODEL_ID = "microsoft/wavlm-base-plus-sv"


@dataclass
class SpeakerEmbeddingStatus:
    """Which embedders are actually usable in this process (honesty first)."""

    wavlm: bool = False
    ecapa: bool = False
    dsp: bool = True                     # always available (low reliability)
    messages: dict[str, str] = field(default_factory=dict)

    @property
    def neural_count(self) -> int:
        return int(self.wavlm) + int(self.ecapa)

    @property
    def reliability(self) -> str:
        if self.neural_count >= 2:
            return "high"                # two independent SV models agree
        if self.neural_count == 1:
            return "medium"              # single SV model
        return "low"                     # DSP voiceprint only


class _WavlmEmbedder:
    """WavLM base+ speaker-verification x-vector (transformers, 512-dim)."""

    name = "wavlm"

    def __init__(self, device: str) -> None:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self._torch = torch
        self.model = WavLMForXVector.from_pretrained(_WAVLM_MODEL_ID)
        self.model.eval()
        self.device = device
        try:
            if device.startswith("cuda"):
                self.model = self.model.to(device)
            for p in self.model.parameters():
                p.requires_grad_(False)
        except Exception:
            self.device = "cpu"
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(_WAVLM_MODEL_ID)

    def embed(self, wav16k: np.ndarray) -> np.ndarray:
        import torch

        x = np.asarray(wav16k, dtype=np.float32).ravel()
        inputs = self.fe(x, sampling_rate=_SR, return_tensors="pt")
        with torch.no_grad():
            out = self.model(
                **{k: v.to(self.model.device) for k, v in inputs.items()})
        emb = out.embeddings[0].detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb


class _EcapaEmbedder:
    """ECAPA-TDNN wrapper over this project's SpeechBrain integration."""

    name = "ecapa"

    def __init__(self, device: str) -> None:
        from rvc_platform.training.eval_metrics import try_build_speaker_embedder

        embedder = try_build_speaker_embedder(device)
        if embedder is None:
            raise RuntimeError(
                "ECAPA speaker model unavailable (speechbrain missing?)")
        self._embedder = embedder

    def embed(self, wav16k: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(
            np.asarray(wav16k, dtype=np.float32)).unsqueeze(0)
        emb = self._embedder.embed(x)[0].detach().cpu().numpy().astype(
            np.float32)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb


class _DspEmbedder:
    """Real MFCC-statistics voiceprint (the documented DSP fallback).

    40-dim vector: mean and std of 20 MFCCs computed over VAD-speech
    frames only, energy-compressed and L2-normalised. Not a speaker-
    verification model — a cheap always-available voiceprint whose
    similarity is a *timbre proxy*, explicitly reported with LOW
    reliability.
    """

    name = "dsp"

    def __init__(self, device: str) -> None:
        self.device = "cpu"              # pure numpy/librosa — no GPU path

    def embed(self, wav16k: np.ndarray) -> np.ndarray:
        import librosa

        x = np.asarray(wav16k, dtype=np.float32).ravel()
        if x.size < 2048:
            x = np.pad(x, (0, 2048 - x.size))
        mfcc = librosa.feature.mfcc(y=x, sr=_SR, n_mfcc=20, n_fft=1024,
                                     hop_length=256)
        # frame energy weighting: quiet frames (silence edges) weight less
        rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0]
        w = np.clip(rms, 1e-4, None)
        if w.size == mfcc.shape[1]:
            w = w / w.sum()
            mean = (mfcc * w).sum(axis=1)
            var = (w * (mfcc - mean[:, None]) ** 2).sum(axis=1)
        else:
            mean = mfcc.mean(axis=1)
            var = mfcc.var(axis=1)
        vec = np.concatenate([mean, np.sqrt(np.maximum(var, 0.0))])
        # compress the dynamic range so one loud frame cannot dominate
        vec = np.tanh(vec / 30.0)
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm > 0 else vec


class SpeakerEmbedderManager:
    """Lazily constructed, process-wide speaker-embedding registry.

    Mirrors the engine's ``CachedF0`` pattern: models stay resident and are
    shared by every optimization stage (profile build + per-trial scoring),
    so a search never reloads the networks between trials.
    """

    def __init__(self, device: str, prefer: str = "auto") -> None:
        self.device = device
        self.prefer = (prefer or "auto").lower()
        self._lock = threading.Lock()
        self._embedders: dict[str, Any] = {}
        self._status = SpeakerEmbeddingStatus()
        self._built = False

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        if self._built:
            return
        with self._lock:
            if self._built:
                return
            want_wavlm = self.prefer in ("auto", "wavlm", "both", "all")
            want_ecapa = self.prefer in ("auto", "ecapa", "both", "all")
            if want_wavlm:
                try:
                    self._embedders["wavlm"] = _WavlmEmbedder(self.device)
                    self._status.wavlm = True
                    logger.info(
                        "optimizer: WavLM x-vector speaker embedder ready")
                except Exception as exc:
                    self._status.messages["wavlm"] = str(exc)[:300]
                    logger.info("optimizer: WavLM embedder unavailable (%s)",
                                exc)
            if want_ecapa:
                try:
                    self._embedders["ecapa"] = _EcapaEmbedder(self.device)
                    self._status.ecapa = True
                    logger.info(
                        "optimizer: ECAPA-TDNN speaker embedder ready")
                except Exception as exc:
                    self._status.messages["ecapa"] = str(exc)[:300]
                    logger.info("optimizer: ECAPA embedder unavailable (%s)",
                                exc)
            if not self._status.neural_count:
                # the DSP voiceprint is now REAL code (Phase 02): always
                # registered so identity scoring degrades instead of dying
                try:
                    self._embedders["dsp"] = _DspEmbedder(self.device)
                    logger.warning(
                        "optimizer: no neural speaker embedder available — "
                        "DSP voiceprint fallback ACTIVE (reliability LOW)")
                except Exception as exc:                      # pragma: no cover
                    logger.warning(
                        "optimizer: DSP voiceprint unavailable (%s)", exc)
            self._built = True

    @property
    def status(self) -> SpeakerEmbeddingStatus:
        self._build()
        return self._status

    # ------------------------------------------------------------------ embed
    def embeddings(self, wav16k: np.ndarray) -> dict[str, np.ndarray]:
        """Return {backend_name: L2-normalised embedding} for one utterance."""
        self._build()
        out: dict[str, np.ndarray] = {}
        for name, emb in self._embedders.items():
            try:
                out[name] = emb.embed(wav16k)
            except Exception:
                logger.exception("optimizer: embed failed for %s", name)
        return out

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0:
            return 0.0
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))

    # ------------------------------------------------------------- similarity
    def similarity_report(
        self,
        wav16k: np.ndarray,
        reference_embeddings: dict[str, list[np.ndarray]],
        calibrator: "SpeakerScoreCalibrator | None" = None,
    ) -> dict[str, Any]:
        """Compare one utterance against a reference embedding set.

        Returns per-backend similarity (mean cosine to each reference
        embedding), an ensemble view and an honest reliability verdict:

        * ``similarities``  — one RAW score per available backend (never
          averaged behind the scenes);
        * ``ensemble``      — mean over neural backends only (undefined when
          none is available) — RAW scales, kept for backward compatibility;
        * ``calibration``   — Phase 02: when a calibrator with per-backend
          same-speaker cohorts is supplied, per-backend margin/calibrated
          values and a **calibrated_ensemble** (comparable scale). The
          calibrated ensemble is what scoring consumes when present;
        * ``agreement``     — spread between neural backends (0 when single);
        * ``reliability``   — high / medium / low, per SpeakerEmbeddingStatus.

        Backend mismatch guard: a reference built with a *different embedder
        set* (e.g. profile built with WavLM, current process only has ECAPA —
        or the reverse) never crashes the search — the incompatible backend
        is skipped and the mismatch is reported in ``notes``.
        """
        self._build()
        sims: dict[str, float] = {}
        mismatched: list[str] = []
        for name, cand in self.embeddings(wav16k).items():
            refs = reference_embeddings.get(name) or []
            if not refs:
                continue
            dims_ok = all(
                np.asarray(r).shape == np.shape(cand) for r in refs)
            if not dims_ok:
                mismatched.append(name)
                continue
            sims[name] = float(
                np.mean([self.cosine(cand, r) for r in refs]))
        neural = [v for k, v in sims.items() if k in ("wavlm", "ecapa")]
        spread = (max(neural) - min(neural)) if len(neural) >= 2 else 0.0
        reliability = self._status.reliability
        notes: list[str] = []
        if mismatched:
            notes.append(
                f"reference profile built with a different embedder set — "
                f"skipped incompatible backends: {', '.join(mismatched)}; "
                "rebuild the reference profile with the current embedders "
                "for full-strength identity scoring")
            if reliability in ("high", "medium"):
                reliability = "medium-low (embedder mismatch)"
        elif reliability == "high" and spread > 0.15:
            # models disagree -> the ensemble mean is NOT trustworthy
            reliability = "medium-low (models disagree)"
        out: dict[str, Any] = {
            "similarities": sims,
            "ensemble": float(np.mean(neural)) if neural else None,
            "agreement_spread": round(float(spread), 4),
            "reliability": reliability,
            "notes": notes,
        }
        if calibrator is not None:
            try:
                out["calibration"] = calibrator.calibrate_report(sims)
            except Exception:
                logger.exception("optimizer: calibration failed")
        return out
