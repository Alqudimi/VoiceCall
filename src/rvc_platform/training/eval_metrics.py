"""Objective validation metrics for best-model selection and evaluation.

Why: upstream best-model selection watches only the held-out mel L1
(val_mel) — a *reconstruction* proxy that is blind to the two failure modes
voice-conversion users actually care about: losing the target timbre
(speaker identity) and pitch distortion. The research consensus for VC/SVC
model selection is a *composite* objective (see UPGRADE_REPORT_AR.md):

  1. ``val_mel``    — held-out mel L1 (kept; perceptual reconstruction proxy).
  2. ``spk_sim``    — speaker-embedding cosine similarity between the
                      reconstructed audio and the ground truth, via an ECAPA-
                      TDNN speaker model (Desplanques et al., Interspeech
                      2020; SpeechBrain implementation). This is the same
                      "speaker similarity" metric used to report SOTA VC
                      papers (e.g. YourTTS, RVC forks' overtraining detectors
                      such as Applio's) and directly measures timbre
                      preservation.
  3. ``val_f0_rmse``— RMSE of F0 contours (RMVPE) of reconstruction vs
                      ground truth, reported as a diagnostic (pitch
                      preservation).

Selection metric:
    composite = val_mel + spk_weight * (1 - spk_sim)        (lower is better)
Speaker similarity is computed at 16 kHz (ECAPA requirement) after resampling
the sliced validation segment (both reconstructed and ground-truth wave).

Colab-safety: every heavy dependency is OPTIONAL and guarded:
  * ``speechbrain`` missing / fails to download / ``RVC_SPK_SIM=0``  -> the
    metric is silently skipped and selection falls back to val_mel exactly
    as before;
  * RMVPE asset missing or ``RVC_F0_EVAL=0`` -> f0 RMSE skipped.
No metric can ever crash a training run: all failures log-and-continue.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SPEAKER_MODEL_HF = "speechbrain/spkrec-ecapa-voxcelebx"


class SpeakerEmbedder:
    """Thin wrapper over SpeechBrain's ECAPA-TDNN speaker encoder."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @torch.no_grad()
    def embed(self, wav16k: torch.Tensor) -> torch.Tensor:
        """wav16k: (B, T) float tensor at 16 kHz -> L2-normalised (B, D)."""
        if wav16k.dim() == 3:
            wav16k = wav16k.squeeze(1)
        feats = self.model.encode_batch(wavs=wav16k.float().unsqueeze(1))
        emb = feats.squeeze(1) if feats.dim() == 3 else feats
        emb = torch.nn.functional.normalize(emb.float(), p=2, dim=-1)
        return emb


def try_build_speaker_embedder(device: str) -> SpeakerEmbedder | None:
    """Best-effort ECAPA builder — never raises, returns None when unavailable."""
    if os.environ.get("RVC_SPK_SIM", "1").strip().lower() in ("0", "off", "false", "no"):
        return None
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:
        try:  # speechbrain < 1.0 layout
            from speechbrain.pretrained.interfaces import EncoderClassifier  # type: ignore
        except Exception:
            logger.info("speechbrain not installed — speaker-similarity metric "
                        "disabled (install with: uv pip install speechbrain)")
            return None
    try:
        model = EncoderClassifier.from_hparams(
            source=_SPEAKER_MODEL_HF,
            savedir=None,
            run_opts={"device": device},
        )
        return SpeakerEmbedder(model)
    except Exception:
        logger.info("ECAPA speaker model unavailable (download failure?) — "
                    "speaker-similarity metric disabled", exc_info=True)
        return None


def resample_16k(wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
    """(B, T) or (B, 1, T) at orig_sr -> (B, T) float32 at 16 kHz (GPU-friendly)."""
    squeezed = wav.squeeze(1) if wav.dim() == 3 else wav
    if orig_sr == 16_000:
        return squeezed.float()
    try:
        import torchaudio.functional as AF

        out = AF.resample(squeezed.float(), orig_sr, 16_000)
        return out
    except Exception:  # torchaudio functional unavailable — CPU/librosa path
        arr = squeezed.detach().cpu().numpy()
        import librosa

        out = np.stack([librosa.resample(ch.astype(np.float32), orig_sr=orig_sr,
                                         target_sr=16_000) for ch in arr])
        return torch.from_numpy(out).to(squeezed.device, squeezed.dtype).float()


def speaker_similarity(embedder: SpeakerEmbedder, wav_pred: torch.Tensor,
                       wav_true: torch.Tensor, orig_sr: int) -> float:
    """Mean cosine similarity between paired embeddings (1.0 = identical)."""
    pred16 = resample_16k(wav_pred, orig_sr)
    true16 = resample_16k(wav_true, orig_sr)
    if pred16.shape != true16.shape:
        n = min(pred16.shape[-1], true16.shape[-1])
        pred16 = pred16[..., :n]
        true16 = true16[..., :n]
    with torch.no_grad():
        e_pred = embedder.embed(pred16)
        e_true = embedder.embed(true16)
        sim = (e_pred * e_true).sum(dim=-1).clamp(-1.0, 1.0)
    return float(sim.mean().item())


def try_build_f0_helper(device: str, is_half: bool) -> Callable[[np.ndarray], np.ndarray] | None:
    """Best-effort RMVPE f0 extractor for evaluation (reuses shipped weights)."""
    if os.environ.get("RVC_F0_EVAL", "1").strip().lower() in ("0", "off", "false", "no"):
        return None
    try:
        from rvc_platform.engine.rvc import f0 as f0_mod

        extractor = f0_mod.get_f0_extractor("rmvpe", device, is_half)
        return extractor.extract
    except Exception:
        logger.info("RMVPE unavailable for evaluation — f0 RMSE metric disabled")
        return None


def f0_rmse(f0_fn: Callable[[np.ndarray], np.ndarray], wav_pred_16k: np.ndarray,
            f0_true_frames: np.ndarray) -> float | None:
    """Frame-aligned F0 RMSE (Hz) between prediction and ground truth.

    ``f0_true_frames``: (T,) ground-truth f0 at the 10 ms frame grid; the
    prediction is extracted from the 16 kHz reconstruction and aligned to the
    ground-truth length (tail-pad/truncate). Unvoiced GT frames are excluded
    (voiced RMSE), the standard RPE metric in pitch literature.
    """
    try:
        f0_true = np.asarray(f0_true_frames, dtype=np.float32).ravel()
        pred = np.asarray(f0_fn(np.asarray(wav_pred_16k, dtype=np.float32)),
                          dtype=np.float32).ravel()
        n = min(len(pred), len(f0_true))
        if n < 2:
            return None
        pred = pred[:n]
        gt = f0_true[:n]
        voiced = gt > 0.0
        if voiced.sum() < 2:
            return None
        return float(np.sqrt(np.mean((pred[voiced] - gt[voiced]) ** 2)))
    except Exception:
        return None


def composite_score(val_mel: float, spk_sim: float | None,
                    spk_weight: float = 1.0) -> float:
    """Selection metric (lower is better): val_mel + w * (1 - spk_sim)."""
    penalty = (1.0 - float(spk_sim)) if spk_sim is not None else 0.0
    return float(val_mel) + float(spk_weight) * penalty
