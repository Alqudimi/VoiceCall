"""Online data augmentation for the training loop (researched quality lever).

Why: upstream RVC trains on the raw sliced chunks with zero augmentation;
GAN vocoders trained on limited (10-60 min) single-speaker data overfit the
exact recording conditions (channel, level, background) instead of learning
the speaker's timbre. The research consensus for data-limited audio models:

  * noise injection / background mixing is the single most effective speech
    augmentation family (community + ASR literature: Watanabe et al. 2017
    "An Investigation of BLSTM ... Data Augmentation"; Ko et al. 2017 audio
    augmentation for ASR);
  * low-rate gain jitter (±3 dB) prevents the discriminator from keying on
    absolute recording level — the same motivation as the offline loudness
    normalisation already applied per-chunk, but applied online so the GAN
    never sees a constant-level distribution;
  * SpecAugment-style time-domain dropout (brief zero-masked spans) is a
    light regulariser for the content path (Park et al., Interspeech 2019).

All transforms are applied on the GPU to the sliced training batch, inside
the autocast region, strictly to the REAL waveform the discriminator scores
and the mel target region — a differentiable-friendly augmentation applied
identically to real & generated audio is NOT applied to the generator's
output (that would let G "cheat" through the augmentation, per DiffAugment
Zhao et al. NeurIPS 2020 — which showed the opposite trick works for image
GANs; for speech vocoders the standard is augmenting the REAL side only,
as in the noise-robust GAN vocoder literature).

Design contract (Colab-safe):
  * disabled by default at the run-config level via ``augment: false``;
    enabled by the Colab notebook / recommended recipe with conservative
    defaults (small noise, ±2 dB gain jitter, rare dropout);
  * pure-torch ops (mul/add/randn/rand) — no external libraries, guarded
    only by availability of a CUDA/CPU RNG (torch itself);
  * per-step RNG on the device — no host sync, no determinism impact on the
    upstream numerics when disabled;
  * every op is applied with probability p; identity otherwise. p values
    are deliberately small so the augmented distribution stays a minority
    share of the batch stream (aggressive audio augmentation is known to
    hurt vocoder fidelity — the recipe uses light augmentation only).
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

DEFAULTS = {
    # additive noise SNR floor: noise is scaled to a random SNR sampled
    # uniformly in [noise_snr_db_min, noise_snr_db_max] above the chunk RMS
    "noise_snr_db_min": 12.0,
    "noise_snr_db_max": 30.0,
    "noise_p": 0.35,
    # gain jitter amplitude in dB (uniform in [-g, +g])
    "gain_db": 2.0,
    "gain_p": 0.5,
    # time dropout: zero-mask spans of up to `dropout_ms` at up to `n_spans`
    # positions, applied with probability dropout_p
    "dropout_ms": 60,
    "dropout_spans": 2,
    "dropout_p": 0.15,
}


def _augment_enabled(hps) -> bool:
    """run-config flag + RVC_AUGMENT env override (0/off/false/no disables)."""
    env = os.environ.get("RVC_AUGMENT", "").strip().lower()
    if env in ("0", "off", "false", "no"):
        return False
    if env in ("1", "on", "true", "yes"):
        return True
    try:
        return bool(int(hps.train.get("augment", 0) or 0))
    except (TypeError, ValueError):
        return False


def _chunk_rms(wave: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return wave.pow(2).mean(dim=dim, keepdim=True).clamp_min(eps).sqrt()


class WaveformAugmenter:
    """GPU-side online augmentation for the real training waveform.

    ``augment(wave)`` returns the transformed (B, 1, T) or (B, T) float
    waveform — dtype/device preserved, no host syncs. The transform is
    stochastic per call; batches are mixed (some items augmented, some not)
    because every op is drawn per-item via the same RNG.
    """

    def __init__(self, hps, device: str) -> None:
        self.sr = int(hps.data["sampling_rate"])
        self.noise_p = float(hps.train.get("aug_noise_p", DEFAULTS["noise_p"]))
        self.snr_lo = float(hps.train.get("aug_noise_snr_db_min",
                                          DEFAULTS["noise_snr_db_min"]))
        self.snr_hi = float(hps.train.get("aug_noise_snr_db_max",
                                          DEFAULTS["noise_snr_db_max"]))
        self.gain_p = float(hps.train.get("aug_gain_p", DEFAULTS["gain_p"]))
        self.gain_db = float(hps.train.get("aug_gain_db", DEFAULTS["gain_db"]))
        self.dropout_p = float(hps.train.get("aug_dropout_p", DEFAULTS["dropout_p"]))
        self.dropout_ms = int(hps.train.get("aug_dropout_ms", DEFAULTS["dropout_ms"]))
        self.dropout_spans = int(hps.train.get("aug_dropout_spans",
                                               DEFAULTS["dropout_spans"]))

    @torch.no_grad()
    def augment(self, wave: torch.Tensor) -> torch.Tensor:
        """Apply noise / gain-jitter / time-dropout to the real waveform."""
        if wave.dim() == 3 and wave.size(1) == 1:
            flat = wave.squeeze(1)
            squeeze_back = True
        else:
            flat = wave
            squeeze_back = False
        out = flat.float()

        # ---- additive noise at a controlled SNR ---------------------------
        if self.noise_p > 0 and torch.rand(()).item() < self.noise_p:
            # draw one SNR for the whole batch (channel-consistent noise)
            snr_db = self.snr_lo + (self.snr_hi - self.snr_lo) * torch.rand(
                (1,), device=out.device).clamp(0, 1).item() if (
                self.snr_hi > self.snr_lo) else self.snr_lo
            noise = torch.randn_like(out)
            sig_rms = _chunk_rms(out)
            noise_rms = _chunk_rms(noise)
            target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
            out = out + noise * (target_noise_rms / noise_rms)

        # ---- per-item gain jitter ----------------------------------------
        if self.gain_p > 0 and self.gain_db > 0:
            n = out.size(0)
            coin = torch.rand((n,), device=out.device)
            gains_db = (torch.rand((n,), device=out.device) * 2.0 - 1.0) * self.gain_db
            apply = (coin < self.gain_p).float()
            out = out * torch.pow(10.0, (gains_db * apply) / 20.0).unsqueeze(-1)

        # ---- time dropout (brief zero-masked spans) -----------------------
        if self.dropout_p > 0 and self.dropout_ms > 0 and torch.rand(
                ()).item() < self.dropout_p:
            span = min(int(self.sr * self.dropout_ms / 1000.0), out.size(-1) - 1)
            if span > 0:
                for _ in range(max(1, self.dropout_spans)):
                    starts = (torch.rand((out.size(0),), device=out.device)
                              * (out.size(-1) - span)).long()
                    mask = torch.ones_like(out)
                    idx = torch.arange(span, device=out.device).view(1, -1)
                    positions = starts.unsqueeze(-1) + idx
                    valid = positions < out.size(-1)
                    flat_idx = (
                        torch.arange(out.size(0), device=out.device).unsqueeze(-1)
                        .expand_as(positions)[valid]
                        * out.size(-1) + positions[valid]
                    )
                    flat_mask = mask.view(-1)
                    flat_mask.index_fill_(0, flat_idx, 0.0)
                    mask = flat_mask.view_as(out)
                    out = out * mask

        result = out.to(wave.dtype)
        return result.unsqueeze(1) if squeeze_back else result

    def diff_augment_pair(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Symmetric Differentiable Augmentation (DiffAugment, Zhao et al. NeurIPS 2020).

        Applies identical differentiable transforms to BOTH ``real`` and ``fake``
        before discriminator scoring so the Nash equilibrium stays at p_g = p_data
        (unlike real-only noise/dropout which pushes G to generate noise/dropouts!).
        """
        if self.gain_p <= 0 or self.gain_db <= 0:
            return real, fake
        n = real.size(0)
        device = real.device
        with torch.no_grad():
            coin = torch.rand((n,), device=device)
            gains_db = (torch.rand((n,), device=device) * 2.0 - 1.0) * self.gain_db
            apply = (coin < self.gain_p).float()
            scale = torch.pow(10.0, (gains_db * apply) / 20.0)
            while scale.dim() < real.dim():
                scale = scale.unsqueeze(-1)
        return real * scale.to(real.dtype), fake * scale.to(fake.dtype)


def build_augmenter(hps, device: str) -> WaveformAugmenter | None:
    """Factory honouring the run-config flag; None means 'no augmentation'."""
    if not _augment_enabled(hps):
        return None
    augmenter = WaveformAugmenter(hps, device)
    logger.info(
        "online waveform augmentation enabled: noise(p=%.2f, snr %g-%g dB), "
        "gain(±%g dB, p=%.2f), dropout(%d ms ×%d, p=%.2f)",
        augmenter.noise_p, augmenter.snr_lo, augmenter.snr_hi,
        augmenter.gain_db, augmenter.gain_p,
        augmenter.dropout_ms, augmenter.dropout_spans, augmenter.dropout_p,
    )
    return augmenter
