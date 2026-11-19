"""Multi-scale (raw-waveform) discriminator — restoring the full HiFi-GAN recipe.

Upstream RVC's ``MultiPeriodDiscriminator(V2)`` already embeds ONE raw
``DiscriminatorS`` (scale 1). The original HiFi-GAN paper (Kong et al.,
NeurIPS 2020) — the vocoder family RVC's generator is derived from — uses a
3-scale MSD (1/2/4 mean-pooled raw discriminators) alongside MPD, and that
multi-scale adversarial + feature-matching signal measurably improves
high-frequency detail and removes long-period artifacts.

This module adds the MISSING coarser scales (2 and 4, block mean-pooled
before the same ``DiscriminatorS`` conv stack, spectral-normed per the
HiFi-GAN recommendation for raw-waveform discriminators):
  * it is a NEW module with its own optimizer/scheduler, checkpointed under
    ``D2_*.pth`` alongside G/D — so G/D checkpoints, pretrained bases and
    upstream resume stay 100% untouched;
  * a resume with no D2 state (legacy run) simply warm-starts D2 fresh, the
    exact situation of a brand-new run (D2 is adversarial scaffolding, not
    carried model quality);
  * disable with ``use_msd: false`` (run-config) or ``RVC_MSD=0``.

The loss wiring needs no new loss functions: the existing
``losses.discriminator_loss`` / ``generator_loss`` / ``feature_loss`` already
consume concatenated output lists (they iterate zip(r_list, g_list)), so MPD +
MSD outputs are simply concatenated before the existing calls.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

from rvc_platform.engine.rvc.models.models import DiscriminatorS

logger = logging.getLogger(__name__)


class MeanPoolScale(nn.Module):
    """Block mean-pool by ``scale`` then score with a DiscriminatorS stack."""

    def __init__(self, scale: int, use_spectral_norm: bool = True) -> None:
        super().__init__()
        if scale < 2:
            raise ValueError("MeanPoolScale is for coarse scales >= 2 (scale 1 lives in MPD)")
        self.scale = int(scale)
        self.disc = DiscriminatorS(use_spectral_norm=use_spectral_norm)

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:  # (B, 1, T) -> pooled along time
            x = F.avg_pool1d(x.float(), kernel_size=self.scale, stride=self.scale)
        else:             # (B, T)
            x = F.avg_pool1d(x.float().unsqueeze(1), kernel_size=self.scale,
                             stride=self.scale).squeeze(1)
        return self.disc(x)


class MultiScaleDiscriminator(nn.Module):
    """Coarse-scale raw-waveform discriminator set (HiFi-GAN MSD's scales 2/4).

    ``forward(y, y_hat)`` returns the same contract as MPD:
        (y_d_rs, y_d_gs, fmap_rs, fmap_gs) — lists over sub-discriminators.
    """

    def __init__(self, scales: tuple[int, ...] = (2, 4),
                 use_spectral_norm: bool = True) -> None:
        super().__init__()
        self.scales = tuple(scales)
        self.discriminators = nn.ModuleList([
            MeanPoolScale(scale, use_spectral_norm=use_spectral_norm)
            for scale in self.scales
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        if y.dim() == 2:
            y = y.unsqueeze(1)
        if y_hat.dim() == 2:
            y_hat = y_hat.unsqueeze(1)
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
