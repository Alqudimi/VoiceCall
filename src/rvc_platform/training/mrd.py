"""Multi-Resolution Spectrogram Discriminator (MRD) — UnivNet / BigVGAN-v2.

Why: time-domain discriminators (MPD + MSD) inspect raw waveforms and periods,
leaving 2D time-frequency artifacts (metallic high-frequency ringing, phase
smearing, harmonic ghosting) weakly penalized. UnivNet (Kwon et al.,
Interspeech 2021) and BigVGAN (Lee et al., ICLR 2023) pair MPD with an MRD
that discriminates 2D STFT magnitude spectrograms across multiple window/hop
resolutions, using spectral-normalized 2D convolutions and feature matching.

Like MSD (D2), MRD is a discriminator-only training scaffold checkpointed in
its own ``D3_*.pth`` — exported inference ``.pth`` models remain 100%
byte-compatible with RVC v2 and all existing tooling.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

logger = logging.getLogger(__name__)

# (n_fft, hop_length, win_length) — UnivNet / BigVGAN-style 3-resolution ladder
DEFAULT_MRD_RESOLUTIONS: tuple[tuple[int, int, int], ...] = (
    (1024, 120, 600),
    (2048, 240, 1200),
    (512, 50, 240),
)


class DiscriminatorR(nn.Module):
    """Single-resolution 2D STFT magnitude spectrogram discriminator."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int,
                 use_spectral_norm: bool = True) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        norm_f = spectral_norm if use_spectral_norm else (lambda m: m)
        chs = 32
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, chs, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(chs, chs, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(chs, chs, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(chs, chs, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(chs, chs, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(chs, 1, (3, 3), padding=(1, 1)))
        self.register_buffer(
            "window", torch.hann_window(self.win_length, dtype=torch.float32),
            persistent=False,
        )

    def _spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x.float()
        # Guard against short test segments (< win_length)
        if x.shape[-1] < self.n_fft:
            x = F.pad(x, (0, self.n_fft - x.shape[-1]), mode="reflect" if x.shape[-1] > 1 else "constant")
        stft = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=x.device, dtype=torch.float32),
            center=True, return_complex=True,
        )
        # Log-compressed magnitude gives balanced sensitivity across low & high freqs
        mag = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-7)
        mag = torch.log1p(mag)
        # (B, F, T) -> (B, 1, T, F) matching UnivNet/BigVGAN convention
        return mag.transpose(1, 2).unsqueeze(1)

    def forward(self, x: torch.Tensor):
        fmap: list[torch.Tensor] = []
        z = self._spectrogram(x)
        for conv in self.convs:
            z = conv(z)
            z = F.leaky_relu(z, 0.1)
            fmap.append(z)
        z = self.conv_post(z)
        fmap.append(z)
        z = torch.flatten(z, 1, -1)
        return z, fmap


class MultiResolutionDiscriminator(nn.Module):
    """3-resolution STFT spectrogram discriminator matching MPD/MSD contract."""

    def __init__(
        self,
        resolutions: tuple[tuple[int, int, int], ...] = DEFAULT_MRD_RESOLUTIONS,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.resolutions = tuple(resolutions)
        self.discriminators = nn.ModuleList([
            DiscriminatorR(n_fft, hop, win, use_spectral_norm=use_spectral_norm)
            for n_fft, hop, win in self.resolutions
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
