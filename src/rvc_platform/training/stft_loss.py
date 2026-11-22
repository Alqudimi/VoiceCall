"""Multi-resolution STFT (MRSTFT) loss for the generator.

Research-backed auxiliary loss for GAN vocoders — the single-scale mel L1 used
by the upstream RVC trainer is complemented with a multi-resolution mixture of
  * spectral convergence  || |Y| - |Y_hat| ||_F / || |Y| ||_F   (Yamamoto et
    al., Parallel WaveGAN, ICASSP 2020), and
  * log-magnitude L1      mean | log|Y| - log|Y_hat| |            (same paper)
across three FFT sizes (512/1024/2048 with aligned hops), the recipe used by
UnivNet (Kwon et al., Interspeech 2021: generator trained with MRSTFT +
multi-resolution discriminators) and BigVGAN (Lee et al., ICLR 2023: mel +
multi-scale STFT losses). Both report clear perceptual gains over
single-resolution objectives.

Numerics: computed in float32 OUTSIDE autocast (the same policy the upstream
loop already applies to mel/KL/adv losses), on the sliced training segment
(y_hat vs wave_sliced) — exactly the region the mel loss covers.

Weighting: ``c_stft`` in ``train`` (run-config field ``c_stft``; 0 disables and
reproduces the upstream objective bit-for-bit). The default was calibrated on
synthetic speech-band signals so the MRSTFT term contributes a few percent of
the total generator loss — a helpful regulariser, never the dominant term.
"""
from __future__ import annotations

import torch

# (n_fft, hop, win) — the standard UnivNet/BigVGAN resolution ladder
DEFAULT_RESOLUTIONS: tuple[tuple[int, int, int], ...] = (
    (512, 128, 512),
    (1024, 256, 1024),
    (2048, 512, 2048),
)
_EPS = 1e-8


def _mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    window = torch.hann_window(win, device=x.device, dtype=torch.float32)
    spec = torch.stft(
        x.float(), n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, center=True, return_complex=True,
    )
    return spec.abs().clamp_min(_EPS)


def spectral_convergence(mag_ref: torch.Tensor, mag_pred: torch.Tensor) -> torch.Tensor:
    diff_norm = torch.norm(mag_ref - mag_pred, p="fro")
    ref_norm = torch.norm(mag_ref, p="fro")
    return diff_norm / torch.clamp_min(ref_norm, 1e-3)


def log_magnitude_l1(mag_ref: torch.Tensor, mag_pred: torch.Tensor) -> torch.Tensor:
    log_ref = torch.log(torch.clamp_min(mag_ref, 1e-5))
    log_pred = torch.log(torch.clamp_min(mag_pred, 1e-5))
    return torch.mean(torch.abs(log_ref - log_pred))


def mr_stft_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    resolutions: tuple[tuple[int, int, int], ...] = DEFAULT_RESOLUTIONS,
) -> torch.Tensor:
    """Mean over resolutions of (spectral convergence + log-magnitude L1).

    ``y_true`` / ``y_pred``: (B, T) or (B, 1, T) raw waveforms. Returns a
    scalar float32 tensor; both inputs are cast to float32 internally.
    """
    if y_true.dim() == 3:
        y_true = y_true.squeeze(1)
    if y_pred.dim() == 3:
        y_pred = y_pred.squeeze(1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"mr_stft_loss shape mismatch: {tuple(y_true.shape)} vs {tuple(y_pred.shape)}")
    total = torch.zeros((), device=y_true.device, dtype=torch.float32)
    for n_fft, hop, win in resolutions:
        mag_ref = _mag(y_true, n_fft, hop, win)
        mag_pred = _mag(y_pred, n_fft, hop, win)
        total = total + spectral_convergence(mag_ref, mag_pred) + log_magnitude_l1(mag_ref, mag_pred)
    res_loss = total / len(resolutions)
    return torch.clamp(res_loss, max=25.0)
