"""Multi-Scale Mel-Spectrogram reconstruction loss (BigVGAN-v2 / DAC recipe).

Why: single-resolution mel-L1 (2048/480) misses fast consonant transients
(plosives, sibilants) at short time scales, while linear MRSTFT weights all
frequencies uniformly rather than on the human perceptual mel scale where
vocal formants and speaker timbre live. Evaluating mel-L1 across four
complementary STFT resolutions (256/512/1024/2048) forces the generator to
capture both sharp consonant attacks and rich harmonic formant structure.

Numerics: computed in float32 OUTSIDE autocast on the pristine ground-truth
segment vs ``y_hat``. Zero extra model parameters; 100% RVC v2 compatible.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

# (n_fft, hop_length, win_length, n_mels)
DEFAULT_MEL_SCALES: tuple[tuple[int, int, int, int], ...] = (
    (256, 64, 256, 40),
    (512, 128, 512, 80),
    (1024, 256, 1024, 128),
    (2048, 512, 2048, 128),
)

_MEL_BANK_CACHE: dict[tuple[int, int, int, float, float | None, str], torch.Tensor] = {}
_WIN_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _get_window(win_len: int, device: torch.device) -> torch.Tensor:
    key = (win_len, str(device))
    win = _WIN_CACHE.get(key)
    if win is None:
        win = torch.hann_window(win_len, device=device, dtype=torch.float32)
        _WIN_CACHE[key] = win
    return win


def _get_mel_basis(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float | None,
    device: torch.device,
) -> torch.Tensor:
    key = (sr, n_fft, n_mels, float(fmin), float(fmax) if fmax is not None else None, str(device))
    basis = _MEL_BANK_CACHE.get(key)
    if basis is None:
        mel_np = librosa_mel_fn(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
        )
        basis = torch.from_numpy(mel_np).to(device=device, dtype=torch.float32)
        _MEL_BANK_CACHE[key] = basis
    return basis


def _log_mel(
    x: torch.Tensor, sr: int, n_fft: int, hop: int, win: int, n_mels: int,
    fmin: float = 0.0, fmax: float | None = None,
) -> torch.Tensor:
    pad = max(0, (n_fft - hop) // 2)
    if pad > 0 and x.shape[-1] > pad:
        x = F.pad(x.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    window = _get_window(win, x.device)
    stft = torch.stft(
        x, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, center=False, return_complex=True,
    )
    mag = torch.sqrt(stft.real.pow(2) + stft.imag.pow(2) + 1e-7)
    basis = _get_mel_basis(sr, n_fft, n_mels, fmin, fmax, x.device)
    mel = torch.matmul(basis, mag)
    return torch.log(torch.clamp(mel, min=1e-5))


def multi_scale_mel_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sr: int = 48_000,
    scales: tuple[tuple[int, int, int, int], ...] = DEFAULT_MEL_SCALES,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> torch.Tensor:
    """Mean L1 log-mel reconstruction loss across multiple STFT resolutions."""
    if y_true.dim() == 3:
        y_true = y_true.squeeze(1)
    if y_pred.dim() == 3:
        y_pred = y_pred.squeeze(1)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"multi_scale_mel_loss shape mismatch: {tuple(y_true.shape)} vs {tuple(y_pred.shape)}"
        )
    y_true_f = y_true.float()
    y_pred_f = y_pred.float()
    t_len = y_true_f.shape[-1]
    total = torch.zeros((), device=y_true_f.device, dtype=torch.float32)
    used = 0
    for n_fft, hop, win, n_mels in scales:
        if t_len < win or (n_fft // 2 + 1) < n_mels:
            continue
        m_true = _log_mel(y_true_f, sr, n_fft, hop, win, n_mels, fmin=fmin, fmax=fmax)
        m_pred = _log_mel(y_pred_f, sr, n_fft, hop, win, n_mels, fmin=fmin, fmax=fmax)
        total = total + F.l1_loss(m_true, m_pred)
        used += 1
    if used == 0:
        return total
    return torch.clamp(total / used, max=25.0)
