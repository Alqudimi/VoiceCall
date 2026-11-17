"""DSP primitives shared by inference, training and real-time paths.

Everything here is a small, deterministic, unit-testable transform:
resampling, normalisation, high/band-pass filtering, spectral-gate
denoising, energy-based silence detection, chunking with crossfades.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or audio.size == 0:
        return audio.astype(np.float32, copy=True)
    # polyphase is exact enough and much faster than FFT-based for speech SRs
    gcd = np.gcd(int(src_sr), int(dst_sr))
    up, down = int(dst_sr) // gcd, int(src_sr) // gcd
    out = signal.resample_poly(audio, up, down, axis=0).astype(np.float32)
    return out


def normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Peak normalisation; leaves silence untouched (no divide-by-zero)."""
    current = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current <= 1e-8:
        return audio.astype(np.float32, copy=True)
    return (audio / current * peak).astype(np.float32)


def highpass(audio: np.ndarray, sr: int, wn: float = 48.0, order: int = 5) -> np.ndarray:
    """Remove DC/rumble; causal lfilter (matches upstream preprocessing)."""
    b, a = signal.butter(order, wn, btype="high", fs=sr)
    return signal.lfilter(b, a, audio).astype(np.float32)


def denoise_spectral_gate(
    audio: np.ndarray, sr: int, strength: float = 1.0, frame_ms: float = 20.0
) -> np.ndarray:
    """Simple magnitude-domain spectral gate (no extra dependencies).

    Estimates the noise floor from low-energy frames, then attenuates
    frame bins below floor + margin. ``strength`` in [0, 2]: >1 aggressive.
    """
    if audio.size == 0:
        return audio
    frame = int(sr * frame_ms / 1000)
    if frame < 64:
        frame = 256
    # pad to frame multiple
    pad = (-len(audio)) % frame
    padded = np.pad(audio, (0, pad))
    frames = padded.reshape(-1, frame)
    window = np.hanning(frame)
    spec = np.fft.rfft(frames * window, axis=1)
    mag = np.abs(spec)
    # noise floor: 10th-percentile magnitude per bin across time
    floor = np.percentile(mag, 10, axis=0, keepdims=True)
    threshold = floor * (1.0 + strength)
    mask = (mag > threshold).astype(np.float32)
    kept = spec * mask
    frames_out = np.fft.irfft(kept, n=frame, axis=1)
    out = frames_out.reshape(-1)[: len(audio)]
    return out.astype(np.float32)


def rms_envelope(audio: np.ndarray, sr: int, frame_ms: float = 20.0) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = int(np.ceil(len(audio) / frame))
    padded = np.pad(audio, (0, n_frames * frame - len(audio)))
    frames = padded.reshape(n_frames, frame)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def detect_silence(
    audio: np.ndarray, sr: int, threshold_db: float = -40.0,
    min_silence_ms: int = 200, frame_ms: float = 20.0,
) -> list[tuple[int, int]]:
    """Return [start,end) sample ranges classified as silence."""
    env = rms_envelope(audio, sr, frame_ms)
    frame = max(1, int(sr * frame_ms / 1000))
    threshold = 10 ** (threshold_db / 20)
    silent = env < threshold
    spans: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(silent):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start * frame, i * frame))
            start = None
    if start is not None:
        spans.append((start * frame, len(audio)))
    min_samples = int(sr * min_silence_ms / 1000)
    return [(s, e) for s, e in spans if e - s >= min_samples]


def trim_silence(audio: np.ndarray, sr: int, margin_ms: int = 120,
                 threshold_db: float = -45.0) -> np.ndarray:
    """Trim leading/trailing silence, keeping a margin for natural onset."""
    if audio.size == 0:
        return audio
    env = rms_envelope(audio, sr)
    frame = max(1, int(sr * 20 / 1000))
    threshold = 10 ** (threshold_db / 20)
    voiced = np.where(env >= threshold)[0]
    if voiced.size == 0:
        return audio.astype(np.float32)
    first = int(voiced[0]) * frame
    last = min(len(audio), (int(voiced[-1]) + 1) * frame)
    margin = int(sr * margin_ms / 1000)
    return audio[max(0, first - margin):min(len(audio), last + margin)].astype(np.float32)


def vad_mask(audio: np.ndarray, sr: int, threshold_db: float = -40.0,
             frame_ms: float = 20.0) -> np.ndarray:
    """Boolean per-sample mask (True = voiced). Used to skip pure-silence
    blocks in real-time streaming without touching the speech itself."""
    env = rms_envelope(audio, sr, frame_ms)
    frame = max(1, int(sr * frame_ms / 1000))
    mask_frames = env >= 10 ** (threshold_db / 20)
    mask = np.repeat(mask_frames, frame)[: len(audio)]
    if len(mask) < len(audio):
        mask = np.pad(mask, (0, len(audio) - len(mask)), constant_values=True)
    return mask


def crossfade_concat(a: np.ndarray, b: np.ndarray, fade: int) -> np.ndarray:
    """Overlap-add ``b`` onto the tail of ``a`` with a linear crossfade."""
    if fade <= 0 or a.size == 0 or b.size == 0:
        return np.concatenate([a, b]).astype(np.float32)
    fade = min(fade, len(a), len(b))
    tail = a[-fade:]
    head = b[:fade]
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    blended = tail * (1 - ramp) + head * ramp
    return np.concatenate([a[:-fade], blended, b[fade:]]).astype(np.float32)
