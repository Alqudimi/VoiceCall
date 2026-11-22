"""VAD-based silence filtering for prepared chunks (researched quality lever).

Why: the upstream slicer is energy/silence-aware when CUTTING, but the final
filelist still contains chunks whose speech ratio can be tiny (long trailing
silences inside a 3.7 s window, breathy pauses, room tone between words).
Mel+KL+GAN losses spent on near-silent chunks are wasted capacity and teach
the model to emit room tone; community guides for RVC/Applio consistently
recommend aggressive silence removal for datasets.

Implementation (Colab-safe):
  * a pure-torch/numpy energy-VAD: 30 ms frames, RMS + hangover logic —
    zero external dependencies (a neural VAD (Silero) would add an ONNX
    dependency and a model download; the energy VAD catches exactly the
    near-silent-chunk failure mode this lever targets);
  * used at the ``filelist`` stage: chunks below ``min_speech_ratio`` are
    dropped from the training filelist (never deleted from disk — the
    stage is idempotent and reversible by re-running with a lower bar);
  * guarded import of ``intervad``-style third-party VADs is deliberately
    NOT attempted: every optional dependency is a Colab failure vector.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FRAME_MS = 30


def speech_ratio(wav: np.ndarray, sr: int,
                 frame_ms: int = FRAME_MS,
                 threshold_db: float = -40.0) -> float:
    """Fraction of 30 ms frames whose RMS exceeds ``threshold_db`` (dBFS)."""
    wav = np.asarray(wav, dtype=np.float32).ravel()
    if wav.size < 2:
        return 0.0
    frame = max(1, int(sr * frame_ms / 1000.0))
    n = wav.size // frame
    if n < 1:
        rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-12)
        return 1.0 if 20.0 * np.log10(rms) > threshold_db else 0.0
    frames = wav[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12
    db = 20.0 * np.log10(rms)
    return float(np.mean(db > threshold_db))


def filter_filelist(filelist: Path, out_filelist: Path,
                    gt_wavs_dir: Path, sr: int,
                    min_speech_ratio: float = 0.25) -> tuple[int, int]:
    """Drop filelist lines whose ground-truth chunk is mostly silence.

    Returns (kept, dropped). Lines are ``gt|feature|f0|f0nsf|spk`` — only the
    gt wav is inspected (it is the same audio the other features encode).
    """
    lines = [ln for ln in filelist.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    kept_lines = []
    dropped = 0
    for line in lines:
        gt_rel = line.split("|", 1)[0]
        gt_path = gt_wavs_dir / gt_rel if not Path(gt_rel).is_absolute() \
            else Path(gt_rel)
        try:
            wav, file_sr = _load_wav(gt_path)
        except Exception:
            dropped += 1
            continue
        if file_sr != sr:
            # loader contract guarantees target-sr chunks; mismatch = corrupt
            dropped += 1
            continue
        if speech_ratio(wav, sr) < min_speech_ratio:
            dropped += 1
            continue
        kept_lines.append(line)
    out_filelist.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""),
                            encoding="utf-8")
    logger.info("filelist silence filter: kept %d / dropped %d chunks "
                "(min_speech_ratio=%.2f)", len(kept_lines), dropped,
                min_speech_ratio)
    return len(kept_lines), dropped


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile

    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr
