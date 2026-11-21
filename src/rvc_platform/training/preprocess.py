"""Dataset quality check + preprocessing (upstream pipeline, cleaned).

Stages (each idempotent, each reported):
1. ``check``     — inventory, decode probe, corrupt-file detection, duration
                   stats, duplicate detection (content hash), silence ratio;
2. ``preprocess``— silence-aware slicing (vendored Slicer), 3.7 s chunks with
                   0.3 s overlap, high-pass 48 Hz, peak-normalisation
                   (alpha 0.75 / max 0.9 — upstream constants), dual output:
                   target-sr ``0_gt_wavs`` + 16 kHz ``1_16k_wavs``;

Researched dataset-quality levers added on top (see UPGRADE_REPORT_AR.md,
all optional — the upstream numerics are reproduced exactly when both are
off/unused):
* ``denoise``   — best-effort spectral-gating denoise (``noisereduce``) on
                  the raw file BEFORE slicing: removes broadband hiss / hum
                  that otherwise becomes part of the training targets. Guarded:
                  missing library or a failed call is a logged no-op, never a
                  crash (Colab-safe).
* ``loudness``  — per-chunk RMS loudness normalisation to a common target
                  (−20 dBFS, gain-capped, peak-guarded) BEFORE the upstream
                  peak-normalisation: mixed-source datasets land at a common
                  loudness so the GAN is not asked to track level drift.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

from rvc_platform.core.errors import DatasetError
from rvc_platform.infrastructure.audio.io import load_audio
from rvc_platform.infrastructure.audio.slicer import Slicer

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".webm", ".aiff"}
HASH_CHUNK = 1 << 20


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DatasetReport:
    dataset_path: Path
    total_files: int = 0
    usable_files: int = 0
    corrupt: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    total_duration_s: float = 0.0
    min_duration_s: float = 0.0
    max_duration_s: float = 0.0
    empty_audio: list[str] = field(default_factory=list)
    mean_silence_ratio: float | None = None
    capped_at: int | None = None      # set when max_files truncated the scan
    mean_snr_db: float | None = None  # measured dataset cleanliness (FAQ Q9 lever)
    noise_floor_db: float | None = None

    @property
    def ok(self) -> bool:
        return self.usable_files > 0

    @property
    def meets_recommendation(self) -> bool:
        return self.total_duration_s >= 600.0

    def to_dict(self) -> dict:
        return {
            "dataset_path": str(self.dataset_path),
            "total_files": self.total_files,
            "usable_files": self.usable_files,
            "corrupt": self.corrupt,
            "unsupported": self.unsupported,
            "duplicates": [list(d) for d in self.duplicates],
            "duplicates_count": len(self.duplicates),
            "total_duration_s": round(self.total_duration_s, 1),
            "min_duration_s": round(self.min_duration_s, 2),
            "max_duration_s": round(self.max_duration_s, 2),
            "empty_audio": self.empty_audio,
            "mean_silence_ratio": (
                round(self.mean_silence_ratio, 3)
                if self.mean_silence_ratio is not None else None
            ),
            "capped_at": self.capped_at,
            "mean_snr_db": (round(self.mean_snr_db, 1)
                            if self.mean_snr_db is not None else None),
            "noise_floor_db": (round(self.noise_floor_db, 1)
                               if self.noise_floor_db is not None else None),
            "epoch_recommendation": recommend_epochs(
                self.mean_snr_db, self.total_duration_s),
            "recommended_min_duration_s": 600.0,
            "meets_recommendation": self.total_duration_s >= 600.0,
        }


class DatasetChecker:
    def __init__(self, audio_loader=load_audio) -> None:
        self._loader = audio_loader

    def _probe_files(self, audio_files: list[Path]) -> list[tuple]:
        """Per-file (path, audio|None, md5, silence-ratio) probes.

        Threaded only for the real ffmpeg-backed loader (decoding releases the
        GIL); injected loaders (tests) and small scans stay sequential.
        """

        def probe(path: Path):
            try:
                audio = self._loader(path, 16_000)
            except Exception:
                return (path, None, None, None)
            return (path, audio, file_md5(path), _silence_ratio(audio))

        if self._loader is not load_audio or len(audio_files) <= 16:
            return [probe(p) for p in audio_files]
        from concurrent.futures import ThreadPoolExecutor

        workers = min(8, max(2, (os.cpu_count() or 2) * 2))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(probe, audio_files))

    def check(self, dataset_path: Path, max_files: int = 5000,
              recursive: bool = True) -> DatasetReport:
        dataset_path = Path(dataset_path)
        if not dataset_path.is_dir():
            raise DatasetError(f"dataset folder missing: {dataset_path}")
        report = DatasetReport(dataset_path=dataset_path)
        it = dataset_path.rglob("*") if recursive else dataset_path.iterdir()
        files = sorted(p for p in it if p.is_file())
        report.total_files = len(files)
        if len(files) > max_files:
            report.capped_at = max_files
            logger.warning("dataset scan capped at %s files (%s found) — "
                           "raise max_files for a full scan", max_files, len(files))
            files = files[:max_files]
        seen_hashes: dict[str, str] = {}
        durations: list[float] = []
        silence_ratios: list[float] = []
        snr_values: list[float] = []
        noise_floors: list[float] = []
        # decode + md5 + silence ratio per file (thread pool when probing the
        # real loader — ffmpeg decoding releases the GIL); results are
        # consumed in file order so the report matches the sequential scan.
        probes = self._probe_files([
            p for p in files if p.suffix.lower() in AUDIO_EXTS
        ])
        for path in files:
            if path.suffix.lower() not in AUDIO_EXTS:
                report.unsupported.append(path.name)
        for path, audio, digest, silence in probes:
            if audio is None:
                report.corrupt.append(path.name)
                continue
            if audio.size < 1600:  # < 0.1 s of samples
                report.empty_audio.append(path.name)
                continue
            if digest in seen_hashes:
                report.duplicates.append((seen_hashes[digest], path.name))
            else:
                seen_hashes[digest] = path.name
            durations.append(audio.size / 16_000)
            silence_ratios.append(silence)
            snr_values.append(_snr_db(audio))
            noise_floors.append(_noise_floor_db(audio))
        report.usable_files = len(durations)
        if durations:
            report.total_duration_s = float(sum(durations))
            report.min_duration_s = float(min(durations))
            report.max_duration_s = float(max(durations))
            report.mean_silence_ratio = float(np.mean(silence_ratios))
            report.mean_snr_db = float(np.median(snr_values))
            report.noise_floor_db = float(np.median(noise_floors))
        return report


def _silence_ratio(audio: np.ndarray, threshold_db: float = -40.0) -> float:
    """Fraction of frames quieter than threshold (10 ms frames, RMS-based)."""
    frame = 160  # 10 ms @ 16 kHz
    if audio.size < frame:
        return 1.0
    audio = np.abs(np.asarray(audio, dtype=np.float32).ravel())
    frames = audio[: (audio.size // frame) * frame].reshape(-1, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    return float(np.mean(db < threshold_db))


def _noise_floor_db(audio: np.ndarray) -> float:
    """Estimated noise floor (dBFS, 10th-percentile frame RMS).

    Frames quieter than the voice (pauses, breaths, room tone) carry the
    noise floor; the 10th-percentile RMS is a robust estimate that ignores
    both speech and isolated clicks.
    """
    frame = 160  # 10 ms @ 16 kHz — same framing as _silence_ratio
    x = np.asarray(audio, dtype=np.float32).ravel()
    if x.size < frame:
        return -120.0
    frames = x[: (x.size // frame) * frame].reshape(-1, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    return float(np.percentile(db, 10))


def _snr_db(audio: np.ndarray) -> float:
    """Global speech-to-noise-floor ratio in dB (loud 80th pct vs quiet 10th pct).

    A single-number cleanliness metric: >35 dB ≈ studio, 20-35 dB ≈ good,
    <20 dB ≈ noisy (phone/recording noise the GAN will happily learn).
    """
    frame = 160
    x = np.asarray(audio, dtype=np.float32).ravel()
    if x.size < frame * 4:  # need at least 4 frames for percentiles
        return 0.0
    frames = x[: (x.size // frame) * frame].reshape(-1, frame)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    loud = float(np.percentile(db, 80))
    quiet = float(np.percentile(db, 10))
    return loud - quiet


def recommend_epochs(snr_db: float | None, total_duration_s: float) -> dict:
    """Official RVC FAQ Q9 epoch guidance, derived from measured dataset quality.

    FAQ Q9 (docs/en/faq_en.md, RVC-Project): "If the training dataset's
    audio quality is poor and the noise floor is high, 20-30 epochs are
    sufficient. Setting it too high won't improve the audio quality of your
    low-quality training set." / "If the training set audio quality is high,
    the noise floor is low ... 200 is acceptable."
    FAQ Q10: "A dataset of around 10min to 50min is recommended" (less than
    1 min is not supported).

    Returns a dict with the recommended range and the exact reason string,
    so the CLI/API can surface official guidance instead of guessing.
    """
    if snr_db is None:
        return {"min": 20, "max": 200, "quality": "unknown",
                "reason": "SNR not measured — official FAQ allows 20 (noisy) "
                          "to 200 (clean); start low and watch val_mel/speaker-sim"}
    if snr_db < 20.0:
        return {"min": 20, "max": 30, "quality": "noisy",
                "reason": "measured SNR %.1f dB < 20 dB — official FAQ Q9: "
                          "noisy/high-floor data needs only 20-30 epochs; "
                          "more will NOT improve quality (it learns the noise)"
                          % snr_db}
    if snr_db < 35.0:
        return {"min": 50, "max": 150, "quality": "good",
                "reason": "measured SNR %.1f dB — decent quality; 50-150 "
                          "epochs per official FAQ Q9 (10-50 min datasets), "
                          "watch the composite metric" % snr_db}
    return {"min": 100, "max": 200, "quality": "clean",
            "reason": "measured SNR %.1f dB ≥ 35 dB — clean data; official "
                      "FAQ Q9 allows up to 200 epochs for high-quality sets"
                      % snr_db}


def _denoise_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Best-effort spectral-gating denoise (noisereduce); never raises.

    The library is optional (Colab quality group: ``uv pip install
    noisereduce``); when missing, or when the call fails for any reason, the
    ORIGINAL audio is returned unchanged so preprocessing always completes.
    """
    try:
        import noisereduce as nr

        out = nr.reduce_noise(y=np.asarray(audio, dtype=np.float32),
                              sr=int(sr), stationary=False)
        out = np.asarray(out, dtype=np.float32)
        if out.shape == np.asarray(audio).shape and np.isfinite(out).all():
            return out
        logger.warning("denoise output invalid — keeping original audio")
        return audio
    except ImportError:
        logger.info("noisereduce not installed — --denoise is a no-op "
                    "(install the quality group: uv pip install noisereduce)")
        return audio
    except Exception:
        logger.warning("denoise failed — keeping original audio", exc_info=True)
        return audio


def _loudness_norm(audio: np.ndarray, target_db: float = -20.0,
                   max_gain: float = 10.0) -> np.ndarray:
    """Per-chunk RMS loudness normalisation (gain-capped, peak-guarded).

    Scale toward ``target_db`` dBFS RMS; the gain is capped (silence stays
    silent) and a 0.99 peak guard prevents clipping before the upstream
    peak-normalisation runs.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.size < 2:
        return x
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms <= 1e-8:
        return x
    target_rms = 10.0 ** (target_db / 20.0)
    gain = min(target_rms / rms, max_gain)
    out = x * gain
    peak = float(np.abs(out).max()) if out.size else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def _norm_write(tmp_audio: np.ndarray, key: str, idx: int, sr: int, max_peak: float,
                alpha: float, gt_wavs_dir: Path, wavs16k_dir: Path,
                loudness: bool = False, loudness_target_db: float = -20.0) -> bool:
    """Peak-normalise one chunk and write it at target-sr + 16 kHz (upstream).

    With ``loudness`` the chunk is first RMS-normalised to a common target —
    the researched fix for mixed-loudness datasets (consistent training
    targets for the GAN)."""
    tmp_max = float(np.abs(tmp_audio).max()) if tmp_audio.size else 0.0
    if not np.isfinite(tmp_max) or tmp_max <= 0 or tmp_max > 2.5:
        logger.warning("skipping invalid chunk %s_%s (peak=%s)", key, idx, tmp_max)
        return False
    if loudness:
        tmp_audio = _loudness_norm(tmp_audio, loudness_target_db)
    tmp_audio = (tmp_audio / tmp_max * (max_peak * alpha)) + (1 - alpha) * tmp_audio
    wavfile.write(str(gt_wavs_dir / f"{key}_{idx}.wav"), sr,
                  tmp_audio.astype(np.float32))
    audio16k = librosa.resample(tmp_audio, orig_sr=sr, target_sr=16_000)
    wavfile.write(str(wavs16k_dir / f"{key}_{idx}.wav"), 16_000,
                  audio16k.astype(np.float32))
    return True


def _slice_file(path: Path, key: str, sr: int, per: float, overlap: float,
                max_peak: float, alpha: float, bh, ah,
                gt_wavs_dir: Path, wavs16k_dir: Path,
                denoise: bool = False, loudness: bool = False,
                loudness_target_db: float = -20.0) -> int:
    """Slice + normalise + dual-rate write for one dataset file (upstream logic).

    ``denoise`` cleans the raw decoded file once, before slicing; ``loudness``
    normalises each sliced chunk (see module docstring)."""
    slicer = Slicer(
        sr=sr, threshold=-42, min_length=1500, min_interval=400,
        hop_size=15, max_sil_kept=500,
    )
    audio = load_audio(path, sr)
    if audio.size == 0:
        return 0
    if denoise:
        audio = _denoise_audio(audio, sr)
    audio = signal.lfilter(bh, ah, audio)  # causal high-pass
    tail = per + overlap
    idx = 0
    produced = 0
    for sliced in slicer.slice(audio):
        i = 0
        while True:
            start = int(sr * (per - overlap) * i)
            i += 1
            if len(sliced[start:]) > tail * sr:
                tmp = sliced[start : start + int(per * sr)]
                if _norm_write(tmp, key, idx, sr, max_peak, alpha,
                               gt_wavs_dir, wavs16k_dir, loudness,
                               loudness_target_db):
                    produced += 1
                idx += 1
            else:
                tmp = sliced[start:]
                idx += 1
                break
        if _norm_write(tmp, key, idx, sr, max_peak, alpha, gt_wavs_dir,
                       wavs16k_dir, loudness, loudness_target_db):
            produced += 1
    return produced


def _preprocess_task(task: tuple) -> int:
    """Process-pool entry point: unpack args and slice exactly one file."""
    (path_str, key, sr, per, overlap, max_peak, alpha, bh, ah,
     gt_str, wavs16k_str, denoise, loudness, loudness_target_db) = task
    return _slice_file(Path(path_str), key, sr, per, overlap, max_peak, alpha,
                       bh, ah, Path(gt_str), Path(wavs16k_str),
                       bool(denoise), bool(loudness), float(loudness_target_db))


class DatasetPreprocessor:
    """Slice + normalise + dual-rate write (upstream ``preprocess.py``).

    ``denoise``: optional pre-slicing spectral-gate denoise (noisereduce,
    guarded no-op when unavailable). ``loudness``: per-chunk RMS loudness
    normalisation to ``loudness_target_db`` — ON by default (consistent
    training targets on ANY mixed-loudness dataset; pass ``loudness=False``
    to reproduce the upstream output exactly).
    """

    def __init__(self, sr: int, exp_dir: Path, per: float = 3.7,
                 progress=None, denoise: bool = False, loudness: bool = True,
                 loudness_target_db: float = -20.0) -> None:
        self.denoise = bool(denoise)
        self.loudness = bool(loudness)
        self.loudness_target_db = float(loudness_target_db)
        self.slicer = Slicer(
            sr=sr, threshold=-42, min_length=1500, min_interval=400,
            hop_size=15, max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.gt_wavs_dir = exp_dir / "0_gt_wavs"
        self.wavs16k_dir = exp_dir / "1_16k_wavs"
        self.gt_wavs_dir.mkdir(parents=True, exist_ok=True)
        self.wavs16k_dir.mkdir(parents=True, exist_ok=True)
        self._progress = progress

    def _norm_write(self, tmp_audio: np.ndarray, key: str, idx: int) -> bool:
        return _norm_write(tmp_audio, key, idx, self.sr, self.max, self.alpha,
                           self.gt_wavs_dir, self.wavs16k_dir,
                           self.loudness, self.loudness_target_db)

    def process_file(self, path: Path, key: str) -> int:
        return _slice_file(path, key, self.sr, self.per, self.overlap, self.max,
                           self.alpha, self.bh, self.ah,
                           self.gt_wavs_dir, self.wavs16k_dir,
                           self.denoise, self.loudness, self.loudness_target_db)

    def run(self, dataset_path: Path, progress=None) -> dict:
        dataset_path = Path(dataset_path)
        files = sorted(
            p for p in dataset_path.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        )
        if not files:
            raise DatasetError(f"no audio files found in {dataset_path}")
        total = len(files)
        chunks = 0
        failed: list[str] = []
        started = time.time()
        workers = min(4, os.cpu_count() or 1)
        if workers > 1 and total >= 4:
            # parallel slicing (CPU-bound decode + filter + resample); output
            # stays deterministic because chunk names key off the sorted-file
            # index, and a partial pool failure falls back to a full sequential
            # pass that simply overwrites identical chunk files.
            tasks = [
                (str(p), str(i), self.sr, self.per, self.overlap, self.max,
                 self.alpha, self.bh, self.ah,
                 str(self.gt_wavs_dir), str(self.wavs16k_dir),
                 self.denoise, self.loudness, self.loudness_target_db)
                for i, p in enumerate(files)
            ]
            try:
                from concurrent.futures import ProcessPoolExecutor, as_completed

                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_preprocess_task, t): t for t in tasks}
                    for done, fut in enumerate(as_completed(futures), start=1):
                        task = futures[fut]
                        try:
                            chunks += fut.result()
                        except Exception:
                            logger.exception("preprocess failed: %s", task[0])
                            failed.append(Path(task[0]).name)
                        if progress:
                            progress(done / total, Path(task[0]).name)
                report = {
                    "files": total,
                    "chunks": chunks,
                    "failed": failed,
                    "seconds": round(time.time() - started, 1),
                    "gt_wavs": str(self.gt_wavs_dir),
                    "wavs16k": str(self.wavs16k_dir),
                    "parallel_workers": workers,
                }
                logger.info("preprocess complete: %s", report)
                return report
            except Exception:
                logger.warning(
                    "parallel preprocessing unavailable — sequential fallback",
                    exc_info=True,
                )
                chunks = 0
                failed = []
        for i, path in enumerate(files):
            try:
                chunks += self.process_file(path, str(i))
            except Exception:
                logger.exception("preprocess failed: %s", path)
                failed.append(path.name)
            if progress:
                progress((i + 1) / total, path.name)
        report = {
            "files": total,
            "chunks": chunks,
            "failed": failed,
            "seconds": round(time.time() - started, 1),
            "gt_wavs": str(self.gt_wavs_dir),
            "wavs16k": str(self.wavs16k_dir),
        }
        logger.info("preprocess complete: %s", report)
        return report
