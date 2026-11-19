"""Dataset quality check + preprocessing (upstream pipeline, cleaned).

Stages (each idempotent, each reported):
1. ``check``     — inventory, decode probe, corrupt-file detection, duration
                   stats, duplicate detection (content hash), silence ratio;
2. ``preprocess``— silence-aware slicing (vendored Slicer), 3.7 s chunks with
                   0.3 s overlap, high-pass 48 Hz, peak-normalisation
                   (alpha 0.75 / max 0.9 — upstream constants), dual output:
                   target-sr ``0_gt_wavs`` + 16 kHz ``1_16k_wavs``.
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
        report.usable_files = len(durations)
        if durations:
            report.total_duration_s = float(sum(durations))
            report.min_duration_s = float(min(durations))
            report.max_duration_s = float(max(durations))
            report.mean_silence_ratio = float(np.mean(silence_ratios))
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


def _norm_write(tmp_audio: np.ndarray, key: str, idx: int, sr: int, max_peak: float,
                alpha: float, gt_wavs_dir: Path, wavs16k_dir: Path) -> bool:
    """Peak-normalise one chunk and write it at target-sr + 16 kHz (upstream)."""
    tmp_max = float(np.abs(tmp_audio).max()) if tmp_audio.size else 0.0
    if not np.isfinite(tmp_max) or tmp_max <= 0 or tmp_max > 2.5:
        logger.warning("skipping invalid chunk %s_%s (peak=%s)", key, idx, tmp_max)
        return False
    tmp_audio = (tmp_audio / tmp_max * (max_peak * alpha)) + (1 - alpha) * tmp_audio
    wavfile.write(str(gt_wavs_dir / f"{key}_{idx}.wav"), sr,
                  tmp_audio.astype(np.float32))
    audio16k = librosa.resample(tmp_audio, orig_sr=sr, target_sr=16_000)
    wavfile.write(str(wavs16k_dir / f"{key}_{idx}.wav"), 16_000,
                  audio16k.astype(np.float32))
    return True


def _slice_file(path: Path, key: str, sr: int, per: float, overlap: float,
                max_peak: float, alpha: float, bh, ah,
                gt_wavs_dir: Path, wavs16k_dir: Path) -> int:
    """Slice + normalise + dual-rate write for one dataset file (upstream logic)."""
    slicer = Slicer(
        sr=sr, threshold=-42, min_length=1500, min_interval=400,
        hop_size=15, max_sil_kept=500,
    )
    audio = load_audio(path, sr)
    if audio.size == 0:
        return 0
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
                               gt_wavs_dir, wavs16k_dir):
                    produced += 1
                idx += 1
            else:
                tmp = sliced[start:]
                idx += 1
                break
        if _norm_write(tmp, key, idx, sr, max_peak, alpha, gt_wavs_dir, wavs16k_dir):
            produced += 1
    return produced


def _preprocess_task(task: tuple) -> int:
    """Process-pool entry point: unpack args and slice exactly one file."""
    (path_str, key, sr, per, overlap, max_peak, alpha, bh, ah,
     gt_str, wavs16k_str) = task
    return _slice_file(Path(path_str), key, sr, per, overlap, max_peak, alpha,
                       bh, ah, Path(gt_str), Path(wavs16k_str))


class DatasetPreprocessor:
    """Slice + normalise + dual-rate write (upstream ``preprocess.py``)."""

    def __init__(self, sr: int, exp_dir: Path, per: float = 3.7,
                 progress=None) -> None:
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
                           self.gt_wavs_dir, self.wavs16k_dir)

    def process_file(self, path: Path, key: str) -> int:
        return _slice_file(path, key, self.sr, self.per, self.overlap, self.max,
                           self.alpha, self.bh, self.ah,
                           self.gt_wavs_dir, self.wavs16k_dir)

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
                 str(self.gt_wavs_dir), str(self.wavs16k_dir))
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
