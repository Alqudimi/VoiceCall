"""Feature extraction for training: F0 (2a/2b dirs) + HuBERT (3_feature*).

Idempotent — existing outputs are skipped, so interrupted extraction can be
re-run at zero cost. Numerics match upstream exactly (RMVPE threshold 0.03,
 Praat voicing 0.6, mel-bin quantisation, NaN rejection for HuBERT feats).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from rvc_platform.core.errors import AssetMissingError
from rvc_platform.engine.rvc import f0 as f0_mod
from rvc_platform.engine.rvc.hubert import (
    extract_hubert_features,
    hubert_audio_requires_normalization,
    load_hubert_model,
)
from rvc_platform.infrastructure.paths import get_paths

logger = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


def _read_16k(wav_path: Path, normalize: bool) -> torch.Tensor:
    wav, sr = sf.read(str(wav_path))
    assert sr == 16_000, f"expected 16 kHz, got {sr} in {wav_path}"
    feats = torch.from_numpy(np.asarray(wav)).float()
    if feats.dim() == 2:
        feats = feats.mean(-1)
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats.view(1, -1)


def extract_f0(run_dir: Path, method: str, device: str, is_half: bool,
               progress: ProgressFn | None = None) -> dict:
    """Compute 2a_f0 (coarse) + 2b-f0nsf (fine) for every 16k chunk."""
    run_dir = Path(run_dir)
    wav_dir = run_dir / "1_16k_wavs"
    f0_dir = run_dir / "2a_f0"
    f0nsf_dir = run_dir / "2b-f0nsf"
    for d in (f0_dir, f0nsf_dir):
        d.mkdir(parents=True, exist_ok=True)
    files = sorted(wav_dir.glob("*.wav"))
    if not files:
        raise AssetMissingError(
            "no 1_16k_wavs chunks", "run preprocess first (`make preprocess`)"
        )
    extractor = f0_mod.get_f0_extractor(method, device, is_half)
    f0_bin, f0_max, f0_min = 256, 1100.0, 50.0
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    success = skipped = failed = 0
    started = time.time()
    for idx, wav_path in enumerate(files):
        out1 = f0_dir / f"{wav_path.name}.npy"
        out2 = f0nsf_dir / f"{wav_path.name}.npy"
        if out1.exists() and out2.exists():
            skipped += 1
            continue
        try:
            x, _ = sf.read(str(wav_path))
            x = np.asarray(x, dtype=np.float32)
            if x.ndim == 2:
                x = x.mean(axis=1)
            f0 = extractor.extract(x)
            p_len = x.shape[0] // 160
            if f0.shape[0] < p_len:
                f0 = np.pad(f0, (0, p_len - f0.shape[0]))
            f0 = f0[:p_len]
            f0 = f0_mod.interp_unvoiced(f0)
            if np.all(f0 == 0):
                failed += 1
                continue  # fully unvoiced chunk is meaningless to RVC training
            np.save(out2, f0.astype(np.float32), allow_pickle=False)
            f0_mel = 1127 * np.log(1 + f0 / 700)
            f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (
                f0_mel_max - f0_mel_min
            ) + 1
            f0_mel[f0_mel <= 1] = 1
            f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
            coarse = np.rint(f0_mel).astype(np.int64)
            assert coarse.max() <= 255 and coarse.min() >= 1
            np.save(out1, coarse, allow_pickle=False)
            success += 1
        except Exception:
            failed += 1
            logger.exception("f0 failed: %s", wav_path.name)
        if progress:
            progress((idx + 1) / len(files), wav_path.name)
    report = {"files": len(files), "success": success, "skipped": skipped,
              "failed": failed, "method": method, "seconds": round(time.time() - started, 1)}
    logger.info("f0 extraction: %s", report)
    return report


def extract_features(run_dir: Path, version: str, device: str, is_half: bool,
                     progress: ProgressFn | None = None) -> dict:
    """Compute HuBERT content features (3_feature768 for v2 / 3_feature256 v1)."""
    run_dir = Path(run_dir)
    wav_dir = run_dir / "1_16k_wavs"
    out_dir = run_dir / ("3_feature768" if version == "v2" else "3_feature256")
    out_dir.mkdir(parents=True, exist_ok=True)
    hubert_dir = get_paths().hubert
    if not (hubert_dir / "config.json").is_file():
        raise AssetMissingError(
            "HuBERT content encoder (assets/hubert_base)",
            "run `make download-models`",
        )
    model = load_hubert_model(device, is_half)
    normalize = hubert_audio_requires_normalization()
    files = sorted(wav_dir.glob("*.wav"))
    if not files:
        raise AssetMissingError("no 1_16k_wavs chunks", "run preprocess first")
    todo = [f for f in files if not (out_dir / f"{f.stem}.npy").exists()]
    success = failed = 0
    started = time.time()
    for idx, wav_path in enumerate(todo):
        try:
            source = _read_16k(wav_path, normalize)
            padding_mask = torch.BoolTensor(source.shape).fill_(False)
            source = (source.half().to(device) if is_half else source.to(device))
            with torch.no_grad():
                feats = extract_hubert_features(
                    model, source, version, padding_mask=padding_mask.to(device)
                )
            feats = feats.squeeze(0).float().cpu().numpy()
            if np.isnan(feats).sum() == 0:
                np.save(out_dir / f"{wav_path.stem}.npy", feats, allow_pickle=False)
                success += 1
            else:
                failed += 1
                logger.warning("NaN features in %s", wav_path.name)
        except Exception:
            failed += 1
            logger.exception("feature extraction failed: %s", wav_path.name)
        if progress:
            progress((idx + 1) / max(1, len(todo)), wav_path.name)
    report = {
        "files": len(files), "processed": len(todo), "success": success,
        "skipped": len(files) - len(todo), "failed": failed,
        "dim": 768 if version == "v2" else 256,
        "seconds": round(time.time() - started, 1),
    }
    logger.info("hubert features: %s", report)
    return report


def write_filelist(run_dir: Path, version: str, f0: bool,
                   speaker_id: int = 0) -> Path:
    """Generate filelist.txt (upstream line format)."""
    run_dir = Path(run_dir)
    gt_dir = run_dir / "0_gt_wavs"
    feature_dir = run_dir / ("3_feature768" if version == "v2" else "3_feature256")
    f0_dir = run_dir / "2a_f0"
    f0nsf_dir = run_dir / "2b-f0nsf"
    lines: list[str] = []
    for gt in sorted(gt_dir.glob("*.wav")):
        stem = gt.stem
        feat = feature_dir / f"{stem}.npy"
        if not feat.exists():
            continue
        if f0:
            a = f0_dir / f"{stem}.wav.npy"
            b = f0nsf_dir / f"{stem}.wav.npy"
            if not (a.exists() and b.exists()):
                continue
            lines.append(f"{gt}|{feat}|{a}|{b}|{speaker_id}")
        else:
            lines.append(f"{gt}|{feat}|{speaker_id}")
    if not lines:
        raise AssetMissingError(
            "no complete training entries",
            "preprocess + f0 + feature extraction must finish first",
        )
    rng = np.random.default_rng(1234)
    rng.shuffle(lines)
    filelist = run_dir / "filelist.txt"
    filelist.write_text("\n".join(lines), encoding="utf-8")
    return filelist
