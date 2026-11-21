"""Feature extraction for training: F0 (2a/2b dirs) + HuBERT (3_feature*).

Idempotent — existing outputs are skipped, so interrupted extraction can be
re-run at zero cost. Numerics match upstream exactly (RMVPE threshold 0.03,
 Praat voicing 0.6, mel-bin quantisation, NaN rejection for HuBERT feats).
"""
from __future__ import annotations

import logging
import os
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


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Write an npy atomically so interrupted Colab jobs never cache a partial file."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.part")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _valid_npy(path: Path) -> bool:
    """Return true only for a readable, non-empty numeric cache file."""
    try:
        if not path.is_file() or path.stat().st_size < 128:
            return False
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        return arr.ndim >= 1 and arr.size > 0 and np.isfinite(arr).all()
    except (OSError, ValueError, EOFError):
        return False


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
        if _valid_npy(out1) and _valid_npy(out2):
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
            _atomic_save_npy(out2, f0.astype(np.float32))
            f0_mel = 1127 * np.log(1 + f0 / 700)
            f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (
                f0_mel_max - f0_mel_min
            ) + 1
            f0_mel[f0_mel <= 1] = 1
            f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
            coarse = np.rint(f0_mel).astype(np.int64)
            assert coarse.max() <= 255 and coarse.min() >= 1
            _atomic_save_npy(out1, coarse)
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


def _hubert_forward(model, source: torch.Tensor, version: str) -> torch.Tensor:
    """Padding-free batched forward (equal-length clips only).

    With identical clip lengths there is no padding and no attention mask, so
    this is the same computation the single-clip path performs — batching
    only reduces kernel-launch counts and raises GPU utilisation.
    """
    if version == "v1":
        outputs = model(
            input_values=source, attention_mask=None,
            output_hidden_states=True, return_dict=True,
        )
        return model.final_proj(outputs.hidden_states[9])
    return model(
        input_values=source, attention_mask=None,
        output_hidden_states=False, return_dict=True,
    ).last_hidden_state


def _extract_one(model, wav_path: Path, out_dir: Path, version: str, device: str,
                 is_half: bool, normalize: bool) -> bool:
    """Single-clip extraction (original CUDA-graph path). Returns success."""
    source = _read_16k(wav_path, normalize)
    padding_mask = torch.BoolTensor(source.shape).fill_(False)
    source = (source.half().to(device) if is_half else source.to(device))
    with torch.no_grad():
        feats = extract_hubert_features(
            model, source, version, padding_mask=padding_mask.to(device)
        )
    feats = feats.squeeze(0).float().cpu().numpy()
    if np.isnan(feats).sum() != 0:
        logger.warning("NaN features in %s", wav_path.name)
        return False
    _atomic_save_npy(out_dir / f"{wav_path.stem}.npy", feats)
    return True


def extract_features(run_dir: Path, version: str, device: str, is_half: bool,
                     progress: ProgressFn | None = None) -> dict:
    """Compute HuBERT content features (3_feature768 for v2 / 3_feature256 v1).

    Speed-up: clips are grouped by exact sample count (header-only probe) and
    pushed through the model in padding-free batches — the same numerics path
    as one-by-one processing (no padding, no attention mask) with far fewer
    kernel launches and much better GPU utilisation. Any batch failure falls
    back to per-file extraction, so this can never lose a clip.
    """
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
    todo = [f for f in files if not _valid_npy(out_dir / f"{f.stem}.npy")]
    success = failed = 0
    started = time.time()

    # group by exact frame count so every batch is padding-free
    by_len: dict[int, list[Path]] = {}
    for f in todo:
        try:
            by_len.setdefault(int(sf.info(str(f)).frames), []).append(f)
        except Exception:
            by_len.setdefault(-1, []).append(f)

    processed = 0
    for n, group in sorted(by_len.items(), key=lambda kv: kv[0]):
        # ~12 clips of 3.7 s per batch keeps activations well inside 16 GB
        max_batch = max(1, min(12, 700_000 // max(1, n)))
        for start in range(0, len(group), max_batch):
            batch = group[start : start + max_batch]
            batched_ok = False
            if n > 0:
                try:
                    rows = [_read_16k(p, normalize).squeeze(0) for p in batch]
                    source = torch.stack(rows, 0)
                    source = (source.half() if is_half else source).to(device)
                    with torch.no_grad():
                        feats = _hubert_forward(model, source, version)
                    feats = feats.float().cpu()
                    if feats.size(0) != len(batch):
                        raise RuntimeError("unexpected HuBERT output shape")
                    for i, p in enumerate(batch):
                        arr = feats[i].numpy()
                        if np.isnan(arr).sum() == 0:
                            _atomic_save_npy(out_dir / f"{p.stem}.npy", arr)
                            success += 1
                        else:
                            failed += 1
                            logger.warning("NaN features in %s", p.name)
                    batched_ok = True
                except Exception:
                    logger.warning(
                        "batched extraction failed (%s..%s) — per-file retry",
                        batch[0].name, batch[-1].name, exc_info=True,
                    )
            if not batched_ok:
                for p in batch:
                    try:
                        if _extract_one(model, p, out_dir, version, device,
                                        is_half, normalize):
                            success += 1
                        else:
                            failed += 1
                    except Exception:
                        failed += 1
                        logger.exception("feature extraction failed: %s", p.name)
            processed += len(batch)
            if progress:
                progress(processed / max(1, len(todo)), batch[-1].name)

    report = {
        "files": len(files), "processed": len(todo), "success": success,
        "skipped": len(files) - len(todo), "failed": failed,
        "dim": 768 if version == "v2" else 256,
        "seconds": round(time.time() - started, 1),
    }
    logger.info("hubert features: %s", report)
    return report


def write_filelist(run_dir: Path, version: str, f0: bool,
                   speaker_id: int = 0, sample_rate_hz: int | None = None) -> Path:
    """Generate filelist.txt (upstream line format).

    Upstream webui.py appends TWO mute-silence lines per speaker id (by
    reference into logs/mute) before shuffling — teaching silence-in ->
    silence-out. We do the same via ``mute_asset.mute_filelist_lines``.
    ``sample_rate_hz`` (the experiment's target rate) selects mute32k/40k/48k.
    """
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
    # upstream dose: two mute lines per speaker, shuffled into the filelist
    from rvc_platform.training.mute_asset import mute_filelist_lines
    sr_hz = int(sample_rate_hz or _infer_sr_hz(run_dir) or 48_000)
    lines.extend(mute_filelist_lines(sr_hz, version, f0, speaker_id))
    rng = np.random.default_rng(1234)
    rng.shuffle(lines)
    filelist = run_dir / "filelist.txt"
    filelist.write_text("\n".join(lines), encoding="utf-8")
    return filelist


def _infer_sr_hz(run_dir: Path) -> int | None:
    """Best-effort target-sr discovery for the mute line (0_gt_wavs header)."""
    try:
        wavs = sorted((Path(run_dir) / "0_gt_wavs").glob("*.wav"))
        if wavs:
            import soundfile as sf

            info = sf.info(str(wavs[0]))
            return int(info.samplerate)
    except Exception:
        pass
    return None
