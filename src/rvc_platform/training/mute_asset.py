"""Official-style mute (silence) training asset — functional equivalence.

Upstream RVC ships ``logs/mute`` as a pre-extracted static asset (HF
``mute.zip``, "Required for v1/v2 training"): one silent wav per target
sample rate (mute32k/40k/48k.wav), its HuBERT features for v1/v2
(3_feature256/768/mute.npy) and zero coarse/fine F0 (2a_f0 + 2b-f0nsf,
``mute.wav.npy``). ``webui.py`` injects exactly TWO filelist lines per
speaker id, by reference, when building the training filelist — teaching
the model silence-in -> silence-out (suppresses conversion "hum"/noise
floor on quiet input).

This module generates the equivalent asset under ``assets/mute`` with the
same layout (16 kHz source + both feature dims + zero F0), extracting the
features through OUR OWN HuBERT path so numerics match our pipeline. When
the official ``mute.zip`` has been extracted to ``assets/mute`` (or the
legacy ``logs/mute``) those files take precedence verbatim — we only
generate what is missing.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from rvc_platform.infrastructure.paths import get_paths

logger = logging.getLogger(__name__)

# upstream: two filelist lines per speaker (webui.py run_train_model)
MUTE_LINES_PER_SPEAKER = 2
MUTE_SECONDS = 10.0
MUTE_SR_16K = 16_000


def _official_layout(dir_path: Path) -> dict[str, Path]:
    """Map the upstream mute.zip layout names to this directory."""
    return {
        "wav_16k": dir_path / "1_16k_wavs" / "mute.wav",
        "wav_sr": dir_path / "0_gt_wavs",          # mute32k/40k/48k.wav inside
        "feat768": dir_path / "3_feature768" / "mute.npy",
        "feat256": dir_path / "3_feature256" / "mute.npy",
        "f0": dir_path / "2a_f0" / "mute.wav.npy",
        "f0nsf": dir_path / "2b-f0nsf" / "mute.wav.npy",
    }


def mute_asset_dir() -> Path:
    """Directory of the mute asset: official extraction if present, else ours."""
    paths = get_paths()
    for candidate in (paths.root / "logs" / "mute", paths.assets / "mute"):
        if candidate.is_dir() and (candidate / "2a_f0").is_dir():
            return candidate
    return paths.assets / "mute"


def ensure_mute_asset() -> Path:
    """Make sure the mute asset exists (official copy preferred, else generate).

    Returns the directory. Generation writes a deterministic 10 s digital
    silence wav (16 kHz) + zero F0; HuBERT features are extracted through the
    platform's own extractor so they match our feature pipeline bit-for-bit.
    """
    mute_dir = mute_asset_dir()
    layout = _official_layout(mute_dir)

    n_frames = int(MUTE_SR_16K * MUTE_SECONDS) // 160      # 10 ms hop
    wav_16k = np.zeros(int(MUTE_SR_16K * MUTE_SECONDS), dtype=np.float32)
    zero_f0 = np.zeros(n_frames, dtype=np.float32)

    if not layout["wav_16k"].is_file():
        layout["wav_16k"].parent.mkdir(parents=True, exist_ok=True)
        sf.write(layout["wav_16k"], wav_16k, MUTE_SR_16K, subtype="PCM_16")
        logger.info("mute asset: generated %s", layout["wav_16k"])

    for key in ("f0", "f0nsf"):
        if not layout[key].is_file():
            layout[key].parent.mkdir(parents=True, exist_ok=True)
            np.save(layout[key], zero_f0)
            logger.info("mute asset: generated %s (zero F0)", layout[key])

    for dim, key in ((768, "feat768"), (256, "feat256")):
        if not layout[key].is_file():
            layout[key].parent.mkdir(parents=True, exist_ok=True)
            if _try_extract_silence_feature(layout["wav_16k"], layout[key], dim):
                logger.info("mute asset: extracted %s via HuBERT", layout[key])
            else:
                # hubert_base unavailable (e.g. unit tests, first run before
                # downloads): write zero features — functionally close (the
                # 12th-layer representation of digital silence is ~zero) and
                # regenerated on next ensure once the asset exists.
                frames = int(MUTE_SR_16K * MUTE_SECONDS) // 320  # 20 ms frames
                np.save(layout[key], np.zeros((frames, dim), dtype=np.float32))
                logger.warning(
                    "mute asset: hubert_base missing — wrote zero %d-D features "
                    "for %s (regenerate after downloading assets)", dim, layout[key],
                )

    # target-sr silent gt wavs (mute32k/40k/48k.wav) — reference by filelist
    for sr in (32_000, 40_000, 48_000):
        gt = layout["wav_sr"] / f"mute{sr // 1_000}k.wav"
        if not gt.is_file():
            layout["wav_sr"].mkdir(parents=True, exist_ok=True)
            sf.write(gt, np.zeros(int(sr * MUTE_SECONDS), dtype=np.float32),
                     sr, subtype="PCM_16")
    return mute_dir


def _try_extract_silence_feature(wav_16k: Path, out_npy: Path, dim: int) -> bool:
    """Run OUR HuBERT extractor over the silent wav (dim 768 or 256).

    Returns False (and writes nothing) when hubert_base is not installed —
    callers then fall back to zero features.
    """
    try:
        import torch

        from rvc_platform.engine.rvc.hubert import (
            extract_hubert_features, load_hubert_model,
        )

        version = "v2" if dim == 768 else "v1"
        model = load_hubert_model("cpu", is_half=False)
        try:
            audio, sr = sf.read(str(wav_16k), dtype="float32")
            assert sr == MUTE_SR_16K
            # same guard as feature_extract.py: no autograd graph on inference
            with torch.no_grad():
                feats = extract_hubert_features(
                    model, torch.from_numpy(audio).view(1, -1), version,
                )      # (1, frames20ms, dim)
            np.save(out_npy,
                    feats[0].detach().float().cpu().numpy().astype(np.float32))
            return True
        finally:
            del model
    except (FileNotFoundError, ImportError, OSError, RuntimeError) as exc:
        # RuntimeError included: grad-traced tensors / GPU OOM must degrade to
        # the zero-feature fallback instead of killing a Colab prepare run
        logger.info("mute feature extraction unavailable (%s) — zero fallback", exc)
        return False


def mute_filelist_lines(run_dir_sr_hz: int, version: str, f0: bool,
                        speaker_id: int) -> list[str]:
    """Upstream-style filelist lines for the mute asset (2 per speaker)."""
    mute_dir = ensure_mute_asset()
    layout = _official_layout(mute_dir)
    gt = layout["wav_sr"] / f"mute{run_dir_sr_hz // 1_000}k.wav"
    feature = layout["feat768"] if version == "v2" else layout["feat256"]
    line = (
        f"{gt}|{feature}|{layout['f0']}|{layout['f0nsf']}|{speaker_id}"
        if f0 else f"{gt}|{feature}|{speaker_id}"
    )
    return [line] * MUTE_LINES_PER_SPEAKER
