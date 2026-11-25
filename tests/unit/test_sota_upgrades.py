"""Unit tests for the SOTA voice-cloning upgrades and critical bug fixes."""
from __future__ import annotations

import json
import numpy as np
import pytest
import torch


# --------------------------------------------------------------------------- #
# 1) Unvoiced-preserving F0 octave cleaner
# --------------------------------------------------------------------------- #
def test_clean_f0_octaves_preserves_unvoiced_and_fixes_spike():
    from rvc_platform.engine.rvc.f0 import clean_f0_octaves

    f0 = np.array([0.0, 80.0, 82.0, 240.0, 81.0, 79.0, 0.0, 0.0], dtype=np.float32)
    cleaned = clean_f0_octaves(f0)
    # Unvoiced frames MUST remain strictly 0.0 (protects SineGen U/V and protect)
    assert cleaned[0] == 0.0
    assert cleaned[6] == 0.0 and cleaned[7] == 0.0
    # Isolated 1-frame octave spike (240 Hz between ~82 & ~81) is smoothed
    assert 78.0 <= cleaned[3] <= 84.0


# --------------------------------------------------------------------------- #
# 2) Preprocess loudness norm no-clip + duplicate skip
# --------------------------------------------------------------------------- #
def test_preprocess_norm_write_loudness_no_clip(tmp_path):
    from scipy.io import wavfile
    from rvc_platform.training.preprocess import _norm_write

    sr = 48_000
    t = np.arange(sr * 2) / sr
    # Very quiet input (peak = 0.02) — old bug amplified it to > 1.0!
    quiet = (0.02 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    gt_dir = tmp_path / "0_gt"
    w16_dir = tmp_path / "1_16k"
    gt_dir.mkdir()
    w16_dir.mkdir()
    ok = _norm_write(quiet, "0", 0, sr, 0.9, 0.75, gt_dir, w16_dir, loudness=True)
    assert ok is True
    _, written = wavfile.read(str(gt_dir / "0_0.wav"))
    assert float(np.abs(written).max()) <= 0.99 + 1e-5


def test_preprocess_deduplicates_identical_files(tmp_path):
    import soundfile as sf
    from rvc_platform.training.preprocess import DatasetPreprocessor

    ds = tmp_path / "ds"
    ds.mkdir()
    sr = 48_000
    t = np.arange(int(sr * 2.0)) / sr
    sig = (0.3 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    sf.write(str(ds / "a.wav"), sig, sr, subtype="PCM_16")
    sf.write(str(ds / "b_copy.wav"), sig, sr, subtype="PCM_16")

    exp = tmp_path / "exp"
    pp = DatasetPreprocessor(sr=sr, exp_dir=exp, loudness=True)
    report = pp.run(ds)
    assert report["files"] == 1
    assert report["duplicates_skipped"] == 1


# --------------------------------------------------------------------------- #
# 3) VAD filter preserves official mute silence lines
# --------------------------------------------------------------------------- #
def test_vad_filter_keeps_mute_lines(tmp_path):
    from scipy.io import wavfile
    from rvc_platform.training.vad_filter import filter_filelist

    sr = 48_000
    gt = tmp_path / "0_gt_wavs"
    gt.mkdir()
    mute_dir = tmp_path / "mute"
    mute_dir.mkdir()
    voice = (0.5 * np.sin(2 * np.pi * 200 * np.arange(sr * 2) / sr)).astype(np.float32)
    wavfile.write(str(gt / "voice.wav"), sr, voice)
    wavfile.write(str(gt / "room_silence.wav"), sr, np.zeros(sr * 2, dtype=np.float32))
    wavfile.write(str(mute_dir / "mute48k.wav"), sr, np.zeros(sr * 2, dtype=np.float32))

    fl = tmp_path / "filelist.txt"
    fl.write_text(
        f"{gt / 'voice.wav'}|f|a|b|0\n"
        f"{gt / 'room_silence.wav'}|f|a|b|0\n"
        f"{mute_dir / 'mute48k.wav'}|m|ma|mb|0\n",
        encoding="utf-8",
    )
    kept, dropped = filter_filelist(fl, fl, gt, sr, min_speech_ratio=0.25)
    txt = fl.read_text(encoding="utf-8")
    assert kept == 2 and dropped == 1
    assert "voice.wav" in txt
    assert "mute48k.wav" in txt
    assert "room_silence.wav" not in txt


# --------------------------------------------------------------------------- #
# 4) Multi-Scale Mel Loss + MRD Discriminator
# --------------------------------------------------------------------------- #
def test_multi_scale_mel_loss_behavior():
    from rvc_platform.training.ms_mel_loss import multi_scale_mel_loss

    t = torch.arange(17_280).float() / 48_000
    y = (0.3 * torch.sin(2 * torch.pi * 180 * t)).unsqueeze(0).unsqueeze(0)
    assert float(multi_scale_mel_loss(y, y.clone(), sr=48_000)) == pytest.approx(0.0, abs=1e-5)

    y_hat = (y * 0.5 + 0.05 * torch.randn_like(y)).requires_grad_(True)
    loss = multi_scale_mel_loss(y, y_hat, sr=48_000)
    assert torch.isfinite(loss) and float(loss) > 0.0
    loss.backward()
    assert y_hat.grad is not None and float(y_hat.grad.abs().sum()) > 0.0


def test_mrd_forward_and_backward():
    from rvc_platform.training import losses as L
    from rvc_platform.training.mrd import MultiResolutionDiscriminator

    mrd = MultiResolutionDiscriminator()
    y = torch.randn(2, 1, 8000)
    y_hat = torch.randn(2, 1, 8000, requires_grad=True)
    y_r, y_g, f_r, f_g = mrd(y, y_hat)
    assert len(y_r) == 3 and len(y_g) == 3
    assert len(f_r) == 3 and len(f_g) == 3
    loss_g, _ = L.generator_loss(y_g)
    loss_fm = L.feature_loss(f_r, f_g)
    (loss_g + loss_fm).backward()
    assert y_hat.grad is not None and float(y_hat.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# 5) Symmetric DiffAugment + Unified Scheduler Resume
# --------------------------------------------------------------------------- #
def test_diff_augment_pair_symmetric():
    from rvc_platform.training.augment import WaveformAugmenter

    class _H:
        data = {"sampling_rate": 48_000}
        train = {"augment": 1, "aug_gain_p": 1.0, "aug_gain_db": 2.0}

    aug = WaveformAugmenter(_H(), "cpu")
    real = torch.ones(4, 1, 1000)
    fake = torch.ones(4, 1, 1000) * 0.5
    r_aug, f_aug = aug.diff_augment_pair(real, fake)
    # Exact identical scale factor per batch element
    assert torch.allclose(r_aug, f_aug * 2.0, atol=1e-6)


def test_unified_scheduler_sync_to_step():
    from rvc_platform.training.schedules import WarmupCosineScheduler

    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-4)
    sched = WarmupCosineScheduler(
        opt, steps_per_epoch=10, warmup_epochs=1,
        total_epochs=20, cosine_frac=0.1, epoch_gamma=0.999,
    )
    sched.sync_to_step(50)
    assert sched.last_epoch == 50
    expected_lr = 1e-4 * sched._factor(50) * (0.999 ** 5)
    assert opt.param_groups[0]["lr"] == pytest.approx(expected_lr, rel=1e-6)


# --------------------------------------------------------------------------- #
# 6) Top-K Best Export Soup
# --------------------------------------------------------------------------- #
def test_build_best_export_soup(tmp_path):
    from rvc_platform.training.soup import build_best_export_soup

    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "exports").mkdir(parents=True)

    for i, (ep, sc, val) in enumerate(((10, 0.50, 1.0), (20, 0.30, 3.0), (30, 0.40, 2.0))):
        name = f"run_best_e{ep}_s{ep * 10}.pth"
        torch.save(
            {"weight": {"w": torch.full((2, 2), val).half()}, "info": f"{ep}epoch"},
            run_dir / "exports" / name,
        )
    hist = [
        {"epoch": 20, "score": 0.30, "export": "run_best_e20_s200.pth"},
        {"epoch": 30, "score": 0.40, "export": "run_best_e30_s300.pth"},
        {"epoch": 10, "score": 0.50, "export": "run_best_e10_s100.pth"},
    ]
    (run_dir / "checkpoints" / "best_history.json").write_text(json.dumps(hist), encoding="utf-8")

    res = build_best_export_soup(run_dir, k=3)
    assert res["ok"] is True
    out = torch.load(res["output"], map_location="cpu", weights_only=False)
    # Best model (3.0) has highest MWA weight -> average > uniform 2.0
    assert float(out["weight"]["w"].float().mean()) > 2.0
