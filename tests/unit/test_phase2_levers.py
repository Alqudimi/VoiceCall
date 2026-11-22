"""Unit tests for the Phase-2 researched quality levers.

Covers: online waveform augmentation (shapes/dtype/noise-SNR bound/gain
bound/dropout masking + disabled path), the warmup+cosine factor schedule
(ramp values, tail, pure-upstream passthrough), the discriminator LR scale
plumbing, the energy-VAD speech-ratio + filelist filter, and the new
run-config fields validation.
"""
from __future__ import annotations


import numpy as np
import pytest
import torch

from rvc_platform.training.augment import (
    WaveformAugmenter,
    build_augmenter,
)
from rvc_platform.training.schedules import WarmupCosineScheduler
from rvc_platform.training.vad_filter import filter_filelist, speech_ratio


# ------------------------------------------------------------------ helpers
class _Hps:
    """Minimal hps-like container mirroring the trainer's attribute layout."""

    def __init__(self, sr=40000, train=None):
        self.data = {"sampling_rate": sr}
        self.train = {"augment": 0, **(train or {})}


# ---------------------------------------------------------------- augment
def test_augmenter_disabled_by_default():
    assert build_augmenter(_Hps(), "cpu") is None


def test_augmenter_enabled_by_flag():
    aug = build_augmenter(_Hps(train={"augment": 1}), "cpu")
    assert aug is not None and isinstance(aug, WaveformAugmenter)


def test_augment_preserves_shape_dtype_and_finiteness():
    torch.manual_seed(0)
    aug = WaveformAugmenter(_Hps(), "cpu")
    wave = torch.randn(4, 1, 16000)
    out = aug.augment(wave)
    assert out.shape == wave.shape
    assert out.dtype == wave.dtype
    assert torch.isfinite(out).all()


def test_augment_noise_respects_snr_floor():
    torch.manual_seed(1)
    # force the noise branch on every call: p=1.0, tight SNR band = known SNR
    aug = WaveformAugmenter(_Hps(sr=16000, train={
        "aug_noise_p": 1.0, "aug_noise_snr_db_min": 20.0,
        "aug_noise_snr_db_max": 20.0, "aug_gain_p": 0.0, "aug_dropout_p": 0.0,
    }), "cpu")
    clean = torch.randn(2, 1, 32000) * 0.1
    out = aug.augment(clean)
    noise = (out - clean).squeeze(1)
    sig_rms = clean.pow(2).mean().sqrt().clamp_min(1e-12)
    noise_rms = noise.pow(2).mean().sqrt().clamp_min(1e-12)
    snr_db = 20.0 * torch.log10(sig_rms / noise_rms)
    # one draw around the configured 20 dB (per-batch single draw)
    assert 18.0 <= float(snr_db) <= 22.0


def test_augment_gain_bounded():
    torch.manual_seed(2)
    aug = WaveformAugmenter(_Hps(train={
        "aug_noise_p": 0.0, "aug_gain_p": 1.0, "aug_gain_db": 3.0,
        "aug_dropout_p": 0.0,
    }), "cpu")
    wave = torch.ones(3, 1, 8000) * 0.5
    out = aug.augment(wave).squeeze(1)
    per_item_ratio = out.mean(dim=-1) / 0.5
    # each item's gain is within ±3 dB
    lower = 10 ** (-3 / 20) - 1e-4
    upper = 10 ** (3 / 20) + 1e-4
    assert bool(((per_item_ratio >= lower) & (per_item_ratio <= upper)).all())


def test_augment_dropout_zeroes_spans():
    torch.manual_seed(3)
    aug = WaveformAugmenter(_Hps(sr=16000, train={
        "aug_noise_p": 0.0, "aug_gain_p": 0.0, "aug_dropout_p": 1.0,
        "aug_dropout_ms": 100, "aug_dropout_spans": 2,
    }), "cpu")
    wave = torch.ones(2, 1, 16000)
    out = aug.augment(wave).squeeze(1)
    # some samples must now be exactly zero (masked spans)
    assert float((out == 0).float().mean()) > 0.0
    assert float((out == 1).float().mean()) > 0.5  # most of the wave intact


def test_augment_identity_when_all_probs_zero():
    torch.manual_seed(4)
    aug = WaveformAugmenter(_Hps(train={
        "aug_noise_p": 0.0, "aug_gain_p": 0.0, "aug_dropout_p": 0.0,
    }), "cpu")
    wave = torch.randn(2, 1, 4096)
    assert torch.equal(aug.augment(wave), wave)


# --------------------------------------------------------------- schedule
def _optim():
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-4)


def test_schedule_pure_upstream_when_disabled():
    opt = _optim()
    sched = WarmupCosineScheduler(opt, steps_per_epoch=10, warmup_epochs=0,
                                  total_epochs=10, cosine_frac=0.0)
    for _ in range(25):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_schedule_warmup_ramp():
    opt = _optim()
    sched = WarmupCosineScheduler(opt, steps_per_epoch=10, warmup_epochs=1,
                                  total_epochs=5, cosine_frac=0.0)
    factors = []
    for _ in range(10):
        factors.append(opt.param_groups[0]["lr"] / 1e-4)
        sched.step()
    # strictly increasing ramp reaching 1.0 within the first epoch
    assert factors[0] == pytest.approx(1 / 10)
    assert all(b >= a - 1e-9 for a, b in zip(factors, factors[1:]))
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_schedule_cosine_tail_decays_to_fraction():
    opt = _optim()
    total_epochs = 4
    steps_per_epoch = 5
    sched = WarmupCosineScheduler(opt, steps_per_epoch=steps_per_epoch,
                                  warmup_epochs=0, total_epochs=total_epochs,
                                  cosine_frac=0.1)
    lrs = []
    for _ in range(total_epochs * steps_per_epoch):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[0] == pytest.approx(1e-4)
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:]))  # monotone decay
    # final steps approach the 0.1 target (the last factor is applied at the
    # final step; the very next would be the exact tail — allow one step's
    # progress worth of slack)
    assert lrs[-1] <= 1e-4 * 0.2


def test_schedule_factor_monotone_no_spike():
    opt = _optim()
    sched = WarmupCosineScheduler(opt, steps_per_epoch=7, warmup_epochs=1,
                                  total_epochs=3, cosine_frac=0.05)
    prev = 0.0
    for step in range(21):
        factor = sched._factor(step)
        assert factor <= 1.0 + 1e-9
        if step >= 7:  # after warmup the cosine is monotone non-increasing
            assert factor <= prev + 1e-9
        prev = factor


# ------------------------------------------------------------------- VAD
def test_speech_ratio_silence_vs_voice():
    sr = 16000
    silence = np.zeros(sr * 2, dtype=np.float32)
    voice = (np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr)
             * 0.5).astype(np.float32)
    assert speech_ratio(silence, sr) < 0.05
    assert speech_ratio(voice, sr) > 0.9


def test_speech_ratio_leading_silence_half():
    sr = 16000
    voice = (np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr)
             * 0.5).astype(np.float32)
    half = np.concatenate([np.zeros(sr, dtype=np.float32), voice])
    ratio = speech_ratio(half, sr)
    assert 0.4 <= ratio <= 0.75


def test_filter_filelist_drops_silent(tmp_path):
    from scipy.io import wavfile

    sr = 16000
    gt = tmp_path / "0_gt_wavs"
    gt.mkdir()
    voice = (np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr)
            * 0.5).astype(np.float32)
    wavfile.write(str(gt / "voice.wav"), sr, voice)
    wavfile.write(str(gt / "silence.wav"), sr,
                  np.zeros(sr * 2, dtype=np.float32))
    # feature files must exist for lines to be kept
    feat = tmp_path / "3_feature768"
    feat.mkdir()
    for stem in ("voice", "silence"):
        np.save(str(feat / f"{stem}.npy"), np.zeros((10, 768), dtype=np.float32))
    fl = tmp_path / "filelist.txt"
    fl.write_text(
        f"voice.wav|{feat / 'voice.npy'}|x|y|0\n"
        f"silence.wav|{feat / 'silence.npy'}|x|y|0\n",
        encoding="utf-8",
    )
    kept, dropped = filter_filelist(fl, fl, gt, sr, min_speech_ratio=0.5)
    assert kept == 1 and dropped == 1
    assert "voice.wav" in fl.read_text(encoding="utf-8")
    assert "silence.wav" not in fl.read_text(encoding="utf-8")


# -------------------------------------------------------------- run-config
def test_run_config_phase2_fields_defaults_and_validation(tmp_path):
    from rvc_platform.training.run_config import TrainingRunConfig

    cfg = TrainingRunConfig(experiment_name="t", dataset_path=tmp_path)
    assert cfg.augment is False
    assert cfg.warmup_epochs == 1
    assert cfg.cosine_frac == 0.0
    assert cfg.d_lr_scale == 1.0
    assert cfg.speech_filter_ratio == 0.25
    # to_hparams carries the levers into the trainer-visible train dict
    hps = cfg.to_hparams()
    assert hps.train["augment"] == 0
    assert hps.train["warmup_epochs"] == 1
    assert hps.train["cosine_frac"] == 0.0
    assert hps.train["d_lr_scale"] == 1.0


def test_run_config_phase2_enabled_roundtrip(tmp_path):
    from rvc_platform.training.run_config import TrainingRunConfig

    cfg = TrainingRunConfig(
        experiment_name="t2", dataset_path=tmp_path, augment=True,
        warmup_epochs=2, cosine_frac=0.1, d_lr_scale=0.9,
        aug_noise_p=0.2, aug_gain_db=1.5, speech_filter_ratio=0.3,
    )
    path = cfg.save()
    loaded = TrainingRunConfig.load(path.parent)
    assert loaded.augment is True
    assert loaded.warmup_epochs == 2
    assert loaded.cosine_frac == 0.1
    assert loaded.d_lr_scale == 0.9
    assert loaded.speech_filter_ratio == 0.3


def test_run_config_phase2_bounds(tmp_path):
    from pydantic import ValidationError

    from rvc_platform.training.run_config import TrainingRunConfig

    with pytest.raises(ValidationError):
        TrainingRunConfig(experiment_name="t", dataset_path=tmp_path,
                          aug_gain_db=99.0)
    with pytest.raises(ValidationError):
        TrainingRunConfig(experiment_name="t", dataset_path=tmp_path,
                          cosine_frac=1.5)
    with pytest.raises(ValidationError):
        TrainingRunConfig(experiment_name="t", dataset_path=tmp_path,
                          d_lr_scale=0.0)
