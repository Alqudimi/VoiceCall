"""Unit tests for the researched quality levers (UPGRADE_REPORT_AR.md).

Covers: generator EMA (math, warmup, roundtrip), the multi-resolution STFT
loss, the multi-scale discriminator contract, the composite evaluation
metrics (with injected stubs) and the new run-config fields/plumbing.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from rvc_platform.training import stft_loss
from rvc_platform.training.ema import GeneratorEMA


# ------------------------------------------------------------------------ EMA
class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 6)
        self.register_buffer("step_count", torch.zeros(1))

    def forward(self, x):  # pragma: no cover — shape only
        return self.linear(x)


def test_ema_shadow_matches_manual_math():
    torch.manual_seed(0)
    model = _Tiny()
    ema = GeneratorEMA(model, decay=0.9, warmup=False)
    with torch.no_grad():
        for _ in range(3):
            model.linear.weight.add_(torch.randn_like(model.linear.weight))
            expected = ema.shadow["linear.weight"].clone()
            model_w = model.linear.weight.clone()
            expected = 0.9 * expected + 0.1 * model_w
            ema.update(model)
    assert torch.allclose(ema.shadow["linear.weight"], expected, atol=1e-6)
    assert ema.step == 3


def test_ema_warmup_ramp_binds_early_decay():
    model = _Tiny()
    ema = GeneratorEMA(model, decay=0.999, warmup=True)
    assert ema._current_decay() == pytest.approx(0.1)     # (1+0)/(10+0)
    ema.step = 10_000
    assert ema._current_decay() == pytest.approx(0.999)   # ramp released


def test_ema_export_state_dict_keeps_buffers_and_params():
    model = _Tiny()
    ema = GeneratorEMA(model, decay=0.99)
    with torch.no_grad():
        model.linear.weight.mul_(0.5)   # live drifts away from shadow
        model.step_count.fill_(7.0)
    exported = ema.export_state_dict(model)
    live = model.state_dict()
    assert set(exported) == set(live), "must be loadable with strict=True"
    # parameters come from the shadow (unchanged), buffers copied live
    assert not torch.allclose(exported["linear.weight"], live["linear.weight"])
    assert torch.equal(exported["step_count"], live["step_count"])


def test_ema_apply_to_swaps_and_restores():
    model = _Tiny()
    ema = GeneratorEMA(model, decay=0.99)
    with torch.no_grad():
        model.linear.weight.mul_(0.25)   # live drifts away from the shadow
    mutated = model.state_dict()["linear.weight"].clone()
    ema.apply_to(model)
    swapped = model.state_dict()["linear.weight"].clone()
    # the swap loads the shadow (the initial weights), not the mutated ones
    assert torch.allclose(swapped, ema.shadow["linear.weight"])
    assert not torch.allclose(swapped, mutated)


def test_ema_state_dict_roundtrip():
    model = _Tiny()
    ema = GeneratorEMA(model, decay=0.98, warmup=False)
    ema.update(model)
    ema.update(model)
    saved = ema.state_dict()
    ema2 = GeneratorEMA(_Tiny(), decay=0.5)          # different model/state
    ema2.load_state_dict(saved)
    assert ema2.decay == 0.98 and ema2.step == 2
    assert torch.allclose(ema2.shadow["linear.weight"],
                          ema.shadow["linear.weight"])


def test_ema_rejects_bad_decay():
    with pytest.raises(ValueError):
        GeneratorEMA(_Tiny(), decay=1.0)
    with pytest.raises(ValueError):
        GeneratorEMA(_Tiny(), decay=0.0)


# -------------------------------------------------------------------- MRSTFT
def _noisy_pair(n=8000, noise=0.05, seed=1):
    torch.manual_seed(seed)
    t = torch.arange(float(n)) / 16000
    y = torch.sin(2 * torch.pi * 220 * t).unsqueeze(0)
    yhat = y + noise * torch.randn(1, n)
    return y, yhat


def test_mr_stft_zero_on_identical_inputs():
    y, _ = _noisy_pair()
    assert float(stft_loss.mr_stft_loss(y, y.clone())) == pytest.approx(0.0, abs=1e-6)


def test_mr_stft_positive_and_finite_on_degraded_input():
    y, yhat = _noisy_pair()
    value = stft_loss.mr_stft_loss(y, yhat)
    assert torch.isfinite(value) and float(value) > 0
    # batched (B, 1, T) tensors are accepted too
    value_b = stft_loss.mr_stft_loss(y.unsqueeze(0), yhat.unsqueeze(0))
    assert float(value_b) == pytest.approx(float(value))


def test_mr_stft_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        stft_loss.mr_stft_loss(torch.zeros(1, 100), torch.zeros(1, 101))


def test_mr_stft_calibrated_weight_contribution():
    """c_stft=2.0 keeps the term a few percent of the G loss (report §2.2) —
    checked on a harmonic speech-like pair like the calibration script."""
    sr, T = 40_000, 12_800
    t = torch.arange(T).float() / sr
    f0 = 120 + 40 * torch.sin(2 * torch.pi * 3.0 * t)
    def _speechlike():
        x = torch.zeros(T)
        for k in range(1, 40):
            x += (1.0 / k) * torch.sin(2 * torch.pi * k * f0 * t)
        x = x + 0.02 * torch.randn(T)
        return x / x.abs().max()
    y = torch.stack([_speechlike(), _speechlike() * 0.9], 0)
    yhat = y * 0.92 + 0.05 * torch.randn_like(y)
    term = float(stft_loss.mr_stft_loss(y, yhat)) * 2.0
    # the calibrated operating point: a few percent of the mel term (~15-30)
    assert 0.5 < term < 6.0


# ----------------------------------------------------------------------- MSD
def test_msd_forward_contract_matches_mpd():
    from rvc_platform.training.msd import MultiScaleDiscriminator

    torch.manual_seed(0)
    msd = MultiScaleDiscriminator(scales=(2, 4))
    y = torch.randn(2, 1, 8000)
    y_hat = torch.randn(2, 1, 8000)
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = msd(y, y_hat)
    assert len(y_d_rs) == 2 and len(y_d_gs) == 2
    assert len(fmap_rs) == 2 and len(fmap_gs) == 2
    for r, g in zip(y_d_rs, y_d_gs):
        assert r.shape == g.shape
    # adversarial gradient flows through the module
    loss = sum(torch.mean((1 - r) ** 2) + torch.mean(g ** 2)
               for r, g in zip(y_d_rs, y_d_gs))
    loss.backward()
    grads = [p.grad for p in msd.parameters() if p.requires_grad]
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)


def test_msd_works_with_existing_loss_functions():
    from rvc_platform.training import losses as L
    from rvc_platform.training.msd import MultiScaleDiscriminator

    msd = MultiScaleDiscriminator(scales=(2, 4))
    y, y_hat = torch.randn(1, 1, 4000), torch.randn(1, 1, 4000)
    r, g, _, _ = msd(y, y_hat.detach())
    loss, r_list, g_list = L.discriminator_loss(r, g)
    assert torch.isfinite(loss) and len(r_list) == len(g_list) == 2
    gl, gen_list = L.generator_loss(g)
    assert torch.isfinite(gl)


# --------------------------------------------------------------- eval metrics
def test_composite_score_formula():
    from rvc_platform.training import eval_metrics as M

    assert M.composite_score(0.4, 0.9, spk_weight=1.0) == pytest.approx(0.5)
    assert M.composite_score(0.4, None) == pytest.approx(0.4)   # fallback
    assert M.composite_score(0.4, 0.9, spk_weight=0.0) == pytest.approx(0.4)


def test_f0_rmse_voiced_only_alignment():
    from rvc_platform.training import eval_metrics as M

    gt = np.array([100.0, 110.0, 0.0, 130.0], dtype=np.float32)
    pred = np.array([100.0, 110.0, 500.0, 120.0], dtype=np.float32)
    out = M.f0_rmse(lambda wav: pred, np.zeros(64, dtype=np.float32), gt)
    # unvoiced frame (index 2) is excluded; voiced errors are 0, 0, 10 Hz
    assert out == pytest.approx(float(np.sqrt(100.0 / 3)))
    # all-unvoiced ground truth -> no metric
    assert M.f0_rmse(lambda wav: pred, np.zeros(64, dtype=np.float32),
                     np.zeros(4, dtype=np.float32)) is None


def test_speaker_similarity_with_stub_embedder(monkeypatch):
    from rvc_platform.training import eval_metrics as M

    class _Stub:
        def embed(self, wav16k):
            base = wav16k.float()[:, :1]  # (B, 1) — the sign of the first sample
            return torch.nn.functional.normalize(base, dim=-1)

    sim = M.speaker_similarity(_Stub(), torch.ones(2, 1600), torch.ones(2, 1600), 16000)
    assert sim == pytest.approx(1.0, abs=1e-5)
    sim2 = M.speaker_similarity(_Stub(), torch.ones(2, 1600), -torch.ones(2, 1600), 16000)
    assert sim2 == pytest.approx(-1.0, abs=1e-5)


def test_resample_16k_identity_and_downsample():
    from rvc_platform.training import eval_metrics as M

    x = torch.randn(2, 32000)
    assert torch.equal(M.resample_16k(x, 16000), x.float())
    out = M.resample_16k(x, 32000)
    assert out.shape[-1] == 16000 and torch.isfinite(out).all()


def test_build_speaker_embedder_env_off(monkeypatch):
    from rvc_platform.training import eval_metrics as M

    monkeypatch.setenv("RVC_SPK_SIM", "0")
    assert M.try_build_speaker_embedder("cpu") is None


# ------------------------------------------------------------------ run-config
def test_run_config_quality_fields_defaults_and_validation(tmp_path):
    from rvc_platform.training.run_config import TrainingRunConfig

    ds = tmp_path / "ds"
    ds.mkdir()
    cfg = TrainingRunConfig(experiment_name="q", dataset_path=ds)
    assert cfg.ema_decay == 0.999
    assert cfg.c_stft == 2.0
    assert cfg.use_msd is True
    assert cfg.spk_weight == 1.0
    assert cfg.early_stop_patience == 30
    assert cfg.val_split == 0.1
    assert cfg.denoise is False and cfg.loudness is True

    with pytest.raises(Exception):
        TrainingRunConfig(experiment_name="q", dataset_path=ds, ema_decay=1.0)
    with pytest.raises(Exception):
        TrainingRunConfig(experiment_name="q", dataset_path=ds, c_stft=-1.0)
    with pytest.raises(Exception):
        TrainingRunConfig(experiment_name="q", dataset_path=ds,
                         early_stop_patience=-5)

    hps = cfg.to_hparams()
    assert hps.train["ema_decay"] == 0.999
    assert hps.train["c_stft"] == 2.0
    assert hps.train["use_msd"] == 1
    assert hps.train["spk_weight"] == 1.0
    assert hps.train["early_stop_patience"] == 30
    assert hps.train["f0_eval"] == 1


def test_run_config_accepts_crepe_f0_method(tmp_path):
    from rvc_platform.training.run_config import TrainingRunConfig

    ds = tmp_path / "ds"
    ds.mkdir()
    cfg = TrainingRunConfig(experiment_name="q", dataset_path=ds,
                            f0_method="crepe")
    assert cfg.f0_method == "crepe"
    with pytest.raises(Exception):
        TrainingRunConfig(experiment_name="q", dataset_path=ds, f0_method="harvest")


def test_checkpoint_payloads_roundtrip_with_ema_and_msd(tmp_path):
    from rvc_platform.training import checkpointing

    model = _Tiny()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.999)
    payload = checkpointing.build_full_payload(model, optim, sched, None, 3, 30)
    payload["ema"] = {"decay": 0.999, "shadow": {"linear.weight": torch.zeros(6, 6)}}
    payload["msd"] = {"model": {"discs.0.x": torch.zeros(1)}, "optimizer": None}
    path = tmp_path / "G_final.pth"
    checkpointing.write_checkpoint_files(
        payload, path, dict(payload), tmp_path / "D_final.pth",
        None, tmp_path / "state.json", False,
    )
    model2 = _Tiny()
    optim2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.ExponentialLR(optim2, gamma=0.999)
    meta = checkpointing.load_full(path, model2, optim2, sched2)
    assert meta["epoch"] == 3 and meta["global_step"] == 30
    assert meta["ema"]["decay"] == 0.999
    assert "msd" in meta
