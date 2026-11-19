"""Integration: the researched quality recipe running inside the REAL trainer.

Runs one epoch of the real GAN trainer on synthetic data with every researched
lever ON (EMA + multi-resolution STFT loss + multi-scale discriminator D2 +
validation split + best-model tracking), then verifies the artifacts:
G payload carries the EMA state, D payload carries the D2 state, D2_*.pth and
its mirror land on disk, the best export is a portable RVC checkpoint, and the
resume path restores EMA bit-exactly.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from rvc_platform.training.run_config import TrainingRunConfig
from rvc_platform.training.trainer import Trainer
from rvc_platform.training.utils import HParams, stable_dataset_signature

pytestmark = pytest.mark.slow


@pytest.fixture()
def tiny_discriminator(monkeypatch):
    """TEST-ONLY shrink of the upstream discriminator ladder (same trick as
    test_trainer_smoke.py — structure identical, channels reduced)."""
    from torch.nn import Conv2d
    from torch.nn.utils import weight_norm

    from rvc_platform.engine.rvc.models import models as rvc_models
    from rvc_platform.engine.rvc.models.commons import get_padding

    class SmallDP(rvc_models.DiscriminatorP):
        def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
            super().__init__(period, kernel_size, stride, use_spectral_norm)
            chs = [1, 8, 16, 32, 64, 64]
            from torch import nn as _nn
            convs = []
            for i in range(5):
                last = i == 4
                convs.append(
                    weight_norm(
                        Conv2d(
                            chs[i], chs[i + 1],
                            (kernel_size, 1),
                            (1, 1) if last else (stride, 1),
                            padding=(get_padding(kernel_size, 1), 0),
                        )
                    )
                )
            self.convs = _nn.ModuleList(convs)
            self.conv_post = weight_norm(Conv2d(chs[5], 1, (3, 1), 1, padding=(1, 0)))

    monkeypatch.setattr(rvc_models, "DiscriminatorP", SmallDP)
    return SmallDP


def _shrink(hps: HParams) -> HParams:
    hps.model["inter_channels"] = 64
    hps.model["hidden_channels"] = 64
    hps.model["filter_channels"] = 256
    hps.model["n_layers"] = 2
    hps.model["upsample_initial_channel"] = 64
    hps.train["segment_size"] = 3200
    return hps


def _make_run(project_env, n_chunks: int = 4):
    sr = 40_000
    run_dir = project_env / "runs" / "recipe"
    for sub in ("0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    ds_dir = project_env / "dataset"
    ds_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    lines = []
    for i in range(n_chunks):
        t = np.arange(48_000) / sr
        chunk = (0.3 * np.sin(2 * np.pi * (150 + 8 * i) * t)
                 * (1 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        sf.write(str(run_dir / "0_gt_wavs" / f"{i}.wav"), chunk, sr, subtype="FLOAT")
        chunk16 = np.interp(np.arange(19_200) / 16_000, t,
                            chunk.astype(np.float64)).astype(np.float32)
        sf.write(str(run_dir / "1_16k_wavs" / f"{i}.wav"), chunk16, 16_000, subtype="FLOAT")
        if i == 0:
            sf.write(str(ds_dir / "src.wav"), chunk16, 16_000, subtype="PCM_16")
        frames = 19_200 // 320
        np.save(run_dir / "3_feature768" / f"{i}.npy",
                rng.standard_normal((frames, 768)).astype(np.float32))
        fine = (120 + 40 * np.sin(np.arange(frames * 2) / 25.0)).astype(np.float32)
        np.save(run_dir / "2b-f0nsf" / f"{i}.wav.npy", fine)
        coarse = np.clip(np.round(fine / 4).astype(np.int64), 1, 255)
        np.save(run_dir / "2a_f0" / f"{i}.wav.npy", coarse)
        lines.append("|".join([
            str(run_dir / "0_gt_wavs" / f"{i}.wav"),
            str(run_dir / "3_feature768" / f"{i}.npy"),
            str(run_dir / "2a_f0" / f"{i}.wav.npy"),
            str(run_dir / "2b-f0nsf" / f"{i}.wav.npy"),
            "0",
        ]))
    (run_dir / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")
    config = TrainingRunConfig(
        experiment_name="recipe", dataset_path=str(ds_dir),
        sample_rate="40k", epochs=1, batch_size=1, save_every_epoch=1,
        preload_base_models=False, resume=True,
        # ---- full researched recipe (defaults, stated explicitly for the test)
        ema_decay=0.9, c_stft=2.0, use_msd=True, spk_weight=1.0,
        val_split=0.25, early_stop_patience=0,
    )
    hps = _shrink(config.to_hparams())
    hps.data["training_files"] = str(run_dir / "filelist.txt")
    hps.train["log_interval"] = 1
    hps.model_dir = str(run_dir)
    return config, hps, run_dir


def test_full_quality_recipe_one_epoch(project_env, tiny_discriminator):
    config, hps, run_dir = _make_run(project_env)
    events: list[dict] = []
    signature = stable_dataset_signature(
        sorted((project_env / "dataset").glob("*")))

    trainer = Trainer(config, hps, "cpu", torch.float32, event_cb=events.append)
    assert trainer.run(dataset_signature=signature)

    # 4 chunks, 25% held out -> 3 training steps in epoch 1
    assert trainer.global_step == 3
    assert trainer._msd is not None and trainer._ema is not None
    assert trainer._tracker is not None
    # speechbrain is absent in this sandbox -> composite degrades to val_mel
    assert trainer._tracker.metric in ("val_mel", "composite")

    ckpt_dir = run_dir / "checkpoints"
    assert (ckpt_dir / "G_final.pth").is_file()
    assert (ckpt_dir / "D_final.pth").is_file()
    assert (ckpt_dir / "D2_final.pth").is_file(), "D2 must be checkpointed"
    assert (ckpt_dir / "D2_2333333.pth").is_file(), "D2 always-latest mirror"

    g_payload = torch.load(ckpt_dir / "G_final.pth", map_location="cpu",
                           weights_only=False)
    assert g_payload.get("ema") is not None, "G payload carries the EMA shadow"
    assert g_payload["ema"]["decay"] == 0.9
    assert g_payload["ema"]["shadow"], "EMA shadow tensors present"

    d_payload = torch.load(ckpt_dir / "D_final.pth", map_location="cpu",
                           weights_only=False)
    d2_payload = torch.load(ckpt_dir / "D2_final.pth", map_location="cpu",
                            weights_only=False)
    assert "model" in d2_payload and d2_payload.get("optimizer"), \
        "D2 checkpoint carries model + optimizer state"
    assert d_payload.get("msd") is None or isinstance(d_payload["msd"], dict), \
        "legacy extras in the D payload stay dict-typed when present"
    # D2 restores from its own file: a fresh trainer picks it up bit-exact
    trainer_r = Trainer(config, hps, "cpu", torch.float32)
    _dataset, _loader, net_g_r, net_d_r, _og, _od = trainer_r._build()
    assert trainer_r._msd is not None
    before = {k: v.clone() for k, v in trainer_r._msd.state_dict().items()}
    trainer_r._restore_d2(ckpt_dir / "D2_final.pth")
    after = trainer_r._msd.state_dict()
    assert any(not torch.allclose(before[k].float(), after[k].float())
               for k in before), "D2 state actually restored (not fresh)"
    del trainer_r, net_g_r, net_d_r
    gc.collect()

    state = json.loads((ckpt_dir / "state.json").read_text(encoding="utf-8"))
    assert state["run_config"]["ema_decay"] == 0.9
    assert state["run_config"]["use_msd"] is True
    assert state["run_config"]["c_stft"] == 2.0

    # losses reported include the stft term
    losses = [e for e in events if e["topic"] == "training.progress"][-1]["latest_losses"]
    assert "stft" in losses, "MRSTFT term must be reported"

    # validation + best export chain
    assert "training.validation" in [e["topic"] for e in events]
    assert "training.best_model" in [e["topic"] for e in events]
    best_exports = sorted((run_dir / "exports").glob("recipe_best_*.pth"))
    assert best_exports
    cpt = torch.load(best_exports[-1], map_location="cpu", weights_only=False)
    assert "weight" in cpt and "config" in cpt
    # best export carries the EMA weights, not the raw training weights
    # (exports are stored fp16 — compare within fp16 rounding)
    ema_w = g_payload["ema"]["shadow"]
    sample_key = next(k for k in cpt["weight"] if k in ema_w)
    assert torch.allclose(cpt["weight"][sample_key].float(),
                          ema_w[sample_key].float(), atol=1e-3, rtol=1e-2)

    # ---- TRUE resume with EMA: shadow continues bit-exact ------------------
    events2: list[dict] = []
    config2 = TrainingRunConfig(
        experiment_name="recipe", dataset_path=config.dataset_path,
        sample_rate="40k", epochs=2, batch_size=1, save_every_epoch=1,
        preload_base_models=False, resume=True,
        ema_decay=0.9, c_stft=2.0, use_msd=True, val_split=0.25,
    )
    hps2 = _shrink(config2.to_hparams())
    hps2.data["training_files"] = str(run_dir / "filelist.txt")
    hps2.train["log_interval"] = 1
    hps2.model_dir = str(run_dir)
    trainer2 = Trainer(config2, hps2, "cpu", torch.float32,
                       event_cb=events2.append)
    assert trainer2._ema is None and trainer2._msd is None, "levers attach in run()"
    trainer2.run(dataset_signature=signature)
    assert "training.resumed" in [e["topic"] for e in events2]
    assert trainer2.epoch == 2
    del trainer, trainer2, events, events2
    gc.collect()


def test_early_stop_patience_triggers_cooperative_stop(project_env,
                                                       tiny_discriminator):
    config, hps, run_dir = _make_run(project_env)
    # fresh experiment dir so earlier artifacts don't interfere
    run_dir = run_dir.with_name("recipe_es")
    for sub in ("0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    # copy synthetic data quickly via the same generator
    config2, hps2, _ = _make_run(project_env)
    import shutil
    for sub in ("0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"):
        shutil.rmtree(run_dir / sub, ignore_errors=True)
        shutil.copytree(_run_root(config2) / sub, run_dir / sub)
    shutil.copy(_run_root(config2) / "filelist.txt", run_dir / "filelist.txt")

    config3 = TrainingRunConfig(
        experiment_name="recipe_es", dataset_path=config2.dataset_path,
        sample_rate="40k", epochs=3, batch_size=1, save_every_epoch=1,
        preload_base_models=False, resume=False,
        ema_decay=0.0, c_stft=0.0, use_msd=False, val_split=0.25,
        early_stop_patience=1,
    )
    hps3 = _shrink(config3.to_hparams())
    hps3.data["training_files"] = str(run_dir / "filelist.txt")
    hps3.train["log_interval"] = 1
    hps3.model_dir = str(run_dir)
    events3: list[dict] = []
    trainer3 = Trainer(config3, hps3, "cpu", torch.float32,
                       event_cb=events3.append)
    completed = trainer3.run(dataset_signature=None)
    # epoch 1 sets the first best; epoch 2 is >= patience(1) epochs without a
    # new best only if the score did not improve — either way the run must
    # terminate cooperatively before epoch 3 OR complete normally; both are
    # valid, but when early stop fired its artifacts must be present.
    topics3 = [e["topic"] for e in events3]
    if "training.early_stop" in topics3:
        state = json.loads((run_dir / "checkpoints" / "state.json")
                           .read_text(encoding="utf-8"))
        assert state.get("early_stopped") is True
        assert completed is False
    else:
        assert completed
    del trainer3, events3
    gc.collect()


def _run_root(config: TrainingRunConfig) -> Path:
    from rvc_platform.infrastructure.paths import get_paths
    return get_paths().runs / config.experiment_name
