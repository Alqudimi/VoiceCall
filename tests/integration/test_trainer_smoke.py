"""Real GAN training smoke test: full loop + checkpoint resume on CPU.

Builds a complete (synthetic-content) training run directory, runs ONE epoch
of the real trainer (real SynthesizerTrnMs768NSFsid + MultiPeriodDiscriminatorV2
+ real losses + real AdamW/ExponentialLR), verifies checkpoint/state/export
artifacts, then re-runs for a second epoch and verifies TRUE resume
(epoch/global_step continue; resume event emitted; dataset-signature guard).
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn.utils import weight_norm

from rvc_platform.engine.rvc.models import models as rvc_models
from rvc_platform.engine.rvc.models.commons import get_padding
from rvc_platform.training.run_config import TrainingRunConfig
from rvc_platform.training.trainer import Trainer
from rvc_platform.training.utils import HParams, stable_dataset_signature

pytestmark = pytest.mark.slow


@pytest.fixture()
def tiny_discriminator(monkeypatch):
    """TEST-ONLY shrink of the upstream 71.4M-param discriminator ladder
    (1-32-128-512-1024 -> 1-8-16-32-64). Structure, layer count, padding and
    the training loop stay identical; the full-size discriminator behaviour
    is the upstream design and is untouched in production code."""
    class SmallDP(rvc_models.DiscriminatorP):
        def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
            super().__init__(period, kernel_size, stride, use_spectral_norm)
            chs = [1, 8, 16, 32, 64, 64]
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
            self.convs = nn.ModuleList(convs)
            self.conv_post = weight_norm(Conv2d(chs[5], 1, (3, 1), 1, padding=(1, 0)))

    monkeypatch.setattr(rvc_models, "DiscriminatorP", SmallDP)
    return SmallDP

def _shrink(hps):
    """Same classes/loop, smaller tensors (CPU/RAM budget)."""
    hps.model["inter_channels"] = 64
    hps.model["hidden_channels"] = 64
    hps.model["filter_channels"] = 256
    hps.model["n_layers"] = 2
    hps.model["upsample_initial_channel"] = 64
    hps.train["segment_size"] = 3200  # tiny segments keep D-activations RAM small
    return hps




def _make_run(project_env, n_chunks: int = 3) -> tuple[TrainingRunConfig, HParams, Path]:

    sr = 40_000
    run_dir = project_env / "runs" / "smoke"
    for sub in ("0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    ds_dir = project_env / "dataset"
    ds_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    lines = []
    for i in range(n_chunks):
        # 1.2 s speech-like chunk at 40 kHz (+ 16 kHz twin)
        t = np.arange(48_000) / sr
        chunk = (0.3 * np.sin(2 * np.pi * 150 * t) * (1 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        sf.write(str(run_dir / "0_gt_wavs" / f"{i}.wav"), chunk, sr, subtype="FLOAT")
        chunk16 = np.interp(
            np.arange(19_200) / 16_000, t, chunk.astype(np.float64)
        ).astype(np.float32)
        sf.write(str(run_dir / "1_16k_wavs" / f"{i}.wav"), chunk16, 16_000, subtype="FLOAT")
        # one dummy source file so validate_dataset() passes
        if i == 0:
            sf.write(str(ds_dir / "src.wav"), chunk16, 16_000, subtype="PCM_16")

        frames = 19_200 // 320           # HuBERT frames (20 ms)
        np.save(run_dir / "3_feature768" / f"{i}.npy",
                rng.standard_normal((frames, 768)).astype(np.float32))
        fine = (120 + 40 * np.sin(np.arange(frames * 2) / 25.0)).astype(np.float32)
        np.save(run_dir / "2b-f0nsf" / f"{i}.wav.npy", fine)
        coarse = np.clip(np.round(fine / 4).astype(np.int64), 1, 255)
        np.save(run_dir / "2a_f0" / f"{i}.wav.npy", coarse)
        lines.append(
            "|".join(
                [
                    str(run_dir / "0_gt_wavs" / f"{i}.wav"),
                    str(run_dir / "3_feature768" / f"{i}.npy"),
                    str(run_dir / "2a_f0" / f"{i}.wav.npy"),
                    str(run_dir / "2b-f0nsf" / f"{i}.wav.npy"),
                    "0",
                ]
            )
        )
    (run_dir / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")

    config = TrainingRunConfig(
        experiment_name="smoke",
        dataset_path=str(ds_dir),
        sample_rate="40k",
        epochs=1,
        batch_size=1,
        save_every_epoch=1,
        preload_base_models=False,   # no pretrained assets in this sandbox
        resume=True,
    )
    hps = config.to_hparams()
    hps.data["training_files"] = str(run_dir / "filelist.txt")
    hps.train["log_interval"] = 1
    hps.model_dir = str(run_dir)
    return config, _shrink(hps), run_dir


def test_train_one_epoch_and_real_resume(project_env, tiny_discriminator):
    config, hps, run_dir = _make_run(project_env)
    events: list[dict] = []
    signature = stable_dataset_signature(sorted((project_env / "dataset").glob("*")))

    trainer = Trainer(config, hps, "cpu", torch.float32, event_cb=events.append)
    started = time.time()
    completed = trainer.run(dataset_signature=signature)
    assert completed
    assert trainer.global_step == 3, "3 chunks x batch 1 -> 3 steps"
    assert trainer.epoch == 1
    topics = [e["topic"] for e in events]
    assert "training.started" in topics
    assert "training.progress" in topics
    assert "training.epoch_done" in topics
    assert "training.finished" in topics
    losses = [e for e in events if e["topic"] == "training.progress"][-1]["latest_losses"]
    assert {"disc", "gen", "mel", "kl", "fm"} <= set(losses.keys())

    # ---- artifacts on disk (full-state + mirrors + state.json + export) ----
    ckpt_dir = run_dir / "checkpoints"
    assert (ckpt_dir / "G_final.pth").is_file()
    assert (ckpt_dir / "D_final.pth").is_file()
    state = json.loads((ckpt_dir / "state.json").read_text(encoding="utf-8"))
    assert state["epoch"] == 1 and state["global_step"] == 3
    exports = list((run_dir / "exports").glob("*.pth"))
    assert exports, "periodic inference checkpoint must be exported"
    del trainer, events
    gc.collect()  # free ~0.6 GB of model+optimizer state on RAM-tight machines

    # ---- TRUE resume: second run continues epoch/step, no fresh start ----
    config2 = TrainingRunConfig(
        experiment_name="smoke", dataset_path=config.dataset_path,
        sample_rate="40k", epochs=2, batch_size=1, save_every_epoch=1,
        preload_base_models=False, resume=True,
    )
    hps2 = _shrink(config2.to_hparams())
    hps2.data["training_files"] = str(run_dir / "filelist.txt")
    hps2.train["log_interval"] = 1
    hps2.model_dir = str(run_dir)
    events2: list[dict] = []
    trainer2 = Trainer(config2, hps2, "cpu", torch.float32, event_cb=events2.append)
    trainer2.run(dataset_signature=signature)
    topics2 = [e["topic"] for e in events2]
    assert "training.resumed" in topics2, "second run must resume, not restart"
    assert not any(e["topic"] == "training.started" and e.get("epoch", 1) == 1
                   for e in events2 if "epoch" in e), "no duplicate start epochs"
    assert trainer2.epoch == 2
    assert trainer2.global_step >= 6, "steps continue from 3 (not from 0)"
    state2 = json.loads((ckpt_dir / "state.json").read_text(encoding="utf-8"))
    assert state2["epoch"] == 2 and state2["global_step"] == trainer2.global_step
    del trainer2, events2
    gc.collect()

    # ---- dataset-signature guard: changed data must hard-fail ----
    (project_env / "dataset" / "extra.wav").write_bytes(b"RIFF" + b"\x00" * 1000)
    signature_changed = stable_dataset_signature(
        sorted((project_env / "dataset").glob("*"))
    )
    config3 = TrainingRunConfig(
        experiment_name="smoke", dataset_path=config.dataset_path,
        sample_rate="40k", epochs=3, batch_size=1,
        preload_base_models=False, resume=True,
    )
    hps3 = _shrink(config3.to_hparams())
    hps3.data["training_files"] = str(run_dir / "filelist.txt")
    hps3.model_dir = str(run_dir)
    trainer3 = Trainer(config3, hps3, "cpu", torch.float32)
    from rvc_platform.core.errors import CheckpointError

    with pytest.raises(CheckpointError):
        trainer3.run(dataset_signature=signature_changed)
    del trainer3
    gc.collect()

    # ---- legacy (vanilla RVC) checkpoint resume path ----
    legacy_g = {"model": {}, "optimizer": {}, "learning_rate": 1e-4, "iteration": 4}
    legacy_d = {"model": {}, "optimizer": {}, "learning_rate": 1e-4, "iteration": 4}
    # populate model entries with current shapes via a fresh build
    probe = Trainer(config3, hps3, "cpu", torch.float32)
    _, _, net_g, net_d, optim_g, optim_d = probe._build()
    legacy_g["model"] = net_g.state_dict()
    legacy_d["model"] = net_d.state_dict()
    legacy_g["optimizer"] = optim_g.state_dict()
    legacy_d["optimizer"] = optim_d.state_dict()
    del net_g, net_d, optim_g, optim_d, probe
    gc.collect()
    torch.save(legacy_g, ckpt_dir / "G_2333333.pth")
    torch.save(legacy_d, ckpt_dir / "D_2333333.pth")
    (ckpt_dir / "state.json").unlink()
    config4 = TrainingRunConfig(
        experiment_name="smoke", dataset_path=config.dataset_path,
        sample_rate="40k", epochs=5, batch_size=1,
        preload_base_models=False, resume=True,
    )
    hps4 = _shrink(config4.to_hparams())
    hps4.data["training_files"] = str(run_dir / "filelist.txt")
    hps4.model_dir = str(run_dir)
    events4: list[dict] = []
    trainer4 = Trainer(config4, hps4, "cpu", torch.float32, event_cb=events4.append)
    trainer4.run(dataset_signature=signature)
    assert "training.resumed.legacy" in [e["topic"] for e in events4]
    assert trainer4.epoch == 5
    del trainer4, events4
    gc.collect()

    # duration sanity: the whole 3-run suite must not hang
    assert time.time() - started < 900
