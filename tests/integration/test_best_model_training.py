"""Integration: best-model tracking inside a REAL (tiny) training run.

Builds a small synthetic run directory with enough chunks for a validation
split, runs the real GAN trainer on CPU for two epochs with save_best +
val_split enabled, and verifies the complete chain:

  deterministic split -> held-out validation each epoch -> best bar update
  -> best inference export + best.json on disk -> state.json carries best
  -> resume keeps the best bar (no regression on worse epochs).

Also asserts the async I/O contract: run() returns only after every queued
write has landed (G_/D_ checkpoints, mirrors, state.json, exports).
"""
from __future__ import annotations

import gc
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from rvc_platform.engine.rvc.models import models as rvc_models
from rvc_platform.engine.rvc.models.commons import get_padding
from rvc_platform.training.run_config import TrainingRunConfig
from rvc_platform.training.trainer import Trainer
from rvc_platform.training.utils import stable_dataset_signature

pytestmark = pytest.mark.slow


@pytest.fixture()
def tiny_discriminator(monkeypatch):
    """TEST-ONLY shrink (same as the smoke suite)."""
    from torch import nn
    from torch.nn import Conv2d
    from torch.nn.utils import weight_norm

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
    hps.model["inter_channels"] = 64
    hps.model["hidden_channels"] = 64
    hps.model["filter_channels"] = 256
    hps.model["n_layers"] = 2
    hps.model["upsample_initial_channel"] = 64
    hps.train["segment_size"] = 2_880  # multiple of the 48k hop (480)
    return hps


def _make_run(project_env, n_chunks: int = 6):
    sr = 48_000
    run_dir = project_env / "runs" / "bestrun"
    for sub in ("0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    ds_dir = project_env / "dataset"
    ds_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    lines = []
    for i in range(n_chunks):
        t = np.arange(48_000) / sr
        chunk = (0.3 * np.sin(2 * np.pi * (150 + 10 * i) * t)
                 * (1 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        sf.write(str(run_dir / "0_gt_wavs" / f"{i}.wav"), chunk, sr, subtype="FLOAT")
        chunk16 = np.interp(np.arange(19_200) / 16_000, t,
                            chunk.astype(np.float64)).astype(np.float32)
        sf.write(str(run_dir / "1_16k_wavs" / f"{i}.wav"), chunk16, 16_000,
                 subtype="FLOAT")
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
        experiment_name="bestrun", dataset_path=str(ds_dir),
        sample_rate="48k", version="v2", epochs=2, batch_size=1, save_every_epoch=1,
        preload_base_models=False, resume=True,
        save_best=True, val_split=0.34,          # 6 chunks -> 2 val / 4 train
    )
    hps = _shrink(config.to_hparams())
    hps.data["training_files"] = str(run_dir / "filelist.txt")
    hps.train["log_interval"] = 1
    hps.model_dir = str(run_dir)
    return config, hps, run_dir


def test_best_model_tracking_end_to_end(project_env, tiny_discriminator):
    config, hps, run_dir = _make_run(project_env)
    events: list[dict] = []
    signature = stable_dataset_signature(
        sorted((project_env / "dataset").glob("*")))

    # ---- the pinned validation split is derived inside run()
    trainer = Trainer(config, hps, "cpu", torch.float32, event_cb=events.append)
    completed = trainer.run(dataset_signature=signature)
    assert completed

    assert (run_dir / "filelist_train.txt").is_file()
    assert (run_dir / "filelist_val.txt").is_file()
    assert len((run_dir / "filelist_val.txt").read_text().splitlines()) == 2

    # 4 train chunks x batch 1 x 2 epochs = 8 steps (2 chunks held out)
    assert trainer.global_step == 8, "validation split must leave the loop intact"
    assert trainer._tracker is not None
    # v1.3.3: without speechbrain the selection metric folds in voiced-F0
    # RMSE (composite_f0); bare val_mel stayed only when F0 eval is absent
    assert trainer._tracker.metric in ("composite_f0", "val_mel")
    assert trainer._validator is not None

    topics = [e["topic"] for e in events]
    assert "training.validation" in topics
    assert "training.best_model" in topics, "epoch 1 must set the initial best"

    # ---- best bar bookkeeping
    best = json.loads((run_dir / "checkpoints" / "best.json")
                      .read_text(encoding="utf-8"))
    assert best["metric"] in ("composite_f0", "val_mel")
    assert best["epoch"] in (1, 2) and best["global_step"] in (4, 8)
    best_exports = sorted((run_dir / "exports").glob("bestrun_best_*.pth"))
    assert best_exports, "best-model inference export must exist"
    # the export is a real RVC inference checkpoint
    cpt = torch.load(best_exports[-1], map_location="cpu", weights_only=False)
    assert "weight" in cpt and "config" in cpt

    # ---- state.json carries the best snapshot + async writes all landed
    state = json.loads((run_dir / "checkpoints" / "state.json")
                       .read_text(encoding="utf-8"))
    assert state["epoch"] == 2 and state["global_step"] == 8
    assert state["best"]["metric"] in ("composite_f0", "val_mel")
    assert (run_dir / "checkpoints" / "G_final.pth").is_file()
    assert (run_dir / "checkpoints" / "G_2333333.pth").is_file()
    assert (run_dir / "exports" / "bestrun.pth").is_file()

    val_events = [e for e in events if e["topic"] == "training.validation"]
    assert val_events[0]["scores"]["val_mel"] > 0

    del trainer, events
    gc.collect()

    # ---- resume must not regress the bar even if epoch 3 gets worse
    config3 = TrainingRunConfig(
        experiment_name="bestrun", dataset_path=config.dataset_path,
        sample_rate="48k", version="v2", epochs=3, batch_size=1, save_every_epoch=1,
        preload_base_models=False, resume=True,
        save_best=True, val_split=0.34,
    )
    hps3 = _shrink(config3.to_hparams())
    hps3.data["training_files"] = str(run_dir / "filelist.txt")
    hps3.train["log_interval"] = 1
    hps3.model_dir = str(run_dir)
    events3: list[dict] = []
    trainer3 = Trainer(config3, hps3, "cpu", torch.float32,
                       event_cb=events3.append)
    trainer3.run(dataset_signature=signature)
    best3 = json.loads((run_dir / "checkpoints" / "best.json")
                       .read_text(encoding="utf-8"))
    # either kept the old best (epoch 3 worse) or improved it — never worse
    assert best3["score"] <= best["score"] + 1e-12
    assert "training.resumed" in [e["topic"] for e in events3]
    del trainer3, events3
    gc.collect()
