"""Training checkpoint policies: explicit resume_from, mirrors, G_final ranking,
crash-save, D-scheduler restore. CPU, tiny models, real Trainer machinery
(discriminator shrink shared with the smoke test).
"""
from __future__ import annotations

import gc
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from rvc_platform.core.errors import CheckpointError
from rvc_platform.engine.rvc.models import models as rvc_models
from rvc_platform.engine.rvc.models.commons import get_padding
from rvc_platform.training import checkpointing
from rvc_platform.training.run_config import TrainingRunConfig
from rvc_platform.training.trainer import Trainer
from rvc_platform.training.utils import HParams, stable_dataset_signature

pytestmark = pytest.mark.slow


@pytest.fixture()
def tiny_discriminator(monkeypatch):
    """Same shrink as the smoke test (structure identical, channels smaller)."""

    class SmallDP(rvc_models.DiscriminatorP):
        def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
            super().__init__(period, kernel_size, stride, use_spectral_norm)
            chs = [1, 8, 16, 32, 64, 64]
            convs = []
            for i in range(5):
                last = i == 4
                convs.append(
                    torch.nn.Conv2d(
                        chs[i], chs[i + 1],
                        (kernel_size, 1),
                        (1, 1) if last else (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
            self.convs = torch.nn.ModuleList(
                [torch.nn.utils.weight_norm(c) for c in convs]
            )
            self.conv_post = torch.nn.utils.weight_norm(
                torch.nn.Conv2d(chs[5], 1, (3, 1), 1, padding=(1, 0))
            )

    monkeypatch.setattr(rvc_models, "DiscriminatorP", SmallDP)
    return SmallDP


def _shrink(hps):
    hps.model["inter_channels"] = 64
    hps.model["hidden_channels"] = 64
    hps.model["filter_channels"] = 256
    hps.model["n_layers"] = 2
    hps.model["upsample_initial_channel"] = 64
    hps.train["segment_size"] = 3200
    return hps


def _make_run(project_env, n_chunks: int = 2):
    from pathlib import Path

    sr = 40_000
    run_dir = project_env / "runs" / "policy"
    for sub in ("0_gt_wavs", "1_16k_wavs", "2a_f0", "2b-f0nsf", "3_feature768"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    ds_dir = project_env / "dataset"
    ds_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    lines = []
    for i in range(n_chunks):
        t = np.arange(48_000) / sr
        chunk = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        sf.write(str(run_dir / "0_gt_wavs" / f"{i}.wav"), chunk, sr, subtype="FLOAT")
        chunk16 = np.interp(np.arange(19_200) / 16_000, t, chunk.astype(np.float64)).astype(np.float32)
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
        experiment_name="policy", dataset_path=str(ds_dir),
        sample_rate="40k", epochs=1, batch_size=1, save_every_epoch=1,
        preload_base_models=False, resume=True,
    )
    hps = config.to_hparams()
    hps.data["training_files"] = str(run_dir / "filelist.txt")
    hps.train["log_interval"] = 1
    hps.model_dir = str(run_dir)
    return config, _shrink(hps), run_dir


def _trainer_for(config, run_dir, events):
    hps = config.to_hparams()
    hps.data["training_files"] = str(run_dir / "filelist.txt")
    hps.train["log_interval"] = 1
    hps.model_dir = str(run_dir)
    return Trainer(config, _shrink(hps), "cpu", torch.float32, event_cb=events.append)


def test_explicit_resume_from_named_checkpoint(project_env, tiny_discriminator):
    config, _, run_dir = _make_run(project_env)
    events: list[dict] = []
    signature = stable_dataset_signature(sorted((project_env / "dataset").glob("*")))
    ckpt_dir = run_dir / "checkpoints"

    # epoch 1: writes G_2/D_2 (2 steps) + mirrors + G_final
    trainer = _trainer_for(config, run_dir, events)
    trainer.run(dataset_signature=signature)
    assert (ckpt_dir / "G_2.pth").is_file()
    assert (ckpt_dir / "G_final.pth").is_file()
    assert (ckpt_dir / "G_2333333.pth").is_file(), "latest mirror must be written (save_latest default)"
    assert (ckpt_dir / "D_2333333.pth").is_file()
    del trainer, events
    gc.collect()

    # epoch 2 with HIGHER epochs -> periodic save lands on G_4 (steps continue)
    config2 = TrainingRunConfig(
        experiment_name="policy", dataset_path=config.dataset_path,
        sample_rate="40k", epochs=3, batch_size=1, save_every_epoch=1,
        preload_base_models=False, resume=True,
    )
    events2: list[dict] = []
    trainer2 = _trainer_for(config2, run_dir, events2)
    trainer2.run(dataset_signature=signature)
    assert (ckpt_dir / "G_4.pth").is_file()
    assert trainer2.epoch == 3
    mirror_payload = torch.load(ckpt_dir / "G_2333333.pth", map_location="cpu", weights_only=False)
    assert int(mirror_payload["global_step"]) == trainer2.global_step, "mirror tracks latest state"
    del trainer2, events2
    gc.collect()

    # EXPLICIT resume from the older G_2: epoch must rewind to 2 (step 2)
    config3 = TrainingRunConfig(
        experiment_name="policy", dataset_path=config.dataset_path,
        sample_rate="40k", epochs=3, batch_size=1, save_every_epoch=1,
        preload_base_models=False, resume=True, resume_from=ckpt_dir / "G_2.pth",
    )
    events3: list[dict] = []
    trainer3 = _trainer_for(config3, run_dir, events3)
    trainer3.run(dataset_signature=signature)
    resumed_ev = [e for e in events3 if e["topic"] == "training.resumed"]
    assert resumed_ev and resumed_ev[0].get("explicit") is True
    assert "G_2.pth" in str(resumed_ev[0]["checkpoint"])
    assert trainer3.epoch == 3 and trainer3.global_step >= 4
    del trainer3, events3
    gc.collect()


def test_resume_from_by_name_and_latest_keyword(project_env, tiny_discriminator):
    config, _, run_dir = _make_run(project_env)
    events: list[dict] = []
    signature = stable_dataset_signature(sorted((project_env / "dataset").glob("*")))
    trainer = _trainer_for(config, run_dir, events)
    trainer.run(dataset_signature=signature)
    del trainer, events
    gc.collect()
    ckpt_dir = run_dir / "checkpoints"

    # by bare name (relative to the run's checkpoint dir)
    cfg = TrainingRunConfig(
        experiment_name="policy", dataset_path=config.dataset_path,
        sample_rate="40k", epochs=1, batch_size=1,
        preload_base_models=False, resume=True, resume_from="G_2.pth",
    )
    t = _trainer_for(cfg, run_dir, [])
    assert t._resolve_explicit() == ckpt_dir / "G_2.pth"

    # 'latest' keyword resolves to the newest state
    cfg_latest = TrainingRunConfig(
        experiment_name="policy", dataset_path=config.dataset_path,
        sample_rate="40k", epochs=1, batch_size=1,
        preload_base_models=False, resume=True, resume_from="latest",
    )
    t_latest = _trainer_for(cfg_latest, run_dir, [])
    resolved = t_latest._resolve_explicit()
    # G_final exists with step >= numbered -> preferred by the ranking rule
    assert resolved == ckpt_dir / "G_final.pth"

    # bogus path raises a clean error
    cfg_bad = TrainingRunConfig(
        experiment_name="policy", dataset_path=config.dataset_path,
        sample_rate="40k", epochs=1, batch_size=1,
        preload_base_models=False, resume=True, resume_from="G_999.pth",
    )
    t_bad = _trainer_for(cfg_bad, run_dir, [])
    with pytest.raises(CheckpointError):
        t_bad._resolve_explicit()


def test_crash_save_writes_emergency_checkpoint(project_env, tiny_discriminator, monkeypatch):
    config, _, run_dir = _make_run(project_env)
    events: list[dict] = []
    signature = stable_dataset_signature(sorted((project_env / "dataset").glob("*")))

    # force a failure inside the epoch loop after the first batch
    from rvc_platform.training import trainer as trainer_mod

    def boom(*args, **kwargs):
        raise RuntimeError("simulated OOM")

    monkeypatch.setattr(trainer_mod.L, "discriminator_loss", boom)
    trainer = _trainer_for(config, run_dir, events)
    with pytest.raises(RuntimeError):
        trainer.run(dataset_signature=signature)
    assert trainer.state.value == "failed"
    failed_events = [e for e in events if e["topic"] == "training.failed"]
    assert failed_events, "failure event must be emitted"
    ckpt_dir = run_dir / "checkpoints"
    # emergency checkpoint kept the run resumable
    assert (ckpt_dir / "state.json").is_file()
    state = json.loads((ckpt_dir / "state.json").read_text(encoding="utf-8"))
    assert state["epoch"] >= 1
    del trainer, events
    gc.collect()


def test_d_scheduler_state_is_restored(project_env, tiny_discriminator):
    config, _, run_dir = _make_run(project_env)
    events: list[dict] = []
    signature = stable_dataset_signature(sorted((project_env / "dataset").glob("*")))
    trainer = _trainer_for(config, run_dir, events)
    trainer.run(dataset_signature=signature)
    # save_full stored sched_d on the D side
    d_payload = torch.load(run_dir / "checkpoints" / "D_final.pth",
                           map_location="cpu", weights_only=False)
    assert d_payload.get("scheduler") is not None, "D scheduler state must be saved"
    del trainer, events
    gc.collect()

    # build a fresh trainer and drive _restore: sched_d must be loaded
    config2 = TrainingRunConfig(
        experiment_name="policy", dataset_path=config.dataset_path,
        sample_rate="40k", epochs=1, batch_size=1,
        preload_base_models=False, resume=True,
    )
    t2 = _trainer_for(config2, run_dir, [])
    dataset, loader, net_g, net_d, optim_g, optim_d = t2._build()
    sched_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.999875)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.999875)
    t2._restore(net_g, net_d, optim_g, optim_d, sched_g, sched_d, None, signature)
    assert sched_d.state_dict() and "last_epoch" in sched_d.state_dict()
    assert sched_d.state_dict()["last_epoch"] == 1, "D scheduler resumed at epoch 1"
    del net_g, net_d, optim_g, optim_d, t2
    gc.collect()


def test_latest_step_checkpoint_ranking(tmp_path):
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    # numbered only -> highest digits
    for step in (5, 17, 3):
        torch.save({"model": {}, "global_step": step, "epoch": 1}, ckpt / f"G_{step}.pth")
    assert checkpointing.latest_step_checkpoint(ckpt) == ckpt / "G_17.pth"
    # G_final with a HIGHER step wins over numbered
    torch.save({"model": {}, "global_step": 40, "epoch": 4}, ckpt / "G_final.pth")
    assert checkpointing.latest_step_checkpoint(ckpt) == ckpt / "G_final.pth"
    # a NEWER numbered checkpoint written after a final save wins instead
    torch.save({"model": {}, "global_step": 60, "epoch": 6}, ckpt / "G_60.pth")
    assert checkpointing.latest_step_checkpoint(ckpt) == ckpt / "G_60.pth"


def test_training_service_resume_with_checkpoint(project_env, tmp_path):
    from rvc_platform.application.training_service import TrainingService
    from rvc_platform.infrastructure.processes.manager import ProcessManager

    config, _, run_dir = _make_run(project_env)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "G_7.pth").write_bytes(b"G7")
    (run_dir / "run_config.json").write_text(config.model_dump_json(), encoding="utf-8")

    svc = TrainingService(ProcessManager(project_env / "logs" / "processes"),
                          python_bin="true")  # 'true': instant exit, no real training
    result = svc.resume("policy", checkpoint="G_7.pth")
    assert result["run"] == "policy"
    # the resolved resume point is persisted with the run config
    persisted = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert persisted["resume_from"], "resume_from must be persisted for the worker"
    assert "G_7.pth" in persisted["resume_from"]

    # a nonexistent run fails cleanly (404 at the API layer)
    from rvc_platform.core.errors import TrainingError

    with pytest.raises(TrainingError):
        svc.resume("no_such_run")

    # 'latest' resolves to the newest existing checkpoint and starts the run
    (run_dir / "run_config.json").write_text(config.model_dump_json(), encoding="utf-8")
    result_latest = svc.resume("policy", checkpoint="latest")
    assert result_latest["run"] == "policy"
    persisted_latest = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert "G_7.pth" in persisted_latest["resume_from"]

    # an unknown checkpoint name fails cleanly
    with pytest.raises(TrainingError):
        svc.resume("policy", checkpoint="G_999.pth")
