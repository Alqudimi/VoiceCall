from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rvc_platform.training.huggingface import publish_run
from rvc_platform.training.run_config import TrainingRunConfig
from rvc_platform.training.worker import _clear_previous_training


def test_publish_private_models_only(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    exports = run_dir / "exports"
    exports.mkdir(parents=True)
    (exports / "voice.pth").write_bytes(b"weights")
    (run_dir / "added_IVF1.index").write_bytes(b"index")
    (run_dir / "source.wav").write_bytes(b"private audio")
    (run_dir / "filelist.txt").write_text("private dataset paths")
    calls = {"created": [], "uploaded": []}

    class FakeHfApi:
        def __init__(self, token):
            assert token == "hf-secret"

        def create_repo(self, **kwargs):
            calls["created"].append(kwargs)

        def repo_info(self, **kwargs):
            return SimpleNamespace(private=True)

        def upload_file(self, **kwargs):
            calls["uploaded"].append(kwargs)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeHfApi))
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    url = publish_run(run_dir, "owner/private-voice")

    assert url == "https://huggingface.co/owner/private-voice"
    assert calls["created"][0]["private"] is True
    assert calls["created"][0]["exist_ok"] is True
    assert {Path(call["path_or_fileobj"]).name for call in calls["uploaded"]} == {
        "voice.pth", "added_IVF1.index"
    }
    assert all("source.wav" not in call["path_in_repo"] for call in calls["uploaded"])


def test_publish_refuses_existing_public_repository(tmp_path, monkeypatch):
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "voice.pth").write_bytes(b"weights")

    class PublicRepoApi:
        def __init__(self, token):
            pass

        def create_repo(self, **kwargs):
            pass

        def repo_info(self, **kwargs):
            return SimpleNamespace(private=False)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=PublicRepoApi))
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    with pytest.raises(RuntimeError, match="is public"):
        publish_run(tmp_path, "owner/public-repo")


def test_publish_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="Set HF_TOKEN"):
        publish_run(tmp_path, "owner/repo")


def test_clean_start_clears_old_artifacts_but_keeps_prepared_data(project_env):
    config = TrainingRunConfig(
        experiment_name="fresh", dataset_path=project_env / "dataset",
        clean_start=True, resume=True, resume_from="G_1.pth",
    )
    layout = config.layout()
    for key in ("checkpoints", "exports"):
        layout[key].mkdir(parents=True, exist_ok=True)
        (layout[key] / "old.pth").write_bytes(b"old")
    for key in ("progress", "events", "train_log"):
        layout[key].parent.mkdir(parents=True, exist_ok=True)
        layout[key].write_text("old")
    prepared = config.run_dir / "3_feature768" / "feature.npy"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_bytes(b"keep")

    _clear_previous_training(config)

    assert not list(layout["checkpoints"].glob("*.pth"))
    assert not list(layout["exports"].glob("*.pth"))
    assert all(not layout[key].exists() for key in ("progress", "events", "train_log"))
    assert prepared.is_file()
    assert config.resume is False and config.resume_from is None


def test_hf_repo_id_validation(project_env):
    config = TrainingRunConfig(
        experiment_name="valid", dataset_path=project_env / "dataset",
        hf_repo_id="user/model",
    )
    assert config.hf_repo_id == "user/model"
    with pytest.raises(ValueError, match="owner/repository"):
        TrainingRunConfig(
            experiment_name="invalid", dataset_path=project_env / "dataset",
            hf_repo_id="not-a-repo-id",
        )


def test_trainer_clean_start_ignores_explicit_checkpoint(project_env):
    import torch

    from rvc_platform.training.trainer import Trainer

    dataset = project_env / "dataset"
    dataset.mkdir()
    config = TrainingRunConfig(
        experiment_name="trainer-fresh", dataset_path=dataset,
        clean_start=True, resume=True, resume_from="G_old.pth",
    )
    checkpoint_dir = config.layout()["checkpoints"]
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "G_old.pth").write_bytes(b"old checkpoint")
    trainer = Trainer(config, config.to_hparams(), "cpu", torch.float32)

    resumed = trainer._restore(None, None, None, None, None, None, None, None)

    assert resumed is False
    assert trainer.epoch == 0 and trainer.global_step == 0
    assert config.resume is False and config.resume_from is None
