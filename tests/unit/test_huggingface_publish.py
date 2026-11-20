from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from rvc_platform.training.huggingface import publish_run
from rvc_platform.training.run_config import TrainingRunConfig
from rvc_platform.training.worker import _clear_previous_training


class FakeCommitOperationAdd:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


def test_publish_private_models_only(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    exports = run_dir / "exports"
    exports.mkdir(parents=True)
    (exports / "voice.pth").write_bytes(b"weights")
    (run_dir / "added_IVF1.index").write_bytes(b"index")
    (run_dir / "source.wav").write_bytes(b"private audio")
    (run_dir / "filelist.txt").write_text("private dataset paths")
    calls = {"created": [], "commits": []}

    class FakeHfApi:
        def __init__(self, token):
            assert token == "hf-secret"

        def create_repo(self, **kwargs):
            calls["created"].append(kwargs)

        def repo_info(self, **kwargs):
            return SimpleNamespace(private=True)

        def create_commit(self, **kwargs):
            calls["commits"].append(kwargs)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(HfApi=FakeHfApi,
                                        CommitOperationAdd=FakeCommitOperationAdd))
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    url = publish_run(run_dir, "owner/private-voice")

    assert url == "https://huggingface.co/owner/private-voice"
    assert calls["created"][0]["private"] is True
    assert calls["created"][0]["exist_ok"] is True
    # one atomic commit carrying model + index together
    assert len(calls["commits"]) == 1
    (commit,) = calls["commits"]
    paths_in_repo = {op.path_in_repo for op in commit["operations"]}
    assert paths_in_repo == {"models/voice.pth", "indexes/added_IVF1.index"}
    assert all("source.wav" not in path for path in paths_in_repo)
    assert commit["repo_type"] == "model"


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

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(HfApi=PublicRepoApi,
                                        CommitOperationAdd=FakeCommitOperationAdd))
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    with pytest.raises(RuntimeError, match="is public"):
        publish_run(tmp_path, "owner/public-repo")


def test_publish_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "voice.pth").write_bytes(b"weights")
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
    # best-model + validation-split artifacts must be cleared too
    (config.run_dir / "checkpoints" / "best.json").write_text("{}")
    (config.run_dir / "filelist_train.txt").write_text("a")
    (config.run_dir / "filelist_val.txt").write_text("b")
    (config.run_dir / "validation_split.json").write_text("{}")

    _clear_previous_training(config)

    assert not list(layout["checkpoints"].glob("*.pth"))
    assert not (config.run_dir / "checkpoints" / "best.json").exists()
    assert not list(layout["exports"].glob("*.pth"))
    assert all(not layout[key].exists() for key in ("progress", "events", "train_log"))
    assert not (config.run_dir / "filelist_train.txt").exists()
    assert not (config.run_dir / "filelist_val.txt").exists()
    assert not (config.run_dir / "validation_split.json").exists()
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


def test_run_config_best_model_fields(project_env):
    config = TrainingRunConfig(
        experiment_name="bestcfg", dataset_path=project_env / "dataset",
    )
    # smart defaults: tracking on, with a small held-out split for the
    # researched composite best-model metric (UPGRADE_REPORT_AR.md §2.4)
    assert config.save_best is True
    assert config.val_split == 0.1
    config2 = TrainingRunConfig(
        experiment_name="bestcfg2", dataset_path=project_env / "dataset",
        save_best=False, val_split=0.25,
    )
    assert config2.save_best is False and config2.val_split == 0.25
    with pytest.raises(ValueError):
        TrainingRunConfig(
            experiment_name="badcfg", dataset_path=project_env / "dataset",
            val_split=0.9,
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

    resumed, _extras = trainer._restore(None, None, None, None, None, None, None, None)

    assert resumed is False
    assert trainer.epoch == 0 and trainer.global_step == 0
    assert config.resume is False and config.resume_from is None
