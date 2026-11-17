"""CLI surface: help, colab group, checkpoints, logs fallback (CliRunner)."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from rvc_platform.cli.main import app

runner = CliRunner()


def test_help_lists_all_groups():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ("install", "env", "train", "model", "colab", "web", "status"):
        assert group in result.output


def test_colab_group_help():
    result = runner.invoke(app, ["colab", "--help"])
    assert result.exit_code == 0
    for cmd in ("restore", "snapshot", "status", "env"):
        assert cmd in result.output


def test_colab_status_without_drive(project_env):
    import rvc_platform.colab.colab_runner as cr

    original = cr.drive_root
    cr.drive_root = lambda drive_dir="RVC_Iraqi": None
    try:
        result = runner.invoke(app, ["colab", "status"])
        assert result.exit_code == 0
        assert "drive" in result.output and "false" in result.output.lower()
    finally:
        cr.drive_root = original


def test_train_start_requires_dataset_or_existing_run(project_env):
    result = runner.invoke(app, ["train", "start"])
    assert result.exit_code == 2
    assert "no existing run" in result.output


def test_train_start_unknown_dataset_fails_in_worker(project_env):
    result = runner.invoke(app, [
        "train", "start", "--dataset", str(project_env / "nope"),
        "--resume-from", "G_1.pth",
    ])
    # dataset folder missing -> worker validation exit code 3
    assert result.exit_code in (2, 3)


def test_logs_reports_missing_file(project_env):
    result = runner.invoke(app, ["logs"])
    assert result.exit_code == 0
    assert "platform.log" in result.output


def test_train_checkpoints_empty_run(project_env):
    (project_env / "runs" / "empty_exp").mkdir(parents=True)
    result = runner.invoke(app, ["train", "checkpoints", "empty_exp"])
    assert result.exit_code == 0


def test_train_resume_from_persisted_in_run_config(project_env, tmp_path):
    from rvc_platform.training.run_config import TrainingRunConfig

    dataset = project_env / "ds_cli"
    dataset.mkdir()
    import soundfile as sf
    import numpy as np

    sf.write(str(dataset / "a.wav"),
             (0.2 * np.sin(np.arange(16_000) / 16_000 * 2 * 3.14159 * 200)).astype(np.float32),
             16_000, subtype="PCM_16")
    config = TrainingRunConfig(
        experiment_name="cli_exp", dataset_path=str(dataset),
        epochs=1, batch_size=1, preload_base_models=False,
    )
    run_dir = project_env / "runs" / "cli_exp"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(config.model_dump_json(), encoding="utf-8")

    result = runner.invoke(app, [
        "train", "start", "--experiment", "cli_exp", "--resume-from", "G_5.pth",
    ])
    # G_5.pth does not exist -> the worker fails with exit code 3, but the
    # run_config on disk must already carry the explicit resume point.
    assert (run_dir / "run_config.json").is_file()
    persisted = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert persisted.get("resume_from") == "G_5.pth"


def test_model_list_empty_message(project_env):
    result = runner.invoke(app, ["model", "list"])
    assert result.exit_code == 0
    assert "no models yet" in result.output
