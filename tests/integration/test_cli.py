"""CLI surface: help, colab group, checkpoints, logs fallback (CliRunner)."""
from __future__ import annotations

import json

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
    assert result.exit_code == 3
    assert (run_dir / "run_config.json").is_file()
    persisted = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert persisted.get("resume_from") == "G_5.pth"


def test_model_list_empty_message(project_env):
    result = runner.invoke(app, ["model", "list"])
    assert result.exit_code == 0
    assert "no models yet" in result.output


# --------------------------------------------------------------- v1.3.1 ----
def _make_prepared_run(project_env, exp="prepared_exp", sr="40k", f0="rmvpe"):
    """A run with a prepared filelist + saved config (no checkpoints needed)."""
    from rvc_platform.training.run_config import TrainingRunConfig

    dataset = project_env / "ds_prep"
    dataset.mkdir(exist_ok=True)
    (dataset / "a.wav").write_bytes(b"RIFF")
    run_dir = project_env / "runs" / exp
    run_dir.mkdir(parents=True)
    (run_dir / "filelist.txt").write_text(
        f"{run_dir}/0_gt_wavs/a.wav|{run_dir}/3_feature768/a.npy|0\n",
        encoding="utf-8")
    config = TrainingRunConfig(
        experiment_name=exp, dataset_path=str(dataset), sample_rate=sr,
        f0_method=f0, epochs=100, batch_size=4, learning_rate=1e-4,
        preload_base_models=False,
    )
    (run_dir / "run_config.json").write_text(config.model_dump_json(),
                                            encoding="utf-8")
    return run_dir


def test_train_start_applies_knobs_to_existing_run(project_env):
    """v1.3.1: without --dataset, epochs/batch/lr/save-every apply to a
    prepared run instead of being silently ignored."""
    run_dir = _make_prepared_run(project_env)
    result = runner.invoke(app, [
        "train", "start", "--experiment", "prepared_exp",
        "--epochs", "250", "--batch", "8", "--lr", "2e-4",
        "--save-every", "5",
    ])
    # worker will fail (no real data) — but the config on disk must show
    # the applied knobs and the untouched prepared pipeline
    persisted = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert persisted["epochs"] == 250
    assert persisted["batch_size"] == 8
    assert persisted["learning_rate"] == 2e-4
    assert persisted["save_every_epoch"] == 5
    assert result.exit_code in (0, 3)  # worker failure on fake data is fine


def test_train_start_rejects_sr_mismatch_on_prepared_run(project_env):
    """SR/F0 are locked to the prepared run — changing them must fail fast
    instead of silently invalidating the extracted features."""
    _make_prepared_run(project_env, sr="40k")
    result = runner.invoke(app, [
        "train", "start", "--experiment", "prepared_exp", "--sr", "48k",
    ])
    assert result.exit_code != 0
    assert "40k" in result.output


def test_train_start_dataset_only_moves_pointer_on_existing_run(project_env):
    """v1.3.1: --dataset on an existing run only moves the pointer; it must
    not wipe the saved prepare settings (denoise/loudness/speech filter)."""
    from rvc_platform.training.run_config import TrainingRunConfig

    run_dir = _make_prepared_run(project_env)
    # overwrite the saved config with CUSTOM prepare settings
    saved = TrainingRunConfig.load(run_dir)
    saved.denoise, saved.loudness, saved.speech_filter_ratio = True, False, 0.35
    (run_dir / "run_config.json").write_text(saved.model_dump_json(), encoding="utf-8")
    other = project_env / "ds_other"
    other.mkdir(exist_ok=True)
    (other / "b.wav").write_bytes(b"RIFF")
    result = runner.invoke(app, [
        "train", "start", "--experiment", "prepared_exp",
        "--dataset", str(other), "--epochs", "120",
    ])
    persisted = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    # custom prepare settings survive; dataset_path follows the new pointer
    assert persisted["denoise"] is True
    assert persisted["loudness"] is False
    assert persisted["speech_filter_ratio"] == 0.35
    assert persisted["epochs"] == 120
    assert "ds_other" in persisted["dataset_path"]
    assert result.exit_code in (0, 3)


def test_train_start_rejects_f0_mismatch_on_prepared_run(project_env):
    _make_prepared_run(project_env, f0="rmvpe")
    result = runner.invoke(app, [
        "train", "start", "--experiment", "prepared_exp",
        "--f0-method", "fcpe",
    ])
    assert result.exit_code != 0
    assert "rmvpe" in result.output
