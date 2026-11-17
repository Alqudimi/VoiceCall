"""Unit tests: colab_runner restore/snapshot against a fake Drive mount."""
from __future__ import annotations

from pathlib import Path

from rvc_platform.colab import colab_runner


class FakeDrive:
    """Redirect /content/drive/MyDrive into a temp directory."""

    def __init__(self, tmp_path: Path) -> None:
        self.base = tmp_path / "MyDrive"
        self.base.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        import rvc_platform.colab.colab_runner as cr

        self._orig = cr.drive_root
        cr.drive_root = lambda drive_dir="RVC_Iraqi": (self.base / drive_dir) if self._mount() else None
        return self

    def _mount(self) -> bool:
        (self.base / "RVC_Iraqi").mkdir(parents=True, exist_ok=True)
        return True

    def __exit__(self, *args):
        import rvc_platform.colab.colab_runner as cr

        cr.drive_root = self._orig


def test_snapshot_pushes_runs_and_models(project_env, tmp_path):
    # local artifacts
    runs = project_env / "runs" / "exp1"
    (runs / "checkpoints").mkdir(parents=True)
    (runs / "checkpoints" / "G_3.pth").write_bytes(b"G")
    (project_env / "models").mkdir(exist_ok=True)
    (project_env / "models" / "voice.pth").write_bytes(b"M")

    with FakeDrive(tmp_path):
        report = colab_runner.snapshot()
    assert report["snapshot"] is True
    drive = tmp_path / "MyDrive" / "RVC_Iraqi"
    assert (drive / "runs" / "exp1" / "checkpoints" / "G_3.pth").read_bytes() == b"G"
    assert (drive / "models" / "voice.pth").read_bytes() == b"M"


def test_restore_pulls_drive_into_session(project_env, tmp_path):
    drive = tmp_path / "MyDrive" / "RVC_Iraqi"
    (drive / "runs" / "exp2" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (drive / "runs" / "exp2" / "checkpoints" / "G_9.pth").write_bytes(b"9")
    (drive / "models").mkdir(parents=True, exist_ok=True)
    (drive / "models" / "iraqi.pth").write_bytes(b"iraqi")

    with FakeDrive(tmp_path):
        report = colab_runner.restore()
    assert report["restored"] is True
    assert (project_env / "runs" / "exp2" / "checkpoints" / "G_9.pth").read_bytes() == b"9"
    assert (project_env / "models" / "iraqi.pth").read_bytes() == b"iraqi"


def test_status_reports_drive_state(project_env, tmp_path):
    drive = tmp_path / "MyDrive" / "RVC_Iraqi"
    (drive / "runs" / "x" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (drive / "runs" / "x" / "checkpoints" / "G_12.pth").write_bytes(b"x")

    with FakeDrive(tmp_path):
        status = colab_runner.status()
    assert status["drive"] is True
    assert any("G_12.pth" in c for c in status["checkpoints"])


def test_no_drive_mounted(project_env):
    import rvc_platform.colab.colab_runner as cr

    original = cr.drive_root
    cr.drive_root = lambda drive_dir="RVC_Iraqi": None
    try:
        assert colab_runner.restore()["restored"] is False
        assert colab_runner.snapshot()["snapshot"] is False
        assert colab_runner.status()["drive"] is False
    finally:
        cr.drive_root = original
