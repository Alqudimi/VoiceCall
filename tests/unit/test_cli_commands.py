"""Unit tests — CLI batch 6: cli/main.py, every command's real paths.

Strategy: typer.testing.CliRunner invokes each command. Commands that spawn
subprocesses / block (env sync, web, train start) are either asserted on
their argument-validation branches or monkeypatched at the seam the command
itself uses (os.system, subprocess.Popen, worker_main).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from rvc_platform.cli.main import app

runner = CliRunner()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _tiny_dataset(root: Path, n: int = 2, seconds: float = 2.0) -> Path:
    d = root / "ds"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    t = np.arange(int(32_000 * seconds)) / 32_000
    for i in range(n):
        wav = (0.4 * np.sin(2 * np.pi * 150 * t)
               + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
        sf.write(str(d / f"spk{i}.wav"), wav, 32_000, subtype="PCM_16")
    return d


def _prepared_run(project_env, tmp_path, name="exp_cli", sr="32k") -> Path:
    """A minimal run directory with a saved run_config (as prepare leaves)."""
    from rvc_platform.infrastructure.paths import get_paths
    from rvc_platform.training.run_config import TrainingRunConfig

    ds = _tiny_dataset(tmp_path)
    cfg = TrainingRunConfig(experiment_name=name, dataset_path=ds,
                            sample_rate=sr, version="v1", f0_method="pm")
    cfg.save()
    return get_paths().runs / name


# --------------------------------------------------------------------------- #
# install sub-app
# --------------------------------------------------------------------------- #
class TestInstallCommands:
    def test_doctor_reports_environment(self, project_env):
        result = runner.invoke(app, ["install", "doctor"])
        assert result.exit_code == 0
        assert "torch" in result.output

    def test_doctor_fix_downloads_when_missing(self, project_env, monkeypatch):
        from rvc_platform.application import downloads as dl

        # force "missing" so the --fix branch runs; stub the downloader
        monkeypatch.setattr(dl, "asset_status",
                            lambda: {"_summary": {"all_present": False}})
        monkeypatch.setattr(dl, "download_assets", lambda progress=None: None)
        result = runner.invoke(app, ["install", "doctor", "--fix"])
        assert result.exit_code == 0
        assert "assets downloaded" in result.output

    def test_doctor_missing_without_fix_warns(self, project_env, monkeypatch):
        from rvc_platform.application import downloads as dl

        monkeypatch.setattr(dl, "asset_status",
                            lambda: {"_summary": {"all_present": False}})
        result = runner.invoke(app, ["install", "doctor"])
        assert result.exit_code == 0
        assert "assets missing" in result.output

    def test_check_imports(self, project_env):
        result = runner.invoke(app, ["install", "check"])
        assert result.exit_code == 0
        assert "cuda available" in result.output
        assert "faiss" in result.output


# --------------------------------------------------------------------------- #
# env sub-app (uv wrappers — argument validation + os.system seam)
# --------------------------------------------------------------------------- #
class TestEnvCommands:
    def test_gpu_rejects_unknown_variant(self, project_env):
        result = runner.invoke(app, ["env", "gpu", "cu121"])
        assert result.exit_code != 0
        assert "supported" in result.output

    def test_lock_and_update_call_os_system(self, project_env, monkeypatch):
        calls = []
        monkeypatch.setattr("rvc_platform.cli.main.os.system",
                            lambda cmd: calls.append(cmd) or 0)
        assert runner.invoke(app, ["env", "lock"]).exit_code == 0
        assert runner.invoke(app, ["env", "update"]).exit_code == 0
        assert any("uv lock" in c for c in calls)
        assert any("uv sync --upgrade" in c for c in calls)

    def test_sync_no_groups(self, project_env, monkeypatch):
        calls = []
        monkeypatch.setattr("rvc_platform.cli.main.os.system",
                            lambda cmd: calls.append(cmd) or 0)
        assert runner.invoke(app, ["env", "sync", "--no-groups"]).exit_code == 0
        assert calls and "--all-groups" not in calls[0]

    def test_verify_runs_sync_then_check(self, project_env, monkeypatch):
        monkeypatch.setattr("rvc_platform.cli.main.os.system", lambda cmd: 0)
        result = runner.invoke(app, ["env", "verify"])
        assert result.exit_code == 0
        assert "cuda available" in result.output

    def test_info_prints_paths(self, project_env):
        result = runner.invoke(app, ["env", "info"])
        assert result.exit_code == 0
        assert "project root" in result.output
        assert "assets" in result.output


# --------------------------------------------------------------------------- #
# clean
# --------------------------------------------------------------------------- #
class TestClean:
    def test_clean_removes_caches_only(self, project_env):
        from rvc_platform.infrastructure.paths import get_paths

        p = get_paths()
        p.cache.mkdir(parents=True, exist_ok=True)
        p.temp.mkdir(parents=True, exist_ok=True)
        (p.cache / "x.bin").write_bytes(b"1")
        runs_marker = p.runs / "keepme"
        runs_marker.mkdir(parents=True, exist_ok=True)
        result = runner.invoke(app, ["clean"])
        assert result.exit_code == 0
        assert not p.cache.exists() and not p.temp.exists()
        assert runs_marker.exists()          # runs survive plain clean

    def test_clean_all_removes_runs(self, project_env):
        from rvc_platform.infrastructure.paths import get_paths

        p = get_paths()
        p.runs.mkdir(parents=True, exist_ok=True)
        p.data.mkdir(parents=True, exist_ok=True)
        (p.data / "uploads").mkdir(parents=True, exist_ok=True)
        result = runner.invoke(app, ["clean", "--all"])
        assert result.exit_code == 0
        assert not p.runs.exists()
        assert not (p.data / "uploads").exists()


# --------------------------------------------------------------------------- #
# data / dataset commands
# --------------------------------------------------------------------------- #
class TestDataCommands:
    def test_data_check_prints_report(self, project_env, tmp_path):
        ds = _tiny_dataset(tmp_path)
        result = runner.invoke(app, ["data-check", str(ds)])
        assert result.exit_code == 0
        assert "total_files" in result.output

    def test_data_check_warns_below_recommendation(self, project_env, tmp_path):
        ds = _tiny_dataset(tmp_path, n=1, seconds=0.5)   # ~0.5 s of audio
        result = runner.invoke(app, ["data-check", str(ds)])
        assert result.exit_code == 0
        assert "quality will suffer" in result.output

    def test_prepare_audio_dataset_imports(self, project_env, tmp_path):
        src = _tiny_dataset(tmp_path / "src_root")
        result = runner.invoke(
            app, ["prepare-audio-dataset", str(src), "voice1",
                  "--sr", "32k"])
        assert result.exit_code == 0, result.output
        assert "dataset ready" in result.output

    def test_prepare_audio_dataset_bad_sr_fails(self, project_env, tmp_path):
        src = _tiny_dataset(tmp_path / "src2")
        result = runner.invoke(
            app, ["prepare-audio-dataset", str(src), "voice2",
                  "--sr", "16k"])
        assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# train sub-app
# --------------------------------------------------------------------------- #
class TestTrainCommands:
    def test_train_start_requires_dataset_first_time(self, project_env):
        result = runner.invoke(app, ["train", "start"])
        assert result.exit_code == 2
        assert "no existing run" in result.output

    def _stub_worker(self, monkeypatch, code=0):
        from rvc_platform.cli import main as cli_main

        def fake_worker_main(argv):
            assert argv[0] == "--config"
            return code

        monkeypatch.setattr(cli_main, "worker_main", fake_worker_main,
                            raising=False)
        # the command imports it INSIDE the function from the module —
        # patch the source module too
        from rvc_platform.training import worker
        monkeypatch.setattr(worker, "main", fake_worker_main)

    def test_train_start_new_run_invokes_worker(self, project_env, tmp_path,
                                                monkeypatch):
        self._stub_worker(monkeypatch)
        ds = _tiny_dataset(tmp_path)
        result = runner.invoke(
            app, ["train", "start", "--dataset", str(ds),
                  "-e", "newexp", "--sr", "32k", "--epochs", "2"])
        assert result.exit_code == 0, result.output
        # the run dir + config were persisted before the worker ran
        assert (project_env / "runs" / "newexp" / "run_config.json").is_file()

    def test_train_start_sr_mismatch_rejected(self, project_env, tmp_path,
                                              monkeypatch):
        _prepared_run(project_env, tmp_path, name="srlock", sr="32k")
        ds2 = _tiny_dataset(tmp_path / "other")
        result = runner.invoke(
            app, ["train", "start", "--dataset", str(ds2),
                  "-e", "srlock", "--sr", "40k"])
        assert result.exit_code != 0
        assert "does not match the prepared run" in result.output

    def test_train_start_f0_mismatch_rejected(self, project_env, tmp_path):
        _prepared_run(project_env, tmp_path, name="f0lock", sr="32k")
        # prepared run uses pm -> asking for rmvpe must be rejected
        result = runner.invoke(
            app, ["train", "start", "-e", "f0lock", "--f0-method", "rmvpe"])
        assert result.exit_code != 0
        assert "f0-method" in result.output

    def test_train_start_continues_prepared_run(self, project_env, tmp_path,
                                                monkeypatch):
        self._stub_worker(monkeypatch)
        _prepared_run(project_env, tmp_path, name="cont", sr="32k")
        result = runner.invoke(
            app, ["train", "start", "-e", "cont", "--epochs", "5"])
        assert result.exit_code == 0, result.output
        assert "continuing prepared run" in result.output

    def test_train_start_resume_from_conflicts_clean_start(
            self, project_env, tmp_path):
        _prepared_run(project_env, tmp_path, name="rconf", sr="32k")
        result = runner.invoke(
            app, ["train", "start", "-e", "rconf", "--clean-start",
                  "--resume-from", "G_100.pth"])
        assert result.exit_code != 0
        assert "cannot be combined" in result.output

    def test_train_start_epoch_guidance_hint(self, project_env, tmp_path,
                                             monkeypatch):
        self._stub_worker(monkeypatch)
        run = _prepared_run(project_env, tmp_path, name="guide", sr="32k")
        (run / "dataset_report.json").write_text(json.dumps({
            "mean_snr_db": 12.0,
            "epoch_recommendation": {"min": 50, "max": 150,
                                     "quality": "fair", "reason": "test"},
        }), encoding="utf-8")
        result = runner.invoke(
            app, ["train", "start", "-e", "guide", "--epochs", "400"])
        assert result.exit_code == 0, result.output
        assert "official guidance" in result.output

    def test_train_stop_without_process(self, project_env):
        result = runner.invoke(app, ["train", "stop"])
        assert result.exit_code == 0
        assert "no trainer process registered" in result.output

    def test_train_checkpoints_lists(self, project_env, tmp_path,
                                     monkeypatch):

        run = _prepared_run(project_env, tmp_path, name="ckpts", sr="32k")
        ck = run / "checkpoints"
        ck.mkdir(parents=True)
        (ck / "G_100.pth").write_bytes(b"g" * 16)
        (ck / "D_100.pth").write_bytes(b"d" * 16)
        # rich truncates long paths in narrow terminals — force full width
        monkeypatch.setenv("COLUMNS", "300")
        result = runner.invoke(app, ["train", "checkpoints", "ckpts"])
        assert result.exit_code == 0
        assert "G_100" in result.output.replace("…", "")

    def test_train_export_roundtrip(self, project_env, tmp_path):
        """export with no checkpoints -> clear error path (service raises)."""
        _prepared_run(project_env, tmp_path, name="expexp", sr="32k")
        result = runner.invoke(app, ["train", "export", "expexp"])
        # nothing to export -> non-zero exit and an error surfaced
        assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# colab sub-app (drive absent -> clean refusal paths)
# --------------------------------------------------------------------------- #
class TestColabCommands:
    @pytest.fixture(autouse=True)
    def _no_drive(self, monkeypatch):
        from rvc_platform.colab import colab_runner as cr

        monkeypatch.setattr(cr, "drive_root", lambda d="RVC_Iraqi": None)

    def test_restore_without_drive(self, project_env):
        result = runner.invoke(app, ["colab", "restore"])
        assert result.exit_code == 0
        assert "not mounted" in result.output

    def test_snapshot_without_drive(self, project_env):
        result = runner.invoke(app, ["colab", "snapshot"])
        assert result.exit_code == 0
        assert "not mounted" in result.output

    def test_status_without_drive(self, project_env):
        result = runner.invoke(app, ["colab", "status"])
        assert result.exit_code == 0
        assert "mount Drive first" in result.output

    def test_env_wiring(self, project_env):
        result = runner.invoke(app, ["colab", "env"])
        assert result.exit_code == 0
        assert "RVC_DATA_DIR" in result.output


# --------------------------------------------------------------------------- #
# model sub-app
# --------------------------------------------------------------------------- #
class TestModelCommands:
    def test_model_list_empty(self, project_env):
        result = runner.invoke(app, ["model", "list"])
        assert result.exit_code == 0

    def test_model_info_unknown_fails(self, project_env):
        result = runner.invoke(app, ["model", "info", "ghost"])
        assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# status / logs / web-status helpers (server offline branches)
# --------------------------------------------------------------------------- #
class TestStatusCommands:
    def test_status_offline_snapshot(self, project_env, monkeypatch):
        # port closed -> offline branch with detect_system summary
        result = runner.invoke(app, ["status", "--port", "59999"])
        assert result.exit_code == 0
        assert "web server offline" in result.output

    def test_web_status_offline(self, project_env):
        result = runner.invoke(app, ["web-status", "--port", "59998"])
        assert result.exit_code == 0
        assert "not running" in result.output

    def test_logs_missing_file(self, project_env):
        result = runner.invoke(app, ["logs"])
        assert result.exit_code == 0
        assert "no platform.log yet" in result.output

    def test_logs_tails_file(self, project_env):
        from rvc_platform.infrastructure.paths import get_paths

        get_paths().logs.mkdir(parents=True, exist_ok=True)
        (get_paths().logs / "platform.log").write_text(
            "line1\nline2\nline3\n", encoding="utf-8")
        result = runner.invoke(app, ["logs", "--lines", "2"])
        assert result.exit_code == 0
        assert "line2" in result.output and "line1" not in result.output


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
class TestCliHelpers:
    def test_web_pid_missing(self, project_env):
        from rvc_platform.cli.main import _web_pid

        assert _web_pid() is None

    def test_web_pid_reads_file(self, project_env):
        from rvc_platform.cli.main import _PID_FILE, _web_pid

        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text("424242")
        assert _web_pid() == 424242
        _PID_FILE.write_text("not-a-pid")
        assert _web_pid() is None

    def test_port_open_false_on_free_port(self):
        from rvc_platform.cli.main import _port_open

        # 1 is privileged/closed in this sandbox
        assert _port_open(1) is False

    def test_port_open_true_on_listener(self):
        import socket

        from rvc_platform.cli.main import _port_open

        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            assert _port_open(port) is True
        finally:
            srv.close()

    def test_api_none_when_server_down(self):
        from rvc_platform.cli.main import _api

        assert _api(59997) is None

    def test_api_returns_json(self):
        import socket
        import threading

        from rvc_platform.cli.main import _api

        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]

        def serve():
            conn, _ = srv.accept()
            conn.recv(4096)
            body = json.dumps({"os": "Linux", "arch": "x86_64"}).encode()
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: "
                         b"application/json\r\nContent-Length: "
                         + str(len(body)).encode() + b"\r\n\r\n" + body)
            conn.close()

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        try:
            assert _api(port) == {"os": "Linux", "arch": "x86_64"}
        finally:
            srv.close()
            t.join(timeout=2)

    def test_print_status_renders_table(self, project_env, capsys):
        from rvc_platform.cli.main import _print_status

        status = {"os": "Linux", "arch": "x86_64", "python": "3.12",
                  "device": "cpu", "is_half": False, "gpu": {},
                  "engine": "ready", "engine_message": "",
                  "loaded_model": None, "training": {}, "tunnel": None,
                  "realtime_sessions": 0}
        _print_status(status)
        out = capsys.readouterr().out
        assert "component" in out and "Device" in out

    def test_read_pid_missing_and_dead(self, project_env):
        from rvc_platform.cli.main import _read_pid

        assert _read_pid("trainer") is None
        from rvc_platform.infrastructure.paths import get_paths

        f = get_paths().temp / "trainer.pid"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("99999999")           # a pid that cannot exist
        assert _read_pid("trainer") is None
        f.write_text("garbage")
        assert _read_pid("trainer") is None

    def test_terminate_group_missing_pid(self):
        from rvc_platform.cli.main import _terminate_group

        _terminate_group(99999999)         # no raise on dead pid

    def test_stop_tunnel_without_pid_file(self, project_env):
        from rvc_platform.cli.main import _stop_tunnel

        assert _stop_tunnel() is False
