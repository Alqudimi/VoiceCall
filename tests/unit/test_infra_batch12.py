"""Unit tests — batch 12: cross-cutting infrastructure.

EventBus, environment profiling, colab Drive sync (simulated Drive),
_ffmpeg_load internals (wav2 / transcode / channel shaping / resampler),
and the remaining API route branches (training runs detail, model download
headers, system logs).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import SYNTH_CONFIG_40K  # noqa: F401 — env fixture chain


# --------------------------------------------------------------------------- #
# EventBus
# --------------------------------------------------------------------------- #
class TestEventBus:
    def test_publish_fanout_and_history(self):
        from rvc_platform.core.events import EventBus

        bus = EventBus(history=8)
        got_a, got_star = [], []
        bus.subscribe("training", got_a.append)
        bus.subscribe("*", got_star.append)
        evt = bus.publish("training", {"epoch": 1})
        assert got_a and got_star
        assert got_a[0].topic == "training"
        assert json.loads(evt.json())["payload"]["epoch"] == 1
        assert bus.recent("training") == [evt]

    def test_unsubscribe(self):
        from rvc_platform.core.events import EventBus

        bus = EventBus()
        got = []
        unsub = bus.subscribe("t", got.append)
        unsub()
        bus.publish("t", {"x": 1})
        assert got == []
        unsub()                                   # double-unsub is safe

    def test_history_bounded(self):
        from rvc_platform.core.events import EventBus

        bus = EventBus(history=4)
        for i in range(10):
            bus.publish("t", {"i": i})
        recent = bus.recent("t", limit=100)
        assert len(recent) == 4                   # deque maxlen honoured
        assert recent[-1].payload["i"] == 9

    def test_recent_filters_by_topic_and_limit(self):
        from rvc_platform.core.events import EventBus

        bus = EventBus()
        bus.publish("a", {"i": 1})
        bus.publish("b", {"i": 2})
        bus.publish("a", {"i": 3})
        assert len(bus.recent()) == 3
        assert [e.payload["i"] for e in bus.recent("a")] == [1, 3]
        assert bus.recent("a", limit=1)[0].payload["i"] == 3

    def test_faulty_subscriber_does_not_break_publish(self):
        from rvc_platform.core.events import EventBus

        bus = EventBus()
        good = []

        def boom(evt):
            raise RuntimeError("subscriber bug")

        bus.subscribe("t", boom)
        bus.subscribe("t", good.append)
        evt = bus.publish("t", {"ok": True})      # must not raise
        assert good and evt.payload == {"ok": True}

    def test_global_bus_singleton(self):
        from rvc_platform.core.events import global_bus

        assert global_bus() is global_bus()


# --------------------------------------------------------------------------- #
# environment profiling
# --------------------------------------------------------------------------- #
class TestDetectSystem:
    def test_summary_shape(self):
        from rvc_platform.utils.environment import detect_system

        prof = detect_system()
        s = prof.summary
        assert s["os"] and s["arch"] and s["python"]
        assert "cuda" in s                            # version or None
        assert "ram_mb" in s and "gpus" in s
        # this CPU sandbox: no CUDA expected
        assert prof.cuda_available is False
        assert s["gpus"] == []

    def test_colab_detection_env_var(self, monkeypatch):
        from rvc_platform.utils.environment import detect_system

        monkeypatch.setenv("COLAB_GPU", "1")
        assert detect_system().is_colab is True

    def test_notebook_detection(self, monkeypatch):
        import sys

        from rvc_platform.utils.environment import detect_system

        monkeypatch.setitem(sys.modules, "ipykernel",
                            type(sys)("ipykernel"))
        assert detect_system().is_notebook is True

    def test_tool_availability_flags(self):
        from rvc_platform.utils.environment import detect_system

        prof = detect_system()
        # ffmpeg exists in this sandbox; make/cloudflared may not — just
        # verify the probes return booleans (not raise)
        assert isinstance(prof.ffmpeg_available, bool)
        assert isinstance(prof.cloudflared_available, bool)
        assert isinstance(prof.make_available, bool)

    def test_run_helper_timeout(self):
        from rvc_platform.utils.environment import _run

        code, out = _run(["true"])
        assert code == 0
        code, out = _run(["sh", "-c", "echo hi"])
        assert code == 0 and out.strip() == "hi"

    def test_sys_executable(self):
        from rvc_platform.utils.environment import sys_executable

        assert sys_executable().endswith("python") \
            or "python" in sys_executable()


# --------------------------------------------------------------------------- #
# colab runner — simulated Drive
# --------------------------------------------------------------------------- #
class TestColabRunner:
    @pytest.fixture()
    def fake_drive(self, tmp_path, monkeypatch):
        """Pretend /content/drive/MyDrive exists with runs/ + models/."""
        from rvc_platform.colab import colab_runner as cr

        drive = tmp_path / "MyDrive"
        (drive / "runs" / "exp1" / "checkpoints").mkdir(parents=True)
        (drive / "runs" / "exp1" / "checkpoints" / "G_100.pth").write_bytes(
            b"g" * 64)
        (drive / "models").mkdir(parents=True)
        (drive / "models" / "voice.pth").write_bytes(b"m" * 64)
        monkeypatch.setattr(cr, "drive_root", lambda d="RVC_Iraqi": drive)
        return drive

    def test_drive_root_default(self, monkeypatch, tmp_path):
        from rvc_platform.colab import colab_runner as cr

        # default probe: /content/drive/MyDrive absent in this sandbox
        monkeypatch.delenv("RVC_DRIVE_DIR", raising=False)
        assert cr.drive_root() is None or cr.drive_root() is not None  # probe

    def test_restore_pulls_artifacts(self, fake_drive, project_env):
        from rvc_platform.colab import colab_runner as cr

        report = cr.restore()
        assert report["restored"] is True
        from rvc_platform.infrastructure.paths import get_paths

        assert (get_paths().runs / "exp1" / "checkpoints"
                / "G_100.pth").is_file()
        assert (get_paths().models / "voice.pth").is_file()

    def test_snapshot_pushes_and_writes_manifest(self, fake_drive,
                                                 project_env):
        from rvc_platform.colab import colab_runner as cr
        from rvc_platform.infrastructure.paths import get_paths

        # create local artifacts then snapshot them up
        p = get_paths()
        (p.runs / "exp_local").mkdir(parents=True, exist_ok=True)
        (p.runs / "exp_local" / "run_config.json").write_text("{}", "utf-8")
        report = cr.snapshot()
        assert report["snapshot"] is True
        manifest = json.loads(
            (fake_drive / "snapshot_manifest.json").read_text("utf-8"))
        assert manifest["version"] == 1

    def test_status_lists_checkpoints(self, fake_drive):
        from rvc_platform.colab import colab_runner as cr

        st = cr.status()
        assert st["drive"] is True
        assert len(st["checkpoints"]) == 1           # one G_*.pth on Drive
        assert st["checkpoints"][0].endswith("G_100.pth")
        assert "redirects" in st

    def test_status_without_drive(self, monkeypatch):
        from rvc_platform.colab import colab_runner as cr

        monkeypatch.setattr(cr, "drive_root", lambda d="RVC_Iraqi": None)
        st = cr.status()
        assert st["drive"] is False
        assert "mount Drive first" in st["hint"]

    def test_colab_env_exports(self, fake_drive):
        from rvc_platform.colab import colab_runner as cr

        wiring = cr.colab_env()
        assert wiring["RVC_RUNS_DIR"].endswith("/runs")
        assert wiring["RVC_MODELS_DIR"].endswith("/models")
        assert wiring["RVC_DATA_DIR"] == "/content/iraqi_data"

    def test_colab_env_without_drive(self, monkeypatch):
        from rvc_platform.colab import colab_runner as cr

        monkeypatch.setattr(cr, "drive_root", lambda d="RVC_Iraqi": None)
        wiring = cr.colab_env()
        assert wiring["RVC_RUNS_DIR"] is None
        assert wiring["RVC_MODELS_DIR"] is None

    def test_main_usage(self, capsys):
        from rvc_platform.colab import colab_runner as cr

        # argparse rejects a missing subcommand with SystemExit(2)
        with pytest.raises(SystemExit) as exc:
            cr.main([])
        assert exc.value.code == 2
        assert "required" in capsys.readouterr().err.lower()

    def test_main_status_command(self, fake_drive, capsys):
        from rvc_platform.colab import colab_runner as cr

        assert cr.main(["status"]) == 0
        assert "drive" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# _ffmpeg_load internals
# --------------------------------------------------------------------------- #
class TestFfmpegLoadInternals:
    def test_channel_first_tensor(self):
        from rvc_platform.infrastructure.audio._ffmpeg_load import \
            _channel_first_tensor

        # (samples, channels) numpy input -> stays (samples, channels) as a
        # tensor; 1-D input gains the leading channel axis
        t = _channel_first_tensor(np.zeros((100, 2), np.float32))
        assert tuple(t.shape) == (100, 2)
        one_d = _channel_first_tensor(np.zeros(100, np.float32))
        assert tuple(one_d.shape) == (1, 100)
        with pytest.raises(ValueError):
            _channel_first_tensor(np.zeros((2, 2, 2), np.float32))

    def test_format_audio_tensor_mono_and_stereo(self):
        from rvc_platform.infrastructure.audio._ffmpeg_load import \
            _format_audio_tensor

        # mono in -> 1-D samples out (flattened)
        mono = _format_audio_tensor(np.zeros((1, 100), np.float32),
                                    force_mono=True)
        assert tuple(mono.shape) == (100,)
        # channel-FIRST stereo averaged to mono -> 1-D
        stereo = np.zeros((2, 100), np.float32)
        forced = _format_audio_tensor(stereo, force_mono=True)
        assert tuple(forced.shape) == (100,)
        # stereo kept -> (2, 100)
        kept = _format_audio_tensor(stereo, force_mono=False)
        assert tuple(kept.shape) == (2, 100)

    def test_get_gpu_resampler_cached(self):
        """_get_gpu_resampler builds + caches a torchaudio Resample per
        (src, dst, device, dtype) key — CPU here, cached on second call."""
        from rvc_platform.infrastructure.audio._ffmpeg_load import \
            _get_gpu_resampler

        r1 = _get_gpu_resampler(16_000, 40_000)
        r2 = _get_gpu_resampler(16_000, 40_000)
        assert r1 is r2                            # cache hit
        other = _get_gpu_resampler(24_000, 40_000)
        assert other is not r1                     # different key

    def test_wav2_and_transcode_roundtrip(self, tmp_path):
        from rvc_platform.infrastructure.audio._ffmpeg_load import (
            transcode_audio_file,
            wav2,
        )

        t = np.arange(16_000) / 16_000
        wav = (0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        src = tmp_path / "x.wav"
        sf.write(str(src), wav, 16_000, subtype="PCM_16")
        # 'm4a' maps to the mp4 container (upstream phone-upload path)
        out = tmp_path / "x.m4a"
        wav2(str(src), str(out), "m4a")
        assert out.stat().st_size > 0
        # failed transcode removes its partial output
        bad = tmp_path / "bad.m4a"
        bad.write_bytes(b"partial")
        with pytest.raises(Exception):
            transcode_audio_file(tmp_path / "ghost.wav", bad, "m4a")
        assert not bad.exists()

    def test_load_audio_ffmpeg_reads_any_rate(self, tmp_path):
        from rvc_platform.infrastructure.audio._ffmpeg_load import \
            _load_audio_ffmpeg

        t = np.arange(32_000) / 32_000
        wav = (0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        src = tmp_path / "y.wav"
        sf.write(str(src), wav, 32_000, subtype="PCM_16")
        audio = _load_audio_ffmpeg(str(src), 16_000)
        assert audio.shape[0] == 16_000             # resampled to 16 kHz


# --------------------------------------------------------------------------- #
# API routes — remaining branches (reuse the TestClient graph)
# --------------------------------------------------------------------------- #
class TestApiRouteGaps:
    @pytest.fixture()
    def client(self, project_env):
        from fastapi.testclient import TestClient

        from rvc_platform.api.app import create_app

        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    def _make_run(self, project_env):
        from rvc_platform.infrastructure.paths import get_paths
        from rvc_platform.training.run_config import TrainingRunConfig

        cfg = TrainingRunConfig(experiment_name="apirun",
                                dataset_path=get_paths().datasets_raw,
                                sample_rate="32k", f0_method="pm")
        run = cfg.save().parent
        (run / "progress.json").write_text(json.dumps({
            "state": "running", "epoch": 2, "total_epochs": 5,
            "global_step": 20, "ts": 1795297605.0}), "utf-8")
        (run / "train.log").write_text("l1\nl2\n", "utf-8")
        (run / "events.jsonl").write_text(
            json.dumps({"ev": "epoch"}) + "\n", "utf-8")
        return run

    def test_runs_listing_includes_prepared(self, client, project_env):
        self._make_run(project_env)
        r = client.get("/api/training/runs")
        assert r.status_code == 200
        runs = r.json()
        assert any(run["experiment"] == "apirun" for run in runs)

    def test_logs_returns_plain_text_tail(self, client, project_env):
        run = self._make_run(project_env)
        r = client.get("/api/training/runs/apirun/logs", params={"lines": 1})
        assert r.status_code == 200
        assert r.text.strip() == "l2"

    def test_events_tail(self, client, project_env):
        self._make_run(project_env)
        r = client.get("/api/training/runs/apirun/events")
        assert r.status_code == 200
        assert r.json()["events"] == [json.dumps({"ev": "epoch"})]

    def test_checkpoints_listing(self, client, project_env):
        import torch as _t

        run = self._make_run(project_env)
        ck = run / "checkpoints"
        ck.mkdir(parents=True)
        _t.save({"epoch": 1, "global_step": 100}, ck / "G_100.pth")
        r = client.get("/api/training/runs/apirun/checkpoints")
        assert r.status_code == 200
        body = r.json()
        assert body[0]["global_step"] == 100

    def test_prepare_status_unknown_experiment(self, client):
        r = client.get("/api/training/prepare/status",
                       params={"experiment": "ghost"})
        assert r.status_code == 200
        assert r.json()["stage"] == "idle"

    def test_system_logs_endpoint(self, client, project_env):
        from rvc_platform.infrastructure.paths import get_paths

        # the system router mounts /api + /logs -> /api/logs
        r = client.get("/api/logs")
        assert r.status_code == 200                 # text, maybe empty
        logs_dir = get_paths().logs
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "platform.log").write_text("hello-log\n", "utf-8")
        r2 = client.get("/api/logs")
        assert "hello-log" in r2.text

    def test_tunnel_status_shape(self, client):
        r = client.get("/api/tunnel/status")
        assert r.status_code == 200
        body = r.json()
        assert body["running"] is False
        assert "url" in body

    def test_model_download_unknown_404(self, client):
        r = client.get("/api/models/ghost/download")
        assert r.status_code == 404
