"""Unit tests — batch 7: training_service, worker arg-paths, dataset_import,
downloads, model_service.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _dataset(root: Path, n: int = 2, seconds: float = 2.0,
             sr: int = 32_000) -> Path:
    d = root / "ds"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(21)
    t = np.arange(int(sr * seconds)) / sr
    for i in range(n):
        wav = (0.4 * np.sin(2 * np.pi * 150 * t)
               + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
        sf.write(str(d / f"c{i}.wav"), wav, sr, subtype="PCM_16")
    return d


def _service(project_env):
    from rvc_platform.application.training_service import TrainingService
    from rvc_platform.infrastructure.processes.manager import ProcessManager
    from rvc_platform.infrastructure.paths import get_paths

    return TrainingService(ProcessManager(get_paths().logs / "procs"))


# --------------------------------------------------------------------------- #
# TrainingService
# --------------------------------------------------------------------------- #
class TestTrainingServiceConfig:
    def test_make_config_requires_dataset(self, project_env):
        with pytest.raises(Exception):
            _service(project_env).make_config()

    def test_make_config_missing_folder(self, project_env, tmp_path):
        with pytest.raises(Exception):
            _service(project_env).make_config(
                dataset_path=tmp_path / "ghost")

    def test_make_config_defaults_experiment_name(self, project_env, tmp_path):
        ds = _dataset(tmp_path)
        cfg = _service(project_env).make_config(dataset_path=ds)
        assert cfg.experiment_name.startswith("iraqi_run_")


class TestTrainingServiceStatus:
    def test_status_no_runs(self, project_env):
        from rvc_platform.core.entities import JobState

        st = _service(project_env).status()
        assert st.state == JobState.IDLE
        assert "no training runs" in st.message

    def test_status_reads_progress_json(self, project_env, tmp_path):
        from rvc_platform.core.entities import JobState
        from rvc_platform.training.run_config import TrainingRunConfig

        svc = _service(project_env)
        ds = _dataset(tmp_path)
        cfg = TrainingRunConfig(experiment_name="live1", dataset_path=ds)
        run = cfg.save().parent
        (run / "progress.json").write_text(json.dumps({
            "state": "running", "epoch": 3, "total_epochs": 10,
            "global_step": 90, "steps_per_epoch": 30,
            "latest_losses": {"mel": 0.5}, "ts": time.time(),
            "best": {"epoch": 2, "score": 0.4},
        }), encoding="utf-8")
        st = svc.status()
        assert st.epoch == 3 and st.total_epochs == 10
        assert st.global_step == 90
        assert st.best and st.best["epoch"] == 2
        # no live process -> RUNNING payload downgraded to IDLE
        assert st.state == JobState.IDLE
        assert st.message == "live1"

    def test_status_bad_state_falls_back(self, project_env, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        svc = _service(project_env)
        ds = _dataset(tmp_path)
        cfg = TrainingRunConfig(experiment_name="badst", dataset_path=ds)
        run = cfg.save().parent
        (run / "progress.json").write_text(json.dumps({
            "state": "weird-state", "ts": time.time()}), encoding="utf-8")
        st = svc.status()
        assert st.message == "badst"           # tolerated, not crashed

    def test_status_skips_corrupt_progress(self, project_env, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        svc = _service(project_env)
        ds = _dataset(tmp_path)
        cfg = TrainingRunConfig(experiment_name="corrupt", dataset_path=ds)
        (cfg.save().parent / "progress.json").write_text("{not json",
                                                         encoding="utf-8")
        st = svc.status()
        # corrupt file skipped entirely
        assert st.message != "corrupt"


class TestTrainingServiceCheckpointsLogs:
    def test_checkpoints_unknown_run(self, project_env):
        with pytest.raises(Exception):
            _service(project_env).checkpoints("ghost")

    def test_checkpoints_lists_files(self, project_env, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        svc = _service(project_env)
        ds = _dataset(tmp_path)
        cfg = TrainingRunConfig(experiment_name="cksvc", dataset_path=ds)
        run = cfg.save().parent
        (run / "checkpoints").mkdir(parents=True)
        (run / "checkpoints" / "G_50.pth").write_bytes(b"g" * 32)
        ck = svc.checkpoints("cksvc")
        assert len(ck) >= 1 and ck[0]["path"].endswith("G_50.pth")

    def test_logs_missing_and_present(self, project_env, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        svc = _service(project_env)
        ds = _dataset(tmp_path)
        cfg = TrainingRunConfig(experiment_name="logsvc", dataset_path=ds)
        run = cfg.save().parent
        assert svc.logs("logsvc") == []
        (run / "train.log").write_text("a\nb\nc\n", encoding="utf-8")
        assert svc.logs("logsvc", lines=2) == ["b", "c"]


class TestTrainingServiceResumeExport:
    def test_resume_no_runs(self, project_env):
        with pytest.raises(Exception):
            _service(project_env).resume()

    def test_resume_unknown_run(self, project_env):
        with pytest.raises(Exception):
            _service(project_env).resume("ghost")

    def test_resume_latest_run_starts_process(self, project_env, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        svc = _service(project_env)
        ds = _dataset(tmp_path)
        TrainingRunConfig(experiment_name="rsv", dataset_path=ds).save()
        out = svc.resume()                      # picks the newest run
        assert out["run"] == "rsv"
        assert out["pid"]
        svc.stop()

    def test_resume_with_latest_checkpoint(self, project_env, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        svc = _service(project_env)
        ds = _dataset(tmp_path)
        cfg = TrainingRunConfig(experiment_name="rsvck", dataset_path=ds)
        run = cfg.save().parent
        (run / "checkpoints").mkdir(parents=True)
        (run / "checkpoints" / "G_700.pth").write_bytes(b"g" * 32)
        out = svc.resume("rsvck", checkpoint="latest")
        assert out["run"] == "rsvck"
        svc.stop()

    def test_resume_missing_checkpoint_rejected(self, project_env, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        svc = _service(project_env)
        ds = _dataset(tmp_path)
        TrainingRunConfig(experiment_name="rsvbad", dataset_path=ds).save()
        with pytest.raises(Exception):
            svc.resume("rsvbad", checkpoint="G_999999.pth")

    def test_resume_latest_with_no_checkpoints_rejected(self, project_env,
                                                        tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        svc = _service(project_env)
        ds = _dataset(tmp_path)
        TrainingRunConfig(experiment_name="rsvempty", dataset_path=ds).save()
        with pytest.raises(Exception):
            svc.resume("rsvempty", checkpoint="latest")

    def test_export_no_exports_dir(self, project_env, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        svc = _service(project_env)
        ds = _dataset(tmp_path)
        TrainingRunConfig(experiment_name="expsvc", dataset_path=ds).save()
        with pytest.raises(Exception):
            svc.export("expsvc")

    def test_export_picks_latest(self, project_env, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        svc = _service(project_env)
        ds = _dataset(tmp_path)
        cfg = TrainingRunConfig(experiment_name="expok", dataset_path=ds)
        run = cfg.save().parent
        exp = run / "exports"
        exp.mkdir(parents=True)
        (exp / "model_e100_v2.pth").write_bytes(b"m" * 64)
        time.sleep(0.01)
        (exp / "model_e200_v2.pth").write_bytes(b"m" * 64)
        result = svc.export("expok")
        assert result["model_path"].endswith("model_e200_v2.pth")

    def test_export_by_epoch(self, project_env, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        svc = _service(project_env)
        ds = _dataset(tmp_path)
        cfg = TrainingRunConfig(experiment_name="expby", dataset_path=ds)
        run = cfg.save().parent
        exp = run / "exports"
        exp.mkdir(parents=True)
        (exp / "model_e100_v2.pth").write_bytes(b"m" * 64)
        result = svc.export("expby", epoch=100)
        assert result["model_path"].endswith("model_e100_v2.pth")
        with pytest.raises(Exception):
            svc.export("expby", epoch=333)

    def test_export_empty_exports_rejected(self, project_env, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        svc = _service(project_env)
        ds = _dataset(tmp_path)
        cfg = TrainingRunConfig(experiment_name="expmt", dataset_path=ds)
        (cfg.save().parent / "exports").mkdir(parents=True)
        with pytest.raises(Exception):
            svc.export("expmt")

    def test_stop_without_process(self, project_env):
        assert _service(project_env).stop() == {"running": False}


class TestTrainingServicePrepare:
    def test_prepare_dataset_reports_missing_hubert(self, project_env,
                                                    tmp_path):
        """prepare_all needs the 374 MB HuBERT asset — absent in unit tests,
        so the background thread records a clean 'failed' status with the
        actionable AssetMissing message (the exact production behaviour)."""
        svc = _service(project_env)
        ds = _dataset(tmp_path, n=1, seconds=1.0)
        cfg = svc.make_config(dataset_path=ds, experiment_name="prep1",
                              sample_rate="32k", version="v1",
                              f0_method="pm")
        job = svc.prepare_dataset(cfg)
        assert job == "prep_prep1"
        for _ in range(120):
            st = svc.prep_status("prep1")
            if st["stage"] in ("done", "failed"):
                break
            time.sleep(0.5)
        assert st["stage"] == "failed"
        assert "HuBERT" in st["error"]

    def test_prepare_dataset_succeeds_with_stubbed_pipeline(
            self, project_env, tmp_path, monkeypatch):
        """With the pipeline stubbed, the thread completes and stores the
        report under 'done' — the success arm of the status machine.
        (training_service imports DataPipeline INTO its own namespace at
        module load; the patch must target that binding.)"""
        from rvc_platform.application import training_service as ts_mod

        class FakePipeline:
            def __init__(self, cfg):
                pass

            def prepare_all(self, progress=None):
                progress("preprocess", 0.5, "a.wav")
                return {"ok": True}

        monkeypatch.setattr(ts_mod, "DataPipeline", FakePipeline)
        svc = _service(project_env)
        ds = _dataset(tmp_path, n=1, seconds=1.0)
        cfg = svc.make_config(dataset_path=ds, experiment_name="prep3",
                              sample_rate="32k", version="v1")
        svc.prepare_dataset(cfg)
        for _ in range(60):
            st = svc.prep_status("prep3")
            if st["stage"] in ("done", "failed"):
                break
            time.sleep(0.25)
        assert st["stage"] == "done"
        assert st["report"] == {"ok": True}

    def test_prepare_duplicate_job_rejected(self, project_env, tmp_path):
        import threading

        svc = _service(project_env)
        ds = _dataset(tmp_path, n=2, seconds=3.0)
        cfg = svc.make_config(dataset_path=ds, experiment_name="prep2",
                              sample_rate="32k", version="v1")
        # fake a live thread for the same job id
        svc._prep_threads["prep_prep2"] = threading.Thread(
            target=lambda: time.sleep(5), daemon=True)
        svc._prep_threads["prep_prep2"].start()
        try:
            with pytest.raises(Exception):
                svc.prepare_dataset(cfg)
        finally:
            svc._prep_threads["prep_prep2"].join(timeout=0.1)

    def test_prep_status_idle_for_unknown(self, project_env):
        assert _service(project_env).prep_status("nope") == {"stage": "idle"}


# --------------------------------------------------------------------------- #
# worker argument handling (main() guard clauses — no real training)
# --------------------------------------------------------------------------- #
class TestWorkerMainGuards:
    def test_usage_without_config(self, capsys):
        from rvc_platform.training.worker import main

        assert main([]) == 2
        assert "usage" in capsys.readouterr().err

    def test_missing_config_file(self, capsys):
        from rvc_platform.training.worker import main

        assert main(["--config", "/nonexistent/x.json"]) == 2
        assert "missing" in capsys.readouterr().err

    def test_worker_helper_paths(self, project_env):
        from rvc_platform.training import worker

        # pid registry round-trip: register then unregister leaves no file
        worker._register_trainer_pid()
        worker._unregister_trainer_pid()
        from rvc_platform.infrastructure.paths import get_paths

        assert not (get_paths().temp / "trainer.pid").exists()

    def test_best_export_path_none_when_no_metadata(self, project_env,
                                                    tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig
        from rvc_platform.training.worker import _best_export_path

        ds = _dataset(tmp_path)
        cfg = TrainingRunConfig(experiment_name="bexp", dataset_path=ds)
        assert _best_export_path(cfg, {}) is None


# --------------------------------------------------------------------------- #
# dataset_import
# --------------------------------------------------------------------------- #
class TestDatasetImport:
    def test_parse_sample_rate(self):
        from rvc_platform.training.dataset_import import parse_sample_rate

        assert parse_sample_rate("32k") == 32_000
        assert parse_sample_rate("40000") == 40_000
        assert parse_sample_rate("48K") == 48_000
        with pytest.raises(Exception):
            parse_sample_rate("16k")
        with pytest.raises(Exception):
            parse_sample_rate("banana")

    def test_importer_validation(self):
        from rvc_platform.training.dataset_import import AudioDatasetImporter

        with pytest.raises(Exception):
            AudioDatasetImporter(target_sr=22_050)
        with pytest.raises(Exception):
            AudioDatasetImporter(min_duration_s=0)

    def test_import_folder_validates_source_and_name(self, project_env,
                                                     tmp_path):
        from rvc_platform.core.errors import DatasetError
        from rvc_platform.training.dataset_import import AudioDatasetImporter

        imp = AudioDatasetImporter(target_sr=32_000)
        with pytest.raises(DatasetError):
            imp.import_folder(tmp_path / "ghost", "ok_name")
        src = _dataset(tmp_path / "s")
        with pytest.raises(DatasetError):
            imp.import_folder(src, "bad name!!")

    def test_import_folder_requires_force(self, project_env, tmp_path):
        from rvc_platform.core.errors import DatasetError
        from rvc_platform.training.dataset_import import AudioDatasetImporter

        imp = AudioDatasetImporter(target_sr=32_000)
        src = _dataset(tmp_path / "src1")
        out_root = tmp_path / "out"
        imp.import_folder(src, "ds1", output_root=out_root)
        with pytest.raises(DatasetError):
            imp.import_folder(src, "ds1", output_root=out_root)
        # force replaces it
        report = imp.import_folder(src, "ds1", output_root=out_root,
                                   force=True)
        assert report.included_files == 2

    def test_import_folder_overlapping_source_rejected(self, project_env,
                                                       tmp_path):
        from rvc_platform.core.errors import DatasetError
        from rvc_platform.training.dataset_import import AudioDatasetImporter

        imp = AudioDatasetImporter(target_sr=32_000)
        src = _dataset(tmp_path / "src2")
        # output INSIDE source -> overlap error
        with pytest.raises(DatasetError):
            imp.import_folder(src, "inside", output_root=src)

    def test_import_folder_full_report(self, project_env, tmp_path):
        from rvc_platform.training.dataset_import import AudioDatasetImporter

        imp = AudioDatasetImporter(target_sr=32_000, min_duration_s=0.2)
        src = tmp_path / "src3"
        src.mkdir()
        # one good file, one too-short file, one non-audio file
        t = np.arange(32_000) / 32_000
        good = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        sf.write(str(src / "good.wav"), good, 32_000, subtype="PCM_16")
        sf.write(str(src / "tiny.wav"), good[:100], 32_000,
                 subtype="PCM_16")
        (src / "note.txt").write_text("x", encoding="utf-8")
        report = imp.import_folder(src, "rpt", output_root=tmp_path / "out3")
        d = report.to_dict()
        assert d["included_files"] == 1
        assert d["sample_rate"] == 32_000
        assert d["channels"] == 1
        assert d["format"] == "PCM_16 WAV"
        assert d["skipped_count"] >= 1      # tiny (and non-audio files ignored)
        assert d["total_duration_s"] > 0


# --------------------------------------------------------------------------- #
# downloads
# --------------------------------------------------------------------------- #
class TestDownloads:
    def test_asset_status_shape(self, project_env):
        from rvc_platform.application.downloads import asset_status

        status = asset_status()
        assert "_summary" in status
        assert "all_present" in status["_summary"]

    def test_download_assets_with_stubbed_downloader(self, project_env,
                                                      monkeypatch, tmp_path):
        """download_assets iterates cfg.downloads.files; existing assets
        short-circuit, missing ones go through _download_one (stubbed — no
        network)."""
        from rvc_platform.application import downloads as dl
        from rvc_platform.infrastructure.config.loader import get_config

        cfg = get_config().downloads
        assert len(cfg.files) > 0        # manifest ships in default.yaml
        # make every asset absent so all are fetched through the stub
        for remote in cfg.files:
            (dl.get_paths().assets / cfg.files[remote]).unlink(missing_ok=True)
        calls = []

        def fake_one(url, dest, chunk_mb, timeout, progress=None):
            calls.append(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"fake-asset")

        monkeypatch.setattr(dl, "_download_one", fake_one)
        report = dl.download_assets()
        assert len(calls) == len(cfg.files)   # every missing file fetched
        assert set(report) - {"_failures"} == set(cfg.files)
        # second run: everything exists now -> no downloads, no failures
        calls.clear()
        report2 = dl.download_assets()
        assert calls == []
        assert "exists" in list(report2.values())[0]

    def test_download_one_http_error(self, project_env, tmp_path):
        from rvc_platform.application.downloads import _download_one

        with pytest.raises(Exception):
            _download_one("http://127.0.0.1:59996/nope.bin",
                          tmp_path / "x.bin", chunk_mb=1, timeout=2)


# --------------------------------------------------------------------------- #
# model_service
# --------------------------------------------------------------------------- #
class TestModelService:
    def _svc(self, project_env, engine=None):
        from rvc_platform.application.model_service import ModelService
        from rvc_platform.infrastructure.paths import get_paths
        from rvc_platform.infrastructure.persistence.model_repository import \
            ModelRepository

        return ModelService(ModelRepository(get_paths().models),
                            engine=engine)

    def test_list_and_get_empty(self, project_env):
        svc = self._svc(project_env)
        assert svc.list() == []
        with pytest.raises(Exception):
            svc.get("ghost")

    def test_register_upload_payload(self, project_env, tmp_path):
        """Upload path: the payload lands in models/ under the temp
        upload_<hash>_ prefix (repo convention — names come from the file),
        and the checkpoint metadata is probed from the copy."""
        import torch

        svc = self._svc(project_env)
        payload_path = tmp_path / "up.pth"
        torch.save({"sr": 40000, "f0": 1, "version": "v2"}, payload_path)
        payload = payload_path.read_bytes()
        info = svc.register_upload_payload("my_voice.pth", payload)
        assert info.name.startswith("upload_")
        assert info.name.endswith("_my_voice")
        assert info.path.is_file()
        assert info.sample_rate == 40_000
        # a second upload gets its own temp-hash name (uuid per call)
        info2 = svc.register_upload_payload("my_voice.pth", payload)
        assert info2.name.startswith("upload_")
        assert info2.name.endswith("_my_voice")
        # both copies exist in the registry
        assert info.path.is_file() and info2.path.is_file()

    def test_register_upload_payload_rejects_bad_name(self, project_env):
        svc = self._svc(project_env)
        with pytest.raises(Exception):
            svc.register_upload_payload("no-extension", b"data")
        with pytest.raises(Exception):
            svc.register_upload_payload("../evil.pth", b"data")

    def test_register_upload_from_path(self, project_env, tmp_path):
        import torch

        svc = self._svc(project_env)
        src = tmp_path / "src_model.pth"
        torch.save({"model": {}}, src)
        info = svc.register_upload(src, name="renamed")
        assert info.name == "renamed"

    def test_info_unknown_model(self, project_env):
        with pytest.raises(Exception):
            self._svc(project_env).info("ghost")

    def test_delete_removes_model(self, project_env, tmp_path):
        import torch

        class FakeEngine:
            loaded_model = None

            def unload(self):
                self.loaded_model = None

        svc = self._svc(project_env, engine=FakeEngine())
        src = tmp_path / "del.pth"
        torch.save({"sr": 40000, "f0": 1, "version": "v2"}, src)
        info = svc.register_upload(src, name="todelete")
        svc.delete("todelete")
        with pytest.raises(Exception):
            svc.get("todelete")

    def test_loaded_and_unload_with_engine(self, project_env):
        class FakeEngine:
            loaded_model = None

            def unload(self):
                self.loaded_model = None

        eng = FakeEngine()
        svc = self._svc(project_env, engine=eng)
        assert svc.loaded() is None
        svc.unload()                     # delegates to engine.unload
        assert eng.loaded_model is None
