"""Training lifecycle service: dataset preparation + trainer subprocess.

All heavy work runs out-of-process (trainer) or in a background thread
(dataset preparation) so the API stays responsive and a crashed training
never takes the server down.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from rvc_platform.core.entities import JobState, TrainingProgress
from rvc_platform.core.errors import TrainingError
from rvc_platform.infrastructure.paths import get_paths
from rvc_platform.infrastructure.processes.manager import ProcessManager
from rvc_platform.training import checkpointing
from rvc_platform.training.data_pipeline import DataPipeline
from rvc_platform.training.run_config import TrainingRunConfig
from rvc_platform.training.checkpointing import read_state

_PY = "python"


class TrainingService:
    def __init__(self, process_manager: ProcessManager, python_bin: str = "") -> None:
        self.pm = process_manager
        import sys

        self.python = python_bin or sys.executable
        self._prep_threads: dict[str, threading.Thread] = {}
        self._prep_status: dict[str, dict] = {}
        self.paths = get_paths()

    # ------------------------------------------------------------------ config
    def make_config(self, **kwargs) -> TrainingRunConfig:
        dataset = kwargs.pop("dataset_path", None)
        if dataset is None:
            raise TrainingError("dataset_path is required")
        kwargs["dataset_path"] = str(dataset)
        name = kwargs.get("experiment_name") or f"iraqi_run_{int(time.time())}"
        kwargs.setdefault("experiment_name", name)
        existing = Path(dataset)
        if not existing.is_dir():
            raise TrainingError(f"dataset folder not found: {existing}")
        return TrainingRunConfig(**kwargs)

    # ------------------------------------------------------------------ prepare
    def prepare_dataset(self, config: TrainingRunConfig) -> str:
        """Run the full data pipeline in a background thread. Returns job id."""
        job_id = f"prep_{config.experiment_name}"
        if job_id in self._prep_threads and self._prep_threads[job_id].is_alive():
            raise TrainingError("dataset preparation already running for this experiment")

        def _stage(stage: str, frac: float, name: str) -> None:
            self._prep_status[job_id] = {
                "stage": stage,
                "fraction": round(frac, 4),
                "file": name,
                "updated_at": time.time(),
            }

        def _run() -> None:
            try:
                pipeline = DataPipeline(config)
                self._prep_status[job_id] = {"stage": "starting", "fraction": 0.0,
                                              "updated_at": time.time()}
                report = pipeline.prepare_all(progress=_stage)
                self._prep_status[job_id] = {
                    "stage": "done", "fraction": 1.0, "report": report,
                    "updated_at": time.time(),
                }
            except Exception as exc:  # noqa: BLE001 - report into status dict
                self._prep_status[job_id] = {
                    "stage": "failed", "error": str(exc), "updated_at": time.time(),
                }

        thread = threading.Thread(target=_run, name=job_id, daemon=True)
        self._prep_threads[job_id] = thread
        thread.start()
        return job_id

    def prep_status(self, experiment: str) -> dict:
        return self._prep_status.get(
            f"prep_{experiment}", {"stage": "idle"}
        )

    # ------------------------------------------------------------------ train
    def start(self, config: TrainingRunConfig, resume: bool = True) -> dict:
        config.resume = resume
        config.validate_dataset()
        cfg_path = config.save()
        proc = self.pm.create(
            "trainer",
            [self.python, "-m", "rvc_platform.training.worker", "--config", str(cfg_path)],
            cwd=str(get_paths().root),
        )
        proc.start()
        return {"pid": proc.proc.pid if proc.proc else None,
                "run": config.experiment_name, "config": str(cfg_path)}

    def stop(self) -> dict:
        proc = self.pm.get("trainer")
        if proc is None:
            return {"running": False}
        info = proc.info()
        ok = self.pm.stop("trainer")
        return {"running": False, "was_pid": info.pid, "clean": ok}

    def status(self) -> TrainingProgress:
        progress = TrainingProgress()
        run_dirs = sorted(get_paths().runs.glob("*/run_config.json"))
        if not run_dirs:
            progress.message = "no training runs yet"
            return progress
        live = self.pm.get("trainer")
        running = live is not None and live.is_running()
        chosen = None
        chosen_dir = None
        for rc_path in run_dirs:
            run_dir = rc_path.parent
            progress_file = run_dir / "progress.json"
            if not progress_file.is_file():
                continue
            try:
                payload = json.loads(progress_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if chosen is None or payload.get("ts", 0) > chosen.get("ts", 0):
                chosen = payload
                chosen_dir = run_dir
        if chosen is not None:
            raw_state = chosen.get("state", "running")
            try:
                progress.state = JobState(raw_state)
            except ValueError:
                progress.state = JobState.RUNNING
            progress.epoch = int(chosen.get("epoch", 0) or 0)
            progress.total_epochs = int(chosen.get("total_epochs", 0) or 0)
            progress.global_step = int(chosen.get("global_step", 0) or 0)
            progress.steps_per_epoch = int(chosen.get("steps_per_epoch", 0) or 0)
            progress.latest_losses = chosen.get("latest_losses", {}) or {}
            progress.updated_at = chosen.get("ts")
            progress.message = chosen_dir.name if chosen_dir else ""
        progress.state = JobState.RUNNING if running else (
            progress.state if progress.state not in (JobState.RUNNING,) else JobState.IDLE
        )
        return progress

    # ------------------------------------------------------------------ checkpoints
    def checkpoints(self, experiment: str) -> list[dict]:
        run_dir = get_paths().runs / experiment
        if not run_dir.is_dir():
            raise TrainingError(f"no such run: {experiment}")
        return [c.to_dict() for c in checkpointing.list_checkpoints(run_dir / "checkpoints")]

    def logs(self, experiment: str, lines: int = 200) -> list[str]:
        log_path = get_paths().runs / experiment / "train.log"
        if not log_path.is_file():
            return []
        return log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]

    def resume(self, experiment: str | None = None) -> dict:
        if experiment is None:
            candidates = sorted(
                get_paths().runs.glob("*/run_config.json"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if not candidates:
                raise TrainingError("no runs to resume")
            experiment = candidates[0].parent.name
        config = TrainingRunConfig.load(get_paths().runs / experiment)
        return self.start(config, resume=True)

    def export(self, experiment: str, epoch: int | None = None,
               model_repo=None) -> dict:
        """Copy a training export (+ index) into the models registry."""
        run_dir = get_paths().runs / experiment
        exports = run_dir / "exports"
        if not exports.is_dir():
            raise TrainingError(f"no exports directory in {run_dir}")
        if epoch is not None:
            matches = sorted(exports.glob(f"*_e{epoch}_*.pth"))
            if not matches:
                raise TrainingError(f"no export for epoch {epoch}")
            chosen = matches[-1]
        else:
            pths = sorted(exports.glob("*.pth"))
            if not pths:
                raise TrainingError("no exported checkpoints")
            chosen = max(pths, key=lambda p: p.stat().st_mtime)
        result: dict = {"model_path": str(chosen)}
        if model_repo is not None:
            info = model_repo.register(chosen)
            result["registered"] = info.to_dict()
        index = list((run_dir).glob("added_IVF*.index"))
        if index and model_repo is not None:
            latest_index = max(index, key=lambda p: p.stat().st_mtime)
            model_repo.register_index(latest_index, chosen.stem + ".pth")
            result["index_path"] = str(latest_index)
        return result
