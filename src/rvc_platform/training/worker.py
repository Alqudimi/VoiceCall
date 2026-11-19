"""Training worker — the subprocess entrypoint launched by the API/CLI.

Protocol (consumed by TrainingService):
    stdin  : none (config via argv --config <run_config.json>)
    stdout : JSONL events  {"topic": "training.progress|started|..."}
    files  : run_dir/progress.json  (atomic, always current)
             run_dir/events.jsonl   (append-only journal)
    signals: SIGTERM -> cooperative stop (checkpoint saved, exit code 0)

Exit codes: 0 success/stop, 2 configuration error, 3 runtime failure.
"""
from __future__ import annotations

import json
import shutil
import signal
import sys
import time
from pathlib import Path

from rvc_platform.infrastructure.logging.setup import setup_logging
from rvc_platform.infrastructure.paths import get_paths
from rvc_platform.training.run_config import TrainingRunConfig
from rvc_platform.training.trainer import Trainer
from rvc_platform.training.utils import (
    atomic_json_dump,
    get_run_logger,
    stable_dataset_signature,
)
from rvc_platform.utils.device import select_device, training_dtype


def _register_trainer_pid() -> None:
    """Publish this worker's pid so `rvc-platform train stop` (CLI or API)
    can send a cooperative-stop SIGTERM from any terminal."""
    pid_file = get_paths().temp / "trainer.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(__import__("os").getpid()), encoding="utf-8")


def _unregister_trainer_pid() -> None:
    try:
        (get_paths().temp / "trainer.pid").unlink(missing_ok=True)
    except OSError:
        pass


def _clear_previous_training(config: TrainingRunConfig) -> None:
    """Clear old weights/status while preserving the prepared data pipeline."""
    layout = config.layout()
    for key in ("checkpoints", "exports"):
        folder = layout[key]
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    for key in ("progress", "events", "train_log"):
        layout[key].unlink(missing_ok=True)
    (config.run_dir / "huggingface.json").unlink(missing_ok=True)
    config.resume = False
    config.resume_from = None


def _publish(config: TrainingRunConfig, emit) -> None:
    if not config.hf_repo_id:
        return
    try:
        from rvc_platform.training.huggingface import publish_run
        url = publish_run(config.run_dir, config.hf_repo_id)
        (config.run_dir / "huggingface.json").write_text(
            json.dumps({"repo_id": config.hf_repo_id, "url": url, "uploaded_at": time.time()},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        emit({"topic": "training.hf_uploaded", "repo_id": config.hf_repo_id, "url": url})
    except Exception as exc:  # publish failures must not invalidate training
        emit({"topic": "training.hf_upload_failed", "repo_id": config.hf_repo_id,
              "error": str(exc)})


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] != "--config":
        print("usage: python -m rvc_platform.training.worker --config <run_config.json>", file=sys.stderr)
        return 2
    config_path = Path(argv[1])
    if not config_path.is_file():
        print(f"run config missing: {config_path}", file=sys.stderr)
        return 2

    setup_logging()
    config = TrainingRunConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    clean_start_requested = config.clean_start
    if clean_start_requested:
        _clear_previous_training(config)
        # clean_start is a one-shot instruction, not a persistent run mode.
        # Persist the normalized config so a later resume cannot erase new checkpoints.
        config.clean_start = False
        config.save()
    layout = config.ensure_layout()
    hps = config.to_hparams()
    hps.pretrainG, hps.pretrainD = ("", "")
    if config.preload_base_models and not clean_start_requested:
        g, d = config.resolve_pretrained()
        if g.is_file() and d.is_file():
            hps.pretrainG, hps.pretrainD = str(g), str(d)
    logger = get_run_logger(f"train.{config.experiment_name}", layout["train_log"])

    profile = select_device(config.gpu)
    dtype = training_dtype(profile.device)
    logger.info("training on %s (%s)", profile.device, dtype)

    events_path = layout["events"]
    progress_path = layout["progress"]

    def emit(payload: dict) -> None:
        payload = dict(payload)
        payload["ts"] = time.time()
        with open(events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if payload.get("topic") in ("training.progress", "training.started",
                                   "training.finished", "training.failed",
                                   "training.resumed"):
            atomic_json_dump(payload, progress_path)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        if payload.get("topic") in ("training.exported", "training.final_model"):
            _publish(config, emit)

    trainer = Trainer(config, hps, profile.device, dtype, event_cb=emit)

    def _handle_sigterm(signum, frame) -> None:  # noqa: ARG001
        logger.info("SIGTERM received -> cooperative stop")
        trainer.request_stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    _register_trainer_pid()
    try:
        dataset_files = sorted(Path(config.dataset_path).rglob("*"))
        signature = stable_dataset_signature(dataset_files)
        completed = trainer.run(dataset_signature=signature)
        return 0 if completed else 0  # cooperative stop is also success (0)
    except Exception as exc:
        logger.exception("training worker failed")
        emit({"topic": "training.failed", "error": str(exc)})
        return 3
    finally:
        _unregister_trainer_pid()


if __name__ == "__main__":
    raise SystemExit(main())
