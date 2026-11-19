"""Training worker — the subprocess entrypoint launched by the API/CLI.

Protocol (consumed by TrainingService):
    stdin  : none (config via argv --config <run_config.json>)
    stdout : JSONL events  {"topic": "training.progress|started|..."}
    files  : run_dir/progress.json  (atomic, always current)
             run_dir/events.jsonl   (append-only journal)
             run_dir/huggingface.json  (last HF upload status, incl. best)
    signals: SIGTERM -> cooperative stop (checkpoint saved, exit code 0)

Exit codes: 0 success/stop, 2 configuration error, 3 runtime failure.

Threading model (nothing slow ever runs on the training thread):
    training thread   numerics + CPU state snapshots only
    io-* thread       checkpoint serialization, mirrors, state.json, exports
    hf-publisher:*    Hugging Face commits (retry/backoff, atomic, ordered)
Uploads are triggered (a) whenever an export lands on disk (the trainer's
background I/O worker emits training.exported only after the file exists) and
(b) whenever a *new best* model lands, chained through the trainer's
best_export_cb. At process exit the best model is double-checked against the
publisher's record so a resumed run that never beat its old best still leaves
best/ up to date on the Hub.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import threading
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

HF_CLOSE_TIMEOUT_DEFAULT = 300.0


def _register_trainer_pid() -> None:
    """Publish this worker's pid so `rvc-platform train stop` (CLI or API)
    can send a cooperative-stop SIGTERM from any terminal."""
    pid_file = get_paths().temp / "trainer.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")


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
    for extra in ("filelist_train.txt", "filelist_val.txt",
                  "validation_split.json"):
        (config.run_dir / extra).unlink(missing_ok=True)
    (config.run_dir / "huggingface.json").unlink(missing_ok=True)
    config.resume = False
    config.resume_from = None


def _best_export_path(config: TrainingRunConfig, metadata: dict) -> Path | None:
    """Locate the best-model export file on disk (metadata hint, else newest)."""
    exports = config.run_dir / "exports"
    candidates = []
    name = metadata.get("export")
    if name:
        candidates.append(exports / name)
    if exports.is_dir():
        candidates.extend(sorted(
            exports.glob(f"{config.experiment_name}_best_e*_s*.pth"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _ensure_best_published(config: TrainingRunConfig, publisher) -> None:
    """Final safety net: make sure the current best.json is on the Hub even
    when this session resumed past it without setting a new best."""
    best_json = config.run_dir / "checkpoints" / "best.json"
    if not best_json.is_file():
        return
    try:
        metadata = json.loads(best_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    key = f"{metadata.get('epoch')}:{metadata.get('global_step')}"
    if publisher.last_best_key == key:
        return
    model_path = _best_export_path(config, metadata)
    if model_path is None:
        return
    from rvc_platform.training.huggingface import newest_index_file

    publisher.submit_best(model_path, newest_index_file(config.run_dir), metadata)


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

    # ---- background Hugging Face publisher (no upload ever blocks training)
    emit_lock = threading.Lock()

    def emit(payload: dict) -> None:
        payload = dict(payload)
        payload["ts"] = time.time()
        with emit_lock:
            with open(events_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if payload.get("topic") in ("training.progress", "training.started",
                                       "training.finished", "training.failed",
                                       "training.resumed"):
                atomic_json_dump(payload, progress_path)
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            if hf_publisher is not None and payload.get("topic") in (
                    "training.exported", "training.final_model"):
                # the export file already exists at this point (the trainer
                # emits from the I/O worker only after the write completes)
                hf_publisher.submit_export(config.run_dir, payload["path"])

    def _on_hf_event(event: dict) -> None:
        """Publisher-thread callback: persist huggingface.json + journal it."""
        try:
            status = {
                "repo_id": event.get("repo_id") or config.hf_repo_id,
                "url": event.get("url"),
                "uploaded_at": time.time(),
                "last_upload": {
                    key: value for key, value in event.items() if key != "topic"
                },
            }
            if event.get("kind") == "best" and event.get("best"):
                status["best"] = event["best"]
            (config.run_dir / "huggingface.json").write_text(
                json.dumps(status, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("could not update huggingface.json", exc_info=True)
        emit({key: value for key, value in event.items() if key != "topic"})

    hf_publisher = None
    if config.hf_repo_id:
        try:
            from rvc_platform.training.huggingface import HfBackgroundPublisher

            hf_publisher = HfBackgroundPublisher(config.hf_repo_id,
                                                 on_event=_on_hf_event)
            hf_publisher.start()
        except Exception as exc:  # publisher setup must never kill a run
            logger.warning("Hugging Face publisher unavailable: %s", exc)
            hf_publisher = None

    def _best_export_cb(model_path: str, metadata: dict) -> None:
        if hf_publisher is None:
            return
        from rvc_platform.training.huggingface import newest_index_file

        hf_publisher.submit_best(model_path,
                                 newest_index_file(config.run_dir), metadata)

    trainer = Trainer(config, hps, profile.device, dtype, event_cb=emit,
                      best_export_cb=_best_export_cb if hf_publisher else None)

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
        if hf_publisher is not None:
            try:
                _ensure_best_published(config, hf_publisher)
                close_timeout = float(os.environ.get(
                    "RVC_HF_CLOSE_TIMEOUT", HF_CLOSE_TIMEOUT_DEFAULT))
                summary = hf_publisher.close(timeout=close_timeout)
                logger.info("HF publisher closed: %s", summary)
            except Exception:
                logger.exception("HF publisher shutdown failed")
        _unregister_trainer_pid()


if __name__ == "__main__":
    raise SystemExit(main())
