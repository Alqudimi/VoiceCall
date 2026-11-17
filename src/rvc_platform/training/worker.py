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
import signal
import sys
import time
from pathlib import Path

from rvc_platform.infrastructure.logging.setup import setup_logging
from rvc_platform.training.run_config import TrainingRunConfig
from rvc_platform.training.trainer import Trainer
from rvc_platform.training.utils import (
    atomic_json_dump,
    get_run_logger,
    stable_dataset_signature,
)
from rvc_platform.utils.device import select_device, training_dtype


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
    layout = config.ensure_layout()
    hps = config.to_hparams()
    hps.pretrainG, hps.pretrainD = ("", "")
    if config.preload_base_models and not config.clean_start:
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

    trainer = Trainer(config, hps, profile.device, dtype, event_cb=emit)

    def _handle_sigterm(signum, frame) -> None:  # noqa: ARG001
        logger.info("SIGTERM received -> cooperative stop")
        trainer.request_stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    try:
        dataset_files = sorted(Path(config.dataset_path).glob("*"))
        signature = stable_dataset_signature(dataset_files)
        completed = trainer.run(dataset_signature=signature)
        return 0 if completed else 0  # cooperative stop is also success (0)
    except Exception as exc:
        logger.exception("training worker failed")
        emit({"topic": "training.failed", "error": str(exc)})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
