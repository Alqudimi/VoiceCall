"""Structured logging: console (concise) + rotating files (verbose)."""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from rvc_platform.infrastructure.config.loader import get_config
from rvc_platform.infrastructure.paths import get_paths

_FORMAT_CONSOLE = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_FORMAT_FILE = (
    "%(asctime)s | %(levelname)-7s | %(name)s | %(process)d | %(message)s"
)

_initialized = False


def setup_logging(level: int | None = None) -> None:
    global _initialized
    if _initialized:
        return
    cfg = get_config().logging
    root = logging.getLogger()
    root.setLevel(level if level is not None else getattr(logging, cfg.level))

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_FORMAT_CONSOLE, datefmt="%H:%M:%S"))
    root.addHandler(console)

    try:
        log_dir = get_paths().logs
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "platform.log",
            maxBytes=cfg.max_bytes,
            backupCount=cfg.backups,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(_FORMAT_FILE))
        root.addHandler(file_handler)
    except OSError:  # read-only FS — console logging still works
        root.warning("file logging disabled (log dir not writable)")

    for noisy in ("httpx", "httpcore", "urllib3", "matplotlib", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
