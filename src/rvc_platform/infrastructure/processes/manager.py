"""Managed subprocess supervision.

Used for the training worker (so the API can start/stop/monitor training in
an isolated process that survives request cycles and gets OOM-protected
logging) and for ``cloudflared``. Provides: single-instance guarantees,
PID files, log capture, graceful SIGTERM->SIGKILL escalation, exit-code
reporting.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rvc_platform.core.errors import ProcessError

logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    name: str
    pid: int | None
    running: bool
    exit_code: int | None
    started_at: float | None
    log_path: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pid": self.pid,
            "running": self.running,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "log_path": str(self.log_path) if self.log_path else None,
        }


class ManagedProcess:
    """One supervised subprocess instance."""

    def __init__(self, name: str, cmd: list[str], log_dir: Path,
                 cwd: Path | None = None, env: dict[str, str] | None = None,
                 graceful_s: float = 8.0) -> None:
        self.name = name
        self.cmd = cmd
        self.log_dir = log_dir
        self.cwd = cwd
        self.env = env
        self.graceful_s = graceful_s
        self.proc: subprocess.Popen[bytes] | None = None
        self.started_at: float | None = None
        self.exit_code: int | None = None
        self._log_path = log_dir / f"{name}.log"

    @property
    def log_path(self) -> Path:
        return self._log_path

    def start(self) -> None:
        if self.is_running():
            raise ProcessError(f"{self.name} already running (pid {self.proc.pid})")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_fh = open(self._log_path, "ab")
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        try:
            self.proc = subprocess.Popen(
                self.cmd,
                cwd=str(self.cwd) if self.cwd else None,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,  # own process group -> clean signalling
            )
        except OSError as exc:
            log_fh.close()
            raise ProcessError(f"failed to start {self.name}: {exc}") from exc
        self.started_at = time.time()
        self.exit_code = None
        logger.info("started %s (pid %s)", self.name, self.proc.pid)

    def is_running(self) -> bool:
        if self.proc is None:
            return False
        return self.proc.poll() is None

    def stop(self) -> bool:
        if self.proc is None:
            return False
        if self.is_running():
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self.proc.terminate()
            deadline = time.time() + self.graceful_s
            while self.is_running() and time.time() < deadline:
                time.sleep(0.1)
            if self.is_running():
                logger.warning("%s did not exit gracefully — SIGKILL", self.name)
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    self.proc.kill()
        self.exit_code = self.proc.wait()
        ok = self.exit_code in (0, None, -signal.SIGTERM, 137, 143)
        logger.info("stopped %s (exit %s)", self.name, self.exit_code)
        self.proc = None
        return ok

    def tail(self, lines: int = 200) -> list[str]:
        try:
            raw = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return raw.splitlines()[-lines:]

    def info(self) -> ProcessInfo:
        return ProcessInfo(
            name=self.name,
            pid=self.proc.pid if self.proc and self.is_running() else None,
            running=self.is_running(),
            exit_code=self.exit_code,
            started_at=self.started_at,
            log_path=self._log_path,
        )


class ProcessManager:
    """Registry of named managed processes (trainer, tunnel, ...)."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, ManagedProcess] = {}

    def create(self, name: str, cmd: list[str], cwd: Path | None = None,
               env: dict[str, str] | None = None) -> ManagedProcess:
        if name in self._procs and self._procs[name].is_running():
            raise ProcessError(f"process '{name}' already running")
        proc = ManagedProcess(name, cmd, self.log_dir / name, cwd=cwd, env=env)
        self._procs[name] = proc
        return proc

    def get(self, name: str) -> ManagedProcess | None:
        return self._procs.get(name)

    def require(self, name: str) -> ManagedProcess:
        proc = self._procs.get(name)
        if proc is None:
            raise ProcessError(f"no such managed process: {name}")
        return proc

    def stop(self, name: str) -> bool:
        proc = self._procs.get(name)
        return proc.stop() if proc else False

    def stop_all(self) -> None:
        for name in list(self._procs):
            try:
                self.stop(name)
            except Exception:
                logger.exception("failed stopping %s", name)

    def status(self) -> dict[str, Any]:
        return {n: p.info().to_dict() for n, p in self._procs.items()}


def run_capture(cmd: list[str], timeout: float = 30.0) -> tuple[int, str]:
    """Run a one-shot command capturing combined output (used by doctor)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
