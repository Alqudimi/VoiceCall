"""Unit tests: process manager (start/stop/escalation/logs)."""
from __future__ import annotations

import sys
import time

import pytest

from rvc_platform.infrastructure.processes.manager import ProcessManager, run_capture
from rvc_platform.core.errors import ProcessError


def test_managed_process_lifecycle(tmp_path):
    pm = ProcessManager(tmp_path / "logs")
    proc = pm.create("sleeper", [sys.executable, "-c", "import time; time.sleep(30)"])
    proc.start()
    assert proc.is_running()
    info = proc.info()
    assert info.running and info.pid and info.pid > 0
    assert info.log_path.name == "sleeper.log"

    ok = pm.stop("sleeper")
    assert ok
    assert not proc.is_running()
    # second stop is a no-op returning False (already gone)
    assert pm.stop("sleeper") is False


def test_process_log_capture(tmp_path):
    pm = ProcessManager(tmp_path / "logs")
    proc = pm.create("echoer", [sys.executable, "-c", "print('hello-from-worker')"])
    proc.start()
    proc.proc.wait()
    time.sleep(0.05)
    lines = proc.tail(10)
    assert any("hello-from-worker" in ln for ln in lines)


def test_single_instance_guard(tmp_path):
    pm = ProcessManager(tmp_path / "logs")
    pm.create("trainer", [sys.executable, "-c", "import time; time.sleep(30)"]).start()
    with pytest.raises(ProcessError):
        pm.create("trainer", [sys.executable, "-c", "pass"]).start()
    pm.stop_all()


def test_stop_all_survives_failures(tmp_path):
    pm = ProcessManager(tmp_path / "logs")
    pm.create("a", [sys.executable, "-c", "print('a')"]).start()
    pm.stop_all()  # must not raise even with finished entries
    assert pm.status() is not None


def test_run_capture():
    code, out = run_capture([sys.executable, "-c", "print('captured')"])
    assert code == 0 and "captured" in out
    code, out = run_capture(["definitely-not-a-binary-xyz"], timeout=5)
    assert code == 127
