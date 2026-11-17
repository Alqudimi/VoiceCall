"""Cloudflare quick-tunnel supervision.

Runs ``cloudflared tunnel --url http://127.0.0.1:<port>`` (official
quick-tunnel mode -> free ephemeral trycloudflare.com URL), scrapes the
public URL from the process log, verifies reachability, supervises the
process, and *enforces bearer auth* on the API while exposed.

Quick tunnels are temporary by design — a new URL on every start; no
persistent hostname is claimed.
"""
from __future__ import annotations

import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

from rvc_platform.core.errors import TunnelError
from rvc_platform.infrastructure.processes.manager import ManagedProcess, run_capture

logger = logging.getLogger(__name__)


class TunnelService:
    PROCESS_NAME = "cloudflared"

    def __init__(self, process_manager, server_port: int, retries: int = 3,
                 start_timeout_s: float = 30.0, auth_enabler=None) -> None:
        self.pm = process_manager
        self.port = server_port
        self.retries = retries
        self.timeout = start_timeout_s
        self.auth_enabler = auth_enabler  # callable -> enable bearer auth
        self.url: str | None = None
        self.started_at: float | None = None
        self.log_dir = process_manager.log_dir

    # ------------------------------------------------------------------ binary
    def binary_available(self) -> bool:
        return shutil.which(self.PROCESS_NAME) is not None

    def _probe_server(self) -> bool:
        code, _ = run_capture(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             f"http://127.0.0.1:{self.port}/api/system"],
            timeout=10,
        )
        # any HTTP answer (incl. 401) proves the server is up
        return code == 0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> dict[str, Any]:
        if self.pm.get(self.PROCESS_NAME) and self.pm.get(self.PROCESS_NAME).is_running():
            return self.status()
        if not self.binary_available():
            raise TunnelError(
                "cloudflared is not installed; see docs/INSTALLATION.md "
                "(Linux: `sudo apt install cloudflared` / wget the .deb; "
                "Windows: winget install Cloudflare.cloudflared)"
            )
        if not self._probe_server():
            raise TunnelError(f"local server on port {self.port} is not responding")
        if self.auth_enabler:
            self.auth_enabler()  # never expose the dashboard unauthenticated

        for attempt in range(self.retries):
            proc = self.pm.create(
                self.PROCESS_NAME,
                [self.PROCESS_NAME, "tunnel", "--url", f"http://127.0.0.1:{self.port}",
                 "--no-autoupdate", "--protocol", "http2"],
            )
            proc.start()
            deadline = time.time() + self.timeout
            url = None
            while time.time() < deadline:
                for line in proc.tail(80):
                    found = _URL_RE_SEARCH(line)
                    if found:
                        url = found
                        break
                if url:
                    break
                time.sleep(0.4)
                if not proc.is_running():
                    break
            if url:
                self.url, self.started_at = url, time.time()
                logger.info("tunnel up: %s", url)
                return self.status()
            logger.warning("tunnel attempt %s/%s failed", attempt + 1, self.retries)
            self.pm.stop(self.PROCESS_NAME)
        raise TunnelError(
            f"could not obtain a trycloudflare URL after {self.retries} attempts; "
            "check the log at " + str(self.log_dir / self.PROCESS_NAME)
        )

    def stop(self) -> dict[str, Any]:
        ok = self.pm.stop(self.PROCESS_NAME)
        self.url = None
        return {"running": False, "clean": ok}

    def status(self) -> dict[str, Any]:
        proc = self.pm.get(self.PROCESS_NAME)
        running = bool(proc and proc.is_running())
        return {
            "running": running,
            "url": self.url if running else None,
            "pid": proc.proc.pid if running and proc.proc else None,
            "started_at": self.started_at if running else None,
            "log_path": str(self.log_dir / self.PROCESS_NAME) if running else None,
        }


def _URL_RE_SEARCH(line: str) -> str | None:
    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    m = pattern.search(line)
    return m.group(0) if m else None
