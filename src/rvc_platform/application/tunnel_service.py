"""Cloudflare quick-tunnel supervision.

Runs ``cloudflared tunnel --url http://127.0.0.1:<port>`` (official
quick-tunnel mode -> free ephemeral trycloudflare.com URL), scrapes the
public URL from the process log, verifies reachability, supervises the
process, and *enforces bearer auth* on the API while exposed.

Quick tunnels are temporary by design — a new URL on every start; no
persistent hostname is claimed.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from rvc_platform.core.errors import TunnelError

logger = logging.getLogger(__name__)


class TunnelService:
    PROCESS_NAME = "cloudflared"

    def __init__(self, process_manager, server_port: int, retries: int = 3,
                 start_timeout_s: float = 30.0, auth_enabler=None,
                 binary: str = "cloudflared") -> None:
        self.pm = process_manager
        self.port = server_port
        self.retries = retries
        self.timeout = start_timeout_s
        self.binary = binary
        self.auth_enabler = auth_enabler  # callable -> enable bearer auth
        self.url: str | None = None
        self.started_at: float | None = None
        self.log_dir = process_manager.log_dir

    # ------------------------------------------------------------------ binary
    def binary_available(self) -> bool:
        return self._resolve_binary() is not None

    def _resolve_binary(self) -> str | None:
        configured = Path(self.binary).expanduser()
        if configured.is_file():
            return str(configured.resolve())
        found = shutil.which(self.binary) or shutil.which(self.PROCESS_NAME)
        if found:
            return found
        candidates = (
            Path.home() / ".local" / "bin" / self.PROCESS_NAME,
            Path.home() / ".local" / "bin" / "cloudflared.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "IraqiVoiceStudio" / "bin" / "cloudflared.exe",
            Path.home() / "AppData" / "Local" / "Programs" / "cloudflared" / "cloudflared.exe",
            Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "cloudflared.exe",
        )
        return next((str(path) for path in candidates if path.is_file()), None)

    def _probe_server(self) -> bool:
        return self._probe_health(f"http://127.0.0.1:{self.port}")

    @staticmethod
    def _probe_health(base_url: str, timeout: float = 4.0) -> bool | None:
        request = urllib.request.Request(
            base_url.rstrip("/") + "/api/health",
            headers={"User-Agent": "rvc-platform-tunnel-health/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status == 200 and json.loads(response.read()).get("ok") is True
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.gaierror):
                return None  # resolver restrictions are distinct from an unhealthy origin
            return False
        except (TimeoutError, ValueError, OSError):
            return False

    @staticmethod
    def _public_hostname_unresolvable(url: str) -> bool:
        hostname = urllib.parse.urlsplit(url).hostname
        if not hostname:
            return False
        try:
            socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
            return False
        except socket.gaierror:
            return True

    @staticmethod
    def _read_log_since(log_path: Path, offset: int) -> list[str]:
        """Read only this process attempt's appended log, never stale URLs."""
        try:
            with log_path.open("rb") as stream:
                stream.seek(offset)
                return stream.read().decode("utf-8", errors="replace").splitlines()
        except OSError:
            return []

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> dict[str, Any]:
        if self.pm.get(self.PROCESS_NAME) and self.pm.get(self.PROCESS_NAME).is_running():
            return self.status()
        if not self.binary_available():
            raise TunnelError(
                "cloudflared is not installed; run the platform installer or "
                "install it from https://developers.cloudflare.com/tunnel/downloads/"
            )
        if not self._probe_server():
            raise TunnelError(f"local server on port {self.port} is not responding")
        if self.auth_enabler:
            self.auth_enabler()  # never expose the dashboard unauthenticated

        for attempt in range(self.retries):
            binary = self._resolve_binary()
            if binary is None:
                raise TunnelError("cloudflared disappeared after preflight")
            proc = self.pm.create(
                self.PROCESS_NAME,
                [binary, "tunnel", "--url", f"http://127.0.0.1:{self.port}",
                 "--no-autoupdate"],
            )
            log_offset = proc.log_path.stat().st_size if proc.log_path.exists() else 0
            proc.start()
            deadline = time.time() + self.timeout
            url = None
            while time.time() < deadline:
                attempt_lines = self._read_log_since(proc.log_path, log_offset)
                for line in attempt_lines:
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
                public_deadline = time.time() + min(30.0, self.timeout)
                public_ready = False
                while time.time() < public_deadline and proc.is_running():
                    public_probe = self._probe_health(url)
                    if public_probe is True:
                        public_ready = True
                        break
                    if (public_probe is None or self._public_hostname_unresolvable(url)) and any(
                        "Registered tunnel connection" in line
                        for line in self._read_log_since(proc.log_path, log_offset)
                    ):
                        logger.warning(
                            "Cloudflare connector is registered, but this runtime cannot resolve the public hostname; "
                            "verify the URL from an external network"
                        )
                        public_ready = True
                        break
                    time.sleep(0.5)
                if not public_ready:
                    logger.warning("public health probe failed for %s", url)
                    self.pm.stop(self.PROCESS_NAME)
                    continue
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
