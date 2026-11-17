"""Unit tests: tunnel service (fake cloudflared binary, no network)."""
from __future__ import annotations

import os
import stat
import textwrap

import pytest

from rvc_platform.application.tunnel_service import TunnelService
from rvc_platform.core.errors import TunnelError
from rvc_platform.infrastructure.processes.manager import ProcessManager


def _make_fake_binary(path, script: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(script))
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture()
def fake_cloudflared(tmp_path, monkeypatch):
    """A cloudflared stub that prints a trycloudflare URL then sleeps."""
    bin_dir = tmp_path / "bin"
    script = """\
    #!/usr/bin/env bash
    echo "2024-01-01T00:00:00Z INF +--------------------------------------------------------------------------------------------+"
    echo "2024-01-01T00:00:00Z INF |  Your quick Tunnel has been created! Visit it at: https://demo-test-quick.trycloudflare.com     |"
    echo "2024-01-01T00:00:00Z INF +--------------------------------------------------------------------------------------------+"
    sleep 60
    """
    binary = bin_dir / "cloudflared"
    _make_fake_binary(binary, script)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return binary


@pytest.fixture()
def fake_server(tmp_path):
    """A tiny HTTP server standing in for the local web API."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):  # silence
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    thread.join(timeout=5)


def test_binary_detection(fake_cloudflared, tmp_path):
    pm = ProcessManager(tmp_path / "logs")
    svc = TunnelService(pm, 8320)
    assert svc.binary_available()


def test_start_scrapes_url_and_stops(fake_cloudflared, fake_server, tmp_path):
    pm = ProcessManager(tmp_path / "logs")
    auth_calls = []
    svc = TunnelService(pm, fake_server, start_timeout_s=10, auth_enabler=lambda: auth_calls.append(1))
    status = svc.start()
    assert status["running"] is True
    assert status["url"] == "https://demo-test-quick.trycloudflare.com"
    assert auth_calls, "auth must be enforced BEFORE the tunnel goes up"

    stopped = svc.stop()
    assert stopped["running"] is False
    assert svc.status()["running"] is False
    assert svc.status()["url"] is None


def test_start_fails_without_server(fake_cloudflared, tmp_path):
    pm = ProcessManager(tmp_path / "logs")
    svc = TunnelService(pm, server_port=1, retries=1, start_timeout_s=5)  # port 1: nothing listens
    with pytest.raises(TunnelError):
        svc.start()


def test_start_fails_without_binary(tmp_path, monkeypatch):
    # remove cloudflared from PATH
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    (tmp_path / "empty-bin").mkdir()
    pm = ProcessManager(tmp_path / "logs")
    svc = TunnelService(pm, 8320)
    with pytest.raises(TunnelError):
        svc.start()


def test_custom_binary_name(tmp_path):
    pm = ProcessManager(tmp_path / "logs")
    svc = TunnelService(pm, 8320, binary="my-cloudflared-x")
    assert svc.binary == "my-cloudflared-x"
    # not installed -> unavailable
    assert svc.binary_available() is False


def test_failing_tunnel_retries_then_raises(fake_server, tmp_path, monkeypatch):
    """A binary that exits immediately: retries then TunnelError."""
    bin_dir = tmp_path / "bin2"
    script = "#!/usr/bin/env bash\nexit 1\n"
    _make_fake_binary(bin_dir / "cloudflared", script)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    pm = ProcessManager(tmp_path / "logs")
    svc = TunnelService(pm, fake_server, retries=2, start_timeout_s=5)
    with pytest.raises(TunnelError):
        svc.start()
