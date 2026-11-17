from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rvc_platform.application.tunnel_service import TunnelService, _URL_RE_SEARCH


def test_quick_tunnel_url_parser_accepts_only_trycloudflare_https():
    assert _URL_RE_SEARCH("https://voice-demo.trycloudflare.com") == "https://voice-demo.trycloudflare.com"
    assert _URL_RE_SEARCH("http://voice-demo.trycloudflare.com") is None
    assert _URL_RE_SEARCH("https://example.com") is None


def test_binary_resolver_finds_user_local_install(monkeypatch, tmp_path):
    bin_dir = tmp_path / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    cloudflared = bin_dir / "cloudflared"
    cloudflared.write_text("binary placeholder", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("rvc_platform.application.tunnel_service.shutil.which", lambda _name: None)
    service = TunnelService(SimpleNamespace(log_dir=tmp_path), 8320)
    assert service.binary_available()
    assert service._resolve_binary() == str(cloudflared.resolve())


def test_local_health_probe_fails_closed_on_unreachable_server(monkeypatch, tmp_path):
    def unavailable(*_args, **_kwargs):
        raise OSError("offline")
    monkeypatch.setattr("rvc_platform.application.tunnel_service.urllib.request.urlopen", unavailable)
    service = TunnelService(SimpleNamespace(log_dir=tmp_path), 8320)
    assert service._probe_server() is False


def test_log_reader_ignores_previous_tunnel_urls(tmp_path):
    log = tmp_path / "cloudflared.log"
    log.write_text("https://old.trycloudflare.com\n", encoding="utf-8")
    offset = log.stat().st_size
    with log.open("a", encoding="utf-8") as stream:
        stream.write("INFO new attempt\nhttps://new.trycloudflare.com\n")
    current = TunnelService._read_log_since(log, offset)
    assert current == ["INFO new attempt", "https://new.trycloudflare.com"]
    assert not any("old.trycloudflare.com" in line for line in current)


def test_start_returns_url_from_current_process_attempt(monkeypatch, tmp_path):
    old_url = "https://old.trycloudflare.com"
    new_url = "https://new.trycloudflare.com"
    log_dir = tmp_path / "cloudflared"
    log_dir.mkdir()
    log_path = log_dir / "cloudflared.log"
    log_path.write_text(f"{old_url}\n", encoding="utf-8")

    class FakeProcess:
        def __init__(self):
            self.log_path = log_path
            self.proc = SimpleNamespace(pid=321)
            self.running = True

        def start(self):
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"created {new_url}\nRegistered tunnel connection\n")

        def tail(self, count=80):
            return log_path.read_text(encoding="utf-8").splitlines()[-count:]

        def is_running(self):
            return self.running

    class FakeManager:
        def __init__(self):
            self.log_dir = tmp_path
            self.process = None

        def get(self, _name):
            return self.process

        def create(self, _name, _cmd):
            self.process = FakeProcess()
            return self.process

        def stop(self, _name):
            if self.process:
                self.process.running = False
            return True

    manager = FakeManager()
    monkeypatch.setattr(TunnelService, "_probe_health", staticmethod(lambda _url: True))
    service = TunnelService(manager, 8320, retries=1)
    monkeypatch.setattr(service, "_resolve_binary", lambda: "cloudflared")
    monkeypatch.setattr(service, "_probe_server", lambda: True)
    result = service.start()
    assert result["running"] is True
    assert result["url"] == new_url
