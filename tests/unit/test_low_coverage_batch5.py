"""Unit tests — batch 5 (final): system_service + tunnel_service coverage."""
from __future__ import annotations


import pytest


class _FakeEngine:
    status = "ready"
    status_message = "ok"

    @property
    def loaded_model(self):
        return None


class _FakeProfile:
    device = "cpu"
    is_half = False
    gpu_name = None


class _FakeRealtime:
    def active(self):
        return 0


class TestSystemService:
    def test_status_full_shape(self, project_env):
        from rvc_platform.application.system_service import SystemService

        svc = SystemService(engine=_FakeEngine(), profile=_FakeProfile(),
                            realtime_service=_FakeRealtime(),
                            training_service=None, tunnel_service=None)
        st = svc.status()
        assert st.app_version
        assert st.engine.value == "ready"
        assert st.conversion_server is True
        assert st.realtime_sessions == 0
        assert st.ram_total_mb > 0
        assert st.python
        # training/tunnel absent -> None fields
        assert st.training is None and st.tunnel is None

    def test_status_with_failing_subservices(self, project_env):
        from rvc_platform.application.system_service import SystemService

        class Boom:
            def status(self):
                raise RuntimeError("down")

        svc = SystemService(engine=_FakeEngine(), profile=_FakeProfile(),
                            realtime_service=_FakeRealtime(),
                            training_service=Boom(), tunnel_service=Boom())
        st = svc.status()
        assert st.training is None and st.tunnel is None  # degraded, not crash


class TestTunnelService:
    def _svc(self, tmp_path, binary="/nonexistent/cloudflared"):
        from rvc_platform.infrastructure.processes.manager import ProcessManager
        from rvc_platform.application.tunnel_service import TunnelService

        pm = ProcessManager(log_dir=tmp_path / "logs")
        return TunnelService(process_manager=pm, server_port=1,
                             binary=binary)

    def test_binary_unavailable(self, tmp_path):
        svc = self._svc(tmp_path)
        assert svc.binary_available() is False

    def test_start_requires_binary(self, tmp_path):
        from rvc_platform.core.errors import TunnelError

        svc = self._svc(tmp_path)
        with pytest.raises(TunnelError):
            svc.start()

    def test_start_requires_live_server(self, tmp_path, monkeypatch):
        from rvc_platform.core.errors import TunnelError

        svc = self._svc(tmp_path, binary="/bin/true")
        monkeypatch.setattr(svc, "binary_available", lambda: True)
        # port 1 -> nothing listens -> probe fails before spawning anything
        with pytest.raises(TunnelError):
            svc.start()

    def test_stop_and_status_idle(self, tmp_path):
        svc = self._svc(tmp_path)
        assert svc.stop() == {"running": False, "clean": False}
        st = svc.status()
        assert st["running"] is False and st["url"] is None

    def test_url_regex(self):
        from rvc_platform.application.tunnel_service import _URL_RE_SEARCH

        line = "2026-11-21 INFO |  https://my-tunnel-abc.trycloudflare.com  |"
        assert _URL_RE_SEARCH(line) == "https://my-tunnel-abc.trycloudflare.com"
        assert _URL_RE_SEARCH("no url here") is None
        assert _URL_RE_SEARCH("https://example.com") is None
