"""Realtime service limits: session cap + idle reaping (config knobs wired)."""
from __future__ import annotations

import time

import pytest

from rvc_platform.application.realtime_service import RealtimeService, RealtimeSession
from rvc_platform.core.errors import RealtimeError
from rvc_platform.infrastructure.config.schema import RealtimeConfig
from rvc_platform.infrastructure.persistence.model_repository import ModelRepository
from rvc_platform.utils.device import DeviceProfile

pytestmark = pytest.mark.slow


class _NoRepo(ModelRepository):
    def __init__(self):  # skip parent init (no dirs needed)
        self.models_dir = None


def _service(tmp_path, max_sessions=2, keep_alive=60.0, f0="rmvpe"):
    profile = DeviceProfile("cpu", False, None, 0.0, 0.0)
    import rvc_platform.infrastructure.config.loader as cfg_loader
    import rvc_platform.application.realtime_service as rs_mod

    original_get = cfg_loader.get_config

    def patched():
        cfg = original_get()
        cfg.realtime.max_sessions = max_sessions
        cfg.realtime.keep_alive_s = keep_alive
        cfg.realtime.f0_method = f0
        return cfg

    # realtime_service imported the symbol into its own namespace at import time
    rs_mod.get_config = patched
    try:
        svc = RealtimeService(profile, _NoRepo())
    finally:
        rs_mod.get_config = original_get
    return svc


def test_session_cap_enforced(project_env, tmp_path):
    svc = _service(tmp_path, max_sessions=2)
    s1 = svc.new_session()
    s2 = svc.new_session()
    with pytest.raises(RealtimeError):
        svc.new_session()
    svc.close(s1)
    svc.close(s2)


def test_idle_sessions_are_reaped(project_env, tmp_path):
    svc = _service(tmp_path, max_sessions=2, keep_alive=5.0)
    s1 = svc.new_session()
    s2 = svc.new_session()  # would exceed cap if s1 not reaped
    # age s1 beyond keep-alive
    s1.last_seen = time.time() - 3600
    s2.last_seen = time.time() - 3600
    s3 = svc.new_session()  # triggers reaper -> both idle dropped
    assert s1.id not in svc.sessions
    assert s2.id not in svc.sessions
    svc.close(s3)


def test_session_engine_uses_config_f0_method(project_env, tmp_path):
    svc = _service(tmp_path, f0="fcpe")
    session = svc.new_session()
    assert session.engine.f0_method == "fcpe", "realtime.f0_method config must reach the engine"


def test_realtime_config_bounds():
    cfg = RealtimeConfig()
    assert cfg.max_sessions == 2
    assert cfg.keep_alive_s == 60.0
    # lookback upper bound aligned with upstream (2500 ms)
    widened = RealtimeConfig(lookback_ms=2500)
    assert widened.lookback_ms == 2500
