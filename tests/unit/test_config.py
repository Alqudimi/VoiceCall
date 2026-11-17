"""Configuration layering + validation tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from rvc_platform.core.errors import ConfigurationError
from rvc_platform.infrastructure.config.loader import ConfigLoader
from rvc_platform.infrastructure.config.schema import PlatformConfig


def test_defaults_validate():
    cfg = PlatformConfig()
    assert cfg.server.port == 8320
    assert cfg.engine.f0_method == "rmvpe"
    assert cfg.realtime.block_ms == 160
    assert cfg.training.sample_rate == "40k"


def test_layered_yaml_overrides(project_env, tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text(
        "server:\n  port: 9100\nrealtime:\n  block_ms: 200\n", encoding="utf-8"
    )
    loader = ConfigLoader(user_yaml=user)
    cfg = loader.load()
    assert cfg.server.port == 9100
    assert cfg.realtime.block_ms == 200
    # untouched defaults survive
    assert cfg.conversion.protect == 0.33


def test_env_overrides_win(project_env, monkeypatch):
    monkeypatch.setenv("RVC_SERVER_PORT", "9111")
    monkeypatch.setenv("RVC_BLOCK_MS", "300")
    cfg = ConfigLoader().load()
    assert cfg.server.port == 9111
    assert cfg.realtime.block_ms == 300


def test_invalid_config_rejected(project_env, tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("realtime:\n  block_ms: 999999\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        ConfigLoader(user_yaml=user).load()


def test_invalid_device_rejected():
    with pytest.raises(Exception):
        PlatformConfig(engine={"device": "vulkan"})


def test_write_user_roundtrip(project_env, tmp_path):
    cfg = ConfigLoader().load()
    cfg.server.port = 9200
    target = ConfigLoader().write_user(cfg)
    assert Path(target).is_file()
    reloaded = ConfigLoader().load()
    assert reloaded.server.port == 9200
