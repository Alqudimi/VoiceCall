"""Unit/integration: conversion service (config-driven f0, caps, exact results)."""
from __future__ import annotations

import pytest

from rvc_platform.application.conversion_service import ConversionService
from rvc_platform.core.errors import ConversionError
from rvc_platform.infrastructure.config.schema import EngineConfig


class _Defaults:
    pitch_shift = 0
    index_rate = 0.0
    filter_radius = 3
    rms_mix_rate = 0.25
    protect = 0.33
    denoise = False
    normalize = True
    output_sample_rate = None


class _FakeEngine:
    net_g = None  # no model loaded


def _svc(engine_cfg=None, max_mb=40):
    return ConversionService(_FakeEngine(), None, _Defaults(),
                             engine_cfg=engine_cfg, max_upload_mb=max_mb)


def test_default_f0_method_comes_from_engine_config():
    svc = _svc(engine_cfg=EngineConfig(f0_method="fcpe"))
    assert svc.default_f0_method == "fcpe"
    settings = svc.build_settings({})
    assert settings.f0_method == "fcpe", "absent client field must fall back to config"


def test_explicit_f0_method_wins_over_config():
    svc = _svc(engine_cfg=EngineConfig(f0_method="fcpe"))
    settings = svc.build_settings({"f0_method": "pm"})
    assert settings.f0_method == "pm"


def test_rmvpe_fallback_without_engine_cfg():
    assert _svc().default_f0_method == "rmvpe"


def test_upload_cap_wired_from_config():
    svc = _svc(max_mb=1)
    with pytest.raises(ConversionError):
        svc.store_upload("big.wav", b"x" * (1024 * 1024 + 1))


def test_upload_rejects_disallowed_suffix():
    svc = _svc()
    with pytest.raises(ConversionError):
        svc.store_upload("evil.exe", b"MZ.....")


def test_upload_rejects_empty():
    svc = _svc()
    with pytest.raises(ConversionError):
        svc.store_upload("empty.wav", b"")


def test_result_lookup_is_exact_not_substring(project_env, tmp_path):
    svc = _svc()
    conv = project_env / "data" / "conversions"
    conv.mkdir(parents=True, exist_ok=True)
    # a legitimate result + a decoy sharing a substring of its id
    good = conv / "1718000000_aaa111_target.wav"
    good.write_bytes(b"good")
    decoy = conv / "9999999999_aaa111x_target.wav"
    decoy.write_bytes(b"decoy")

    # registry path (in-memory map)
    svc._results[good.stem] = good.name
    found = svc.result_path(good.stem)
    assert found.read_bytes() == b"good"

    # exact-name path (server restart: registry empty)
    svc2 = _svc()
    found2 = svc2.result_path(good.name)
    assert found2.read_bytes() == b"good"

    # a substring crafted id must NOT resolve to the decoy
    with pytest.raises(ConversionError):
        svc2.result_path("aaa111")
    with pytest.raises(ConversionError):
        svc2.result_path("../conversions")
