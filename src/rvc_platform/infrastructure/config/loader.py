"""Layered YAML configuration loader with env overrides and write-back."""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from rvc_platform.core.errors import ConfigurationError
from rvc_platform.infrastructure.config.schema import ENV_OVERRIDES, PlatformConfig
from rvc_platform.infrastructure.paths import get_paths

logger = logging.getLogger(__name__)


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Load a ``.env`` file into ``os.environ`` (without clobbering).

    Values already present in the environment always win, so a ``.env`` file
    is a convenience for local runs and never overrides CI/deploy overrides.
    Returns the mapping of *newly applied* variables (handy for logging).
    """
    target = path or (get_paths().root / ".env")
    if not target.is_file():
        return {}
    applied: dict[str, str] = {}
    try:
        for raw_line in target.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if not key:
                continue
            if key not in os.environ:
                os.environ[key] = value
                applied[key] = value
    except OSError as exc:  # unreadable .env must never kill the app
        logger.warning("could not read %s: %s", target, exc)
        return {}
    return applied


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _coerce(value: str) -> Any:
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


class ConfigLoader:
    """Load platform configuration.

    Precedence (lowest -> highest):
        1. code defaults (:class:`PlatformConfig`)
        2. packaged ``configs/default.yaml``
        3. user file: ``$RVC_PLATFORM_CONFIG`` or ``configs/user.yaml``
        4. ``RVC_*`` environment variables
    """

    def __init__(self, default_yaml: Path | None = None, user_yaml: Path | None = None) -> None:
        self._default_yaml = default_yaml or get_paths().root / "configs" / "default.yaml"
        self._user_yaml = user_yaml

    def _user_path(self) -> Path | None:
        if self._user_yaml is not None:
            return self._user_yaml
        env = os.environ.get("RVC_PLATFORM_CONFIG")
        if env:
            return Path(env)
        candidate = get_paths().root / "configs" / "user.yaml"
        return candidate if candidate.is_file() else None

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                raise ConfigurationError(f"{path} must contain a mapping at top level")
            return data
        except FileNotFoundError:
            return {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc

    def load(self) -> PlatformConfig:
        load_dotenv()  # .env convenience layer (never overrides real env vars)
        merged: dict[str, Any] = {}
        if self._default_yaml.is_file():
            merged = _deep_merge(merged, self._read_yaml(self._default_yaml))
        user = self._user_path()
        if user is not None and user.is_file():
            merged = _deep_merge(merged, self._read_yaml(user))
        for env_key, (section, field) in ENV_OVERRIDES.items():
            raw = os.environ.get(env_key)
            if raw is not None:
                merged.setdefault(section, {})
                merged[section][field] = _coerce(raw)
        try:
            return PlatformConfig(**merged)
        except ValidationError as exc:
            raise ConfigurationError(f"invalid configuration: {exc}") from exc

    def write_user(self, config: PlatformConfig) -> Path:
        target = self._user_path() or (get_paths().root / "configs" / "user.yaml")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.loads(config.model_dump_json())
        for env_key, (section, field) in ENV_OVERRIDES.items():
            if os.environ.get(env_key) is not None and section in payload:
                payload.get(section, {}).pop(field, None)
        with open(target, "w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
        return target


_cached: PlatformConfig | None = None


def get_config(reload: bool = False) -> PlatformConfig:
    global _cached
    if _cached is None or reload:
        _cached = ConfigLoader().load()
    return _cached


def reset_config() -> None:
    global _cached
    _cached = None
