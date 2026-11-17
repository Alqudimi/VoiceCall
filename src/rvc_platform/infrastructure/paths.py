"""Canonical filesystem layout of the platform.

All runtime artifacts (models, datasets, training runs, caches, logs) live
under a single project-scoped root so that the installation never pollutes
user directories outside it. The root is resolved in this order:

1. ``RVC_PLATFORM_ROOT`` environment variable (explicit user override —
   the only sanctioned way to place artifacts elsewhere, e.g. on Google
   Drive inside Colab);
2. the directory containing the ``pyproject.toml`` of the installed
   editable checkout;
3. the current working directory (when running from a released archive
   that keeps the standard layout).

Selected sub-directories can additionally be redirected individually via
``RVC_MODELS_DIR``, ``RVC_DATA_DIR``, ``RVC_ASSETS_DIR``, ``RVC_RUNS_DIR``,
``RVC_CACHE_DIR`` so users can keep heavy artifacts on separate drives
while the code stays in place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _marker_dirs(p: Path) -> bool:
    return (p / "pyproject.toml").is_file() or (p / "Makefile").is_file()


def _find_project_root() -> Path:
    override = os.environ.get("RVC_PLATFORM_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # walk up from this file (works for editable/src installs)
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if candidate.name == "src":
            root = candidate.parent
            if _marker_dirs(root):
                return root
        if _marker_dirs(candidate):
            return candidate
    return Path.cwd().resolve()


def _redirect(env: str, default: Path) -> Path:
    override = os.environ.get(env)
    if override:
        return Path(override).expanduser().resolve()
    return default


@dataclass(frozen=True)
class Paths:
    root: Path
    assets: Path
    models: Path
    data: Path
    runs: Path
    cache: Path
    logs: Path
    temp: Path

    @property
    def hubert(self) -> Path:
        return self.assets / "hubert_base"

    @property
    def rmvpe(self) -> Path:
        return self.assets / "rmvpe" / "rmvpe.pt"

    @property
    def pretrained(self) -> Path:
        return self.assets / "pretrained_v2"

    @property
    def datasets_raw(self) -> Path:
        return self.data / "raw"

    @property
    def uploads(self) -> Path:
        return self.data / "uploads"

    @property
    def conversions(self) -> Path:
        return self.data / "conversions"

    def ensure(self) -> "Paths":
        for d in (
            self.assets, self.models, self.data, self.runs, self.cache,
            self.logs, self.temp, self.hubert, self.assets / "rmvpe",
            self.pretrained, self.datasets_raw, self.uploads, self.conversions,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self


_paths: Paths | None = None


def get_paths() -> Paths:
    """Return the process-wide :class:`Paths` singleton (lazily built)."""
    global _paths
    if _paths is None:
        root = _find_project_root()
        _paths = Paths(
            root=root,
            assets=_redirect("RVC_ASSETS_DIR", root / "assets"),
            models=_redirect("RVC_MODELS_DIR", root / "models"),
            data=_redirect("RVC_DATA_DIR", root / "data"),
            runs=_redirect("RVC_RUNS_DIR", root / "runs"),
            cache=_redirect("RVC_CACHE_DIR", root / ".cache"),
            logs=root / "logs",
            temp=root / ".tmp",
        )
    return _paths


def reset_paths() -> None:
    """Reset the singleton — used by tests and by the Colab runner."""
    global _paths
    _paths = None
