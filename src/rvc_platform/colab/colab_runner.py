"""Colab/Drive persistence utilities.

In Colab the VM is ephemeral. Strategy (documented in INSTALLATION.md):

* ``runs/``    -> redirected to Drive (RVC_RUNS_DIR): every checkpoint is
                   written straight to Drive, so an interrupted session
                   loses at most ``save_every_epoch`` epochs;
* ``models/``  -> redirected to Drive: exported models survive;
* ``assets/``  -> ephemeral (large, re-downloadable; cached per session).

``restore()`` pulls an existing Drive workspace into the session when the
redirect was not applied, ``snapshot()`` pushes local artifacts to Drive on
demand (belt-and-braces before a session ends), ``status()`` reports what
would be restored.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def drive_root(drive_dir: str = "RVC_Iraqi") -> Path | None:
    base = Path("/content/drive/MyDrive")
    if not base.is_dir():
        return None
    root = base / drive_dir
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sync(src: Path, dst: Path, label: str) -> dict:
    copied = 0
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.glob("*"):
            if item.is_dir():
                shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
                copied += 1
            elif item.is_file() and not item.name.endswith(".part"):
                shutil.copy2(item, dst / item.name)
                copied += 1
    logger.info("synced %s items (%s): %s -> %s", copied, label, src, dst)
    return {"items": copied, "from": str(src), "to": str(dst)}


def restore(drive_dir: str = "RVC_Iraqi") -> dict:
    """Pull Drive artifacts into the local project (used when redirects are
    not active, e.g. plain bash cells)."""
    root = drive_root(drive_dir)
    if root is None:
        return {"restored": False, "reason": "Google Drive not mounted"}
    from rvc_platform.infrastructure.paths import get_paths

    p = get_paths()
    report = {"restored": True, "runs": None, "models": None}
    if (root / "runs").is_dir():
        report["runs"] = _sync(root / "runs", p.runs, "runs")
    if (root / "models").is_dir():
        report["models"] = _sync(root / "models", p.models, "models")
    return report


def snapshot(drive_dir: str = "RVC_Iraqi") -> dict:
    """Push local runs/models to Drive (call before a session ends)."""
    root = drive_root(drive_dir)
    if root is None:
        return {"snapshot": False, "reason": "Google Drive not mounted"}
    from rvc_platform.infrastructure.paths import get_paths

    p = get_paths()
    report = {"snapshot": True, "runs": None, "models": None, "ts": time.time()}
    if p.runs.is_dir():
        report["runs"] = _sync(p.runs, root / "runs", "runs")
    if p.models.is_dir():
        report["models"] = _sync(p.models, root / "models", "models")
    return report


def status(drive_dir: str = "RVC_Iraqi") -> dict:
    root = drive_root(drive_dir)
    if root is None:
        return {"drive": False, "hint": "mount Drive first (google.colab.drive)"}
    runs = sorted((root / "runs").glob("*/checkpoints/G_*.pth")) if (root / "runs").is_dir() else []
    return {
        "drive": True,
        "root": str(root),
        "redirects": {
            "runs": os.environ.get("RVC_RUNS_DIR"),
            "models": os.environ.get("RVC_MODELS_DIR"),
            "root": os.environ.get("RVC_PLATFORM_ROOT"),
        },
        "checkpoints": [str(c) for c in runs[-10:]],
    }


def colab_env(drive_dir: str = "RVC_Iraqi") -> dict:
    """Recommended env wiring for Colab (what install_colab.sh writes)."""
    root = drive_root(drive_dir)
    wiring = {
        "RVC_RUNS_DIR": str(root / "runs") if root else None,
        "RVC_MODELS_DIR": str(root / "models") if root else None,
        "RVC_DATA_DIR": "/content/iraqi_data",
    }
    return wiring


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Colab/Drive utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("restore", "snapshot", "status", "env"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    if args.cmd == "restore":
        print(restore())
    elif args.cmd == "snapshot":
        print(snapshot())
    elif args.cmd == "status":
        print(status())
    else:
        print(colab_env())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
