"""Colab/Drive persistence utilities.

In Colab the VM is ephemeral. Strategy (documented in INSTALLATION.md):

* ``runs/`` and ``models/`` -> kept on the fast local VM during training;
                   ``snapshot()`` copies completed artifacts to Drive;
* ``assets/``  -> ephemeral (large, re-downloadable; cached per session).

``restore()`` pulls an existing Drive workspace into the session when the
redirect was not applied, ``snapshot()`` pushes local artifacts to Drive on
demand (belt-and-braces before a session ends), ``status()`` reports what
would be restored.
"""
from __future__ import annotations

import logging
import hashlib
import json
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
                target = dst / item.name
                partial = target.with_name(f".{target.name}.part")
                shutil.copy2(item, partial)
                os.replace(partial, target)
                copied += 1
    logger.info("synced %s items (%s): %s -> %s", copied, label, src, dst)
    return {"items": copied, "from": str(src), "to": str(dst)}


def _artifact_manifest(root: Path) -> dict:
    """Describe completed checkpoint/model files without hashing huge datasets."""
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".part"):
            continue
        if not (path.suffix in {".pth", ".index", ".json"} or path.name == "state.json"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                      "sha256": digest.hexdigest()})
    return {"version": 1, "files": files}


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
    manifest = {
        "version": 1,
        "runs": _artifact_manifest(root / "runs") if (root / "runs").is_dir() else None,
        "models": _artifact_manifest(root / "models") if (root / "models").is_dir() else None,
    }
    manifest_path = root / "snapshot_manifest.json"
    partial = manifest_path.with_name(".snapshot_manifest.json.part")
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(partial, manifest_path)
    report["manifest"] = str(manifest_path)
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
