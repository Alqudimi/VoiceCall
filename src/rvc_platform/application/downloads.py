"""Pretrained asset downloader (HuBERT, RMVPE, v2 base G/D).

Assets are fetched from the official RVC HuggingFace release
(lj1995/VoiceConversionWebUI) with byte-range resume and atomic writes.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable

import httpx

from rvc_platform.core.errors import ConfigurationError
from rvc_platform.infrastructure.config.loader import get_config
from rvc_platform.infrastructure.paths import get_paths

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


def _download_one(url: str, dest: Path, chunk_mb: int, timeout: float,
                  progress: ProgressFn | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    offset = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        with client.stream("GET", url, headers=headers) as resp:
            if offset and resp.status_code != 206:
                offset = 0  # server ignored range -> restart
            total = int(resp.headers.get("content-length", 0)) + offset
            if resp.status_code not in (200, 206):
                raise ConfigurationError(
                    f"download failed ({resp.status_code}) for {url}"
                )
            mode = "ab" if offset else "wb"
            done = offset
            with open(tmp, mode) as fh:
                for chunk in resp.iter_bytes(chunk_mb * 1024 * 1024):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress and total:
                        progress(dest.name, done / total)
    tmp.replace(dest)
    logger.info("downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / (1024 * 1024))


def download_assets(progress: ProgressFn | None = None,
                    only: list[str] | None = None) -> dict:
    """Fetch every configured asset that is missing. Returns a report."""
    cfg = get_config().downloads
    paths = get_paths()
    base = cfg.huggingface_base.rstrip("/")
    wanted = only or list(cfg.files.keys())
    report: dict[str, str] = {}
    failures: list[str] = []
    for remote in wanted:
        local_rel = cfg.files.get(remote, remote)
        dest = paths.assets / local_rel
        if dest.is_file() and dest.stat().st_size > 0:
            report[remote] = f"exists ({dest.stat().st_size // (1024 * 1024)} MB)"
            continue
        url = f"{base}/{remote}"
        try:
            _download_one(url, dest, cfg.chunk_mb, cfg.connect_timeout_s, progress)
            report[remote] = f"downloaded ({dest.stat().st_size // (1024 * 1024)} MB)"
        except Exception as exc:
            logger.exception("asset download failed: %s", remote)
            failures.append(remote)
            report[remote] = f"FAILED: {exc}"
    report["_failures"] = failures  # type: ignore[assignment]
    if failures:
        raise ConfigurationError(f"asset downloads failed: {failures}")
    return report


def asset_status() -> dict:
    paths = get_paths()
    cfg = get_config().downloads
    status: dict[str, dict] = {}
    for remote in cfg.files:
        local = paths.assets / cfg.files[remote]
        status[remote] = {
            "path": str(local),
            "present": local.is_file() and local.stat().st_size > 0,
            "size_mb": round(local.stat().st_size / (1024 * 1024), 1)
            if local.is_file() else 0,
        }
    hubert_ready = (paths.hubert / "config.json").is_file() and (
        paths.hubert / "pytorch_model.bin"
    ).is_file()
    rmvpe_ready = paths.rmvpe.is_file()
    status["_summary"] = {  # type: ignore[assignment]
        "hubert_ready": hubert_ready,
        "rmvpe_ready": rmvpe_ready,
        "pretrained_ready": all(
            (paths.pretrained / f).is_file()
            for f in ("f0G40k.pth", "f0D40k.pth")
        ),
        "all_present": hubert_ready and rmvpe_ready,
    }
    return status
