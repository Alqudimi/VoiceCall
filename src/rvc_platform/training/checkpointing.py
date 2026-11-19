"""Checkpoint I/O with *complete* resumable state.

Upstream RVC saves {model, optimizer, learning_rate, iteration(epoch)} which
loses scheduler / scaler / global-step / RNG state. We keep byte-compatibility
(read those checkpoints for resume) and add our own richer format:

    checkpoints/
        G_<step>.pth   {"model", "optimizer", "scheduler", "scaler",
                        "epoch", "global_step", "rng"}          (step snapshot)
        D_<step>.pth   same for the discriminator
        G_2333333.pth / D_2333333.pth    always-latest mirrors
        state.json     {epoch, global_step, dataset_signature,
                        run_config, losses tail, wall_clock, last_save, best}
        best.json      best-so-far model metadata (see best_model.py)

Resume policy (CheckpointManager.resolve):
    1. explicit checkpoint requested -> use it;
    2. state.json exists and matches run config + dataset signature -> latest;
    3. upstream-style G_/D_ present -> legacy resume (epoch from 'iteration');
    4. nothing -> fresh start with optional pretrained base.

I/O model: the trainer thread only *snapshots* live tensor state to CPU
(``build_full_payload``) and hands the payload to the background I/O worker
(``write_checkpoint_files``); serialization, mirror refreshes and the
state.json update run off the training thread so the loop never waits on
disk.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from rvc_platform.core.entities import CheckpointInfo
from rvc_platform.core.errors import CheckpointError
from rvc_platform.training.background_io import snapshot_to_cpu

logger = logging.getLogger(__name__)

LATEST_TAG = 2333333


def _torch_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(".pth.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def build_full_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
) -> dict:
    """Snapshot the complete resume state to private CPU copies.

    Safe to hand to a background thread: training may keep mutating the live
    (GPU) tensors while the payload is serialized elsewhere. This is the only
    part of a checkpoint save that must run on the training thread.
    """
    return {
        "model": snapshot_to_cpu(model.state_dict()),
        "optimizer": snapshot_to_cpu(optimizer.state_dict()) if optimizer is not None else None,
        "scheduler": snapshot_to_cpu(scheduler.state_dict()) if scheduler is not None else None,
        "scaler": snapshot_to_cpu(scaler.state_dict()) if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "rng": snapshot_to_cpu(torch.get_rng_state()),
        "saved_at": time.time(),
    }


def write_checkpoint_files(
    payload_g: dict,
    g_path: Path,
    payload_d: dict,
    d_path: Path,
    state: dict | None,
    state_path: Path,
    mirror: bool,
    payload_d2: dict | None = None,
    d2_path: Path | None = None,
) -> None:
    """Write a G/D checkpoint pair (+ optional always-latest mirrors + state).

    ``payload_d2`` / ``d2_path`` (optional): the multi-scale discriminator
    (D2) rides in its own D2_<step>.pth with its own always-latest mirror —
    G/D file formats stay byte-compatible with upstream tooling.

    Meant to run on the background I/O writer: atomic per file, and ordered —
    state.json is only touched after both checkpoints are on disk, so a crash
    mid-write can never advertise a checkpoint that does not exist.
    """
    g_path, d_path = Path(g_path), Path(d_path)
    _torch_save(payload_g, g_path)
    _torch_save(payload_d, d_path)
    if mirror:
        write_latest_mirrors(g_path.parent, g_path, d_path)
    if payload_d2 is not None and d2_path is not None:
        d2_path = Path(d2_path)
        _torch_save(payload_d2, d2_path)
        if mirror:
            d2_mirror = d2_path.parent / f"D2_{LATEST_TAG}.pth"
            tmp = d2_mirror.with_suffix(".pth.tmp")
            try:
                shutil.copyfile(d2_path, tmp)
                tmp.replace(d2_mirror)
            except OSError:
                logger.warning("could not refresh D2 mirror", d2_mirror,
                               exc_info=True)
    if state is not None:
        write_state(state, state_path)


def save_full(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
) -> None:
    """Synchronous full-state save (kept for tooling/tests; the trainer uses
    build_full_payload + write_checkpoint_files on the background worker)."""
    _torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "rng": torch.get_rng_state(),
            "saved_at": time.time(),
        },
        path,
    )


def load_full(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
              scaler: Any | None = None) -> dict:
    """Load a full-state checkpoint into live objects. Returns metadata.

    Metadata additionally carries the checkpoint-embedded optional states —
    ``ema`` (generator EMA shadow, from the G payload) and ``msd`` (the D2
    multi-scale discriminator state, from the D payload) — so callers can wire
    the quality levers without loading the payloads twice. Both default to
    None on checkpoints written before those features existed.
    """
    if not Path(path).is_file():
        raise CheckpointError(f"checkpoint missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise CheckpointError(f"{path} has no 'model' entry")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    if payload.get("rng") is not None:
        torch.set_rng_state(payload["rng"].to(torch.uint8))
    return {
        "epoch": int(payload.get("epoch", payload.get("iteration", 1)) or 1),
        "global_step": int(payload.get("global_step", 0) or 0),
        "ema": payload.get("ema"),
        "msd": payload.get("msd"),
    }


def legacy_iteration_epoch(path: Path) -> int:
    """Epoch number of an upstream-style checkpoint."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("iteration", 1) or 1)


def write_state(state: dict, path: Path) -> None:
    tmp = Path(path).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def read_state(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_latest_mirrors(checkpoint_dir: Path, g_path: Path, d_path: Path) -> None:
    """Refresh the upstream-compatible always-latest mirrors (atomic copy)."""
    checkpoint_dir = Path(checkpoint_dir)
    for src, tag in ((g_path, "G"), (d_path, "D")):
        dst = checkpoint_dir / f"{tag}_{LATEST_TAG}.pth"
        tmp = dst.with_suffix(".pth.tmp")
        try:
            shutil.copyfile(src, tmp)
            tmp.replace(dst)
        except OSError:
            logger.warning("could not refresh %s mirror", dst, exc_info=True)


def list_checkpoints(checkpoint_dir: Path) -> list[CheckpointInfo]:
    checkpoint_dir = Path(checkpoint_dir)
    out: list[CheckpointInfo] = []
    for path in sorted(checkpoint_dir.glob("G_*.pth")):
        stem = path.stem
        kind = "G-latest" if stem == "G_2333333" else ("G-final" if stem == "G_final" else "G")
        epoch, step = _probe_epoch(path)
        out.append(
            CheckpointInfo(
                run_id=checkpoint_dir.parent.name,
                path=path,
                kind=kind,
                epoch=epoch,
                global_step=step,
                created_at=path.stat().st_mtime,
                size_mb=path.stat().st_size / (1024 * 1024),
            )
        )
    return out


def _probe_epoch(path: Path) -> tuple[int, int]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return int(payload.get("epoch", payload.get("iteration", 0)) or 0), int(
            payload.get("global_step", 0) or 0
        )
    except Exception:
        return 0, int(re.sub(r"\D", "", path.stem) or 0)


def latest_step_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Newest *numbered* G_<step>.pth, or G_final.pth if it is at least as far.

    ``G_final`` is written at completion with the highest global_step seen;
    a numbered ``G_<step>.pth`` only becomes newer when training continued
    past a final save (resume with a higher ``total_epoch``) and saved again.
    We probe G_final's embedded ``global_step`` (a single small load at
    startup) and prefer it when it is >= the highest numbered step.
    The always-latest mirror ``G_2333333`` is excluded — ``_restore``
    compares it by mtime.
    """
    checkpoint_dir = Path(checkpoint_dir)
    numbered = [p for p in checkpoint_dir.glob("G_*.pth")
                if p.stem != f"G_{LATEST_TAG}" and p.stem != "G_final"]
    final = checkpoint_dir / "G_final.pth"

    def digits(p: Path) -> int:
        d = re.sub(r"\D", "", p.stem)
        return int(d) if d else 0

    if not numbered:
        return final if final.is_file() else None
    best_numbered = max(numbered, key=digits)
    if final.is_file():
        try:
            payload = torch.load(final, map_location="cpu", weights_only=False)
            final_step = int(payload.get("global_step", 0) or 0)
        except Exception:
            final_step = 0
        if final_step >= digits(best_numbered):
            return final
    return best_numbered
