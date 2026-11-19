"""Training management routes."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from rvc_platform.api.schemas import (
    TrainingExportRequest,
    TrainingPrepareRequest,
    TrainingStartRequest,
)
from rvc_platform.core.errors import DatasetError, TrainingError
from rvc_platform.infrastructure.paths import get_paths

router = APIRouter(prefix="/api/training", tags=["training"])


def _svc(request: Request):
    return request.app.state.services


def _read_json(path: Path) -> dict:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        pass
    return {}


@router.get("/status")
def status(request: Request):
    return _svc(request)["training"].status().to_dict()


@router.get("/runs")
def runs(request: Request):
    runs_root = get_paths().runs
    out = []
    for cfg_path in sorted(runs_root.glob("*/run_config.json")):
        run_dir = cfg_path.parent
        state = _read_json(run_dir / "checkpoints" / "state.json")
        progress = _read_json(run_dir / "progress.json")
        hf_publish = _read_json(run_dir / "huggingface.json")
        best = _read_json(run_dir / "checkpoints" / "best.json")
        run_cfg = _read_json(cfg_path)
        out.append(
            {
                "experiment": run_dir.name,
                "run_dir": str(run_dir),
                "epoch": state.get("epoch") or progress.get("epoch", 0),
                "global_step": state.get("global_step") or progress.get("global_step", 0),
                "updated_at": max(
                    state.get("saved_at", 0), progress.get("ts", 0),
                    cfg_path.stat().st_mtime,
                ),
                "has_checkpoints": (run_dir / "checkpoints").is_dir()
                and any((run_dir / "checkpoints").glob("G_*.pth")),
                "hf_repo_id": hf_publish.get("repo_id") or run_cfg.get("hf_repo_id"),
                "hf_url": hf_publish.get("url"),
                "best": {
                    key: best.get(key)
                    for key in ("epoch", "global_step", "metric", "score",
                                "improved_at")
                } if best else None,
                "hf_best": hf_publish.get("best"),
            }
        )
    return out


@router.post("/prepare")
def prepare(request: Request, payload: TrainingPrepareRequest):
    try:
        config = _svc(request)["training"].make_config(
            **payload.model_dump(exclude_none=True)
        )
        job_id = _svc(request)["training"].prepare_dataset(config)
        return {"job": job_id, "experiment": config.experiment_name}
    except (TrainingError, DatasetError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/prepare/status")
def prepare_status(request: Request, experiment: str = ""):
    return _svc(request)["training"].prep_status(experiment)


@router.post("/start")
def start(request: Request, payload: TrainingStartRequest):
    data = payload.model_dump(exclude_none=True)
    experiment = data.pop("experiment_name", None)
    if experiment:
        run_dir = get_paths().runs / experiment
        if (run_dir / "run_config.json").is_file():
            # existing run -> resume unless a fresh config is requested
            from rvc_platform.training.run_config import TrainingRunConfig

            config = TrainingRunConfig.load(run_dir)
            for key, value in data.items():
                if hasattr(config, key) and value is not None:
                    setattr(config, key, value)
            config.experiment_name = experiment
        else:
            data["experiment_name"] = experiment
            config = _svc(request)["training"].make_config(**data)
    else:
        config = _svc(request)["training"].make_config(**data)
    if "hf_repo_id" in payload.model_fields_set:
        config.hf_repo_id = payload.hf_repo_id
    try:
        return _svc(request)["training"].start(config, resume=config.resume)
    except (TrainingError, DatasetError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/stop")
def stop(request: Request):
    return _svc(request)["training"].stop()


@router.post("/resume")
def resume(request: Request, experiment: str | None = None,
           checkpoint: str | None = None):
    """Resume the latest run — or an explicit checkpoint (name, path, 'latest')."""
    try:
        return _svc(request)["training"].resume(experiment, checkpoint)
    except TrainingError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{experiment}/checkpoints")
def checkpoints(request: Request, experiment: str):
    try:
        return _svc(request)["training"].checkpoints(experiment)
    except TrainingError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{experiment}/logs", response_class=PlainTextResponse)
def logs(request: Request, experiment: str, lines: int = 200):
    return "\n".join(_svc(request)["training"].logs(experiment, lines))


@router.get("/runs/{experiment}/events")
def events(request: Request, experiment: str, tail: int = 100):
    path = get_paths().runs / experiment / "events.jsonl"
    if not path.is_file():
        return {"events": []}
    return {"events": path.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]}


@router.post("/export")
def export(request: Request, payload: TrainingExportRequest):
    try:
        return _svc(request)["training"].export(
            payload.experiment, payload.epoch,
            model_repo=_svc(request)["models"].repo,
        )
    except TrainingError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
