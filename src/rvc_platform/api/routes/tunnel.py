"""Cloudflare tunnel routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from rvc_platform.api.schemas import TunnelActionResponse
from rvc_platform.core.errors import TunnelError

router = APIRouter(prefix="/api/tunnel", tags=["tunnel"])


def _svc(request: Request):
    return request.app.state.services


@router.post("/start")
def start(request: Request):
    try:
        status = _svc(request)["tunnel"].start()
        return TunnelActionResponse(
            running=status["running"], url=status["url"],
            message="tunnel active — auth token is now required for API access",
        )
    except TunnelError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/stop")
def stop(request: Request):
    return _svc(request)["tunnel"].stop()


@router.get("/status")
def status(request: Request):
    return _svc(request)["tunnel"].status()


@router.get("/logs")
def logs(request: Request, lines: int = 60):
    svc = _svc(request)
    proc = svc["processes"].get("cloudflared")
    return {"lines": proc.tail(lines) if proc else []}
