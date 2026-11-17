"""System / assets routes."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rvc_platform.api.schemas import SimpleStatus

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/system")
def system_status(request: Request):
    services = request.app.state.services
    return services["system"].status().to_dict()


@router.get("/system/detect")
def system_detect(request: Request):
    services = request.app.state.services
    return services["system"]._system_profile().summary


@router.get("/health")
def health():
    return SimpleStatus(ok=True, detail="Iraqi Voice Studio API is alive")


@router.get("/assets")
def assets(request: Request):
    from rvc_platform.application.downloads import asset_status

    return asset_status()


@router.post("/assets/download")
def assets_download(request: Request):
    """Fetch missing pretrained assets in the background."""
    import threading

    from rvc_platform.application.downloads import download_assets

    state = request.app.state.services

    if state.get("download_thread") and state["download_thread"].is_alive():
        return JSONResponse({"started": False, "message": "download already running"}, 202)

    def _run() -> None:
        try:
            download_assets(progress=lambda name, frac: None)
        except Exception:
            import logging

            logging.getLogger(__name__).exception("asset download failed")

    thread = threading.Thread(target=_run, name="asset-download", daemon=True)
    state["download_thread"] = thread
    thread.start()
    return {"started": True, "message": "downloading in background; poll /api/assets"}


@router.get("/logs")
def logs(request: Request, lines: int = 200):
    from rvc_platform.infrastructure.paths import get_paths

    log_file = get_paths().logs / "platform.log"
    if not log_file.is_file():
        return {"lines": []}
    return {"lines": log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]}
