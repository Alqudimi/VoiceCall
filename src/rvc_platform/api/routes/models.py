"""Model management routes."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from rvc_platform.core.errors import ModelNotFoundError, SecurityError

router = APIRouter(prefix="/api/models", tags=["models"])


def _svc(request: Request):
    return request.app.state.services


@router.get("")
def list_models(request: Request):
    return [m.to_dict() for m in _svc(request)["models"].list()]


@router.get("/{name}")
def model_info(request: Request, name: str):
    try:
        return _svc(request)["models"].info(name)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{name}/load")
def load_model(request: Request, name: str):
    try:
        info = _svc(request)["models"].load(name)
        return {"loaded": info.to_dict(), "engine": _svc(request)["engine"].status}
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/unload")
def unload(request: Request):
    _svc(request)["models"].unload()
    return {"loaded": None}


@router.post("/upload")
async def upload_model(request: Request, file: UploadFile = File(...)):
    """Register an uploaded RVC .pth (optionally followed by its .index)."""
    payload = await file.read()
    try:
        info = _svc(request)["models"].register_upload_payload(file.filename, payload)
    except SecurityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid model file: {exc}") from exc
    return {"registered": info.to_dict()}


@router.delete("/{name}")
def delete_model(request: Request, name: str):
    try:
        _svc(request)["models"].delete(name)
        return {"deleted": name}
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{name}/download")
def download_model(request: Request, name: str):
    from fastapi.responses import FileResponse

    try:
        info = _svc(request)["models"].get(name)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(info.path, filename=info.path.name, media_type="application/octet-stream")
