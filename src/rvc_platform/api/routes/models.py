"""Model management routes."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from rvc_platform.core.errors import ModelNotFoundError, SecurityError

router = APIRouter(prefix="/api/models", tags=["models"])


def _svc(request: Request):
    return request.app.state.services


# --------------------------------------------------------------------------
# HuggingFace default voice model (env-driven: RVC_HF_REPO / RVC_HF_TOKEN / …)
# --------------------------------------------------------------------------
@router.get("/hf/status")
def hf_status(request: Request):
    """Live status of the default HuggingFace voice model + download progress."""
    service = request.app.state.services.get("huggingface")
    if service is None:
        raise HTTPException(status_code=503, detail="HuggingFace integration not configured")
    return service.status()


@router.post("/hf/download")
def hf_download(request: Request):
    """Start (or report on) the background download of the HF voice model."""
    service = request.app.state.services.get("huggingface")
    if service is None:
        raise HTTPException(status_code=503, detail="HuggingFace integration not configured")
    result = service.start_download(ensure_assets=True)
    busy = result.get("started")
    return JSONResponse(result, status_code=200 if busy else 202)


@router.post("/hf/load")
def hf_load(request: Request):
    """Load the downloaded HF model into the engine as the default model."""
    service = request.app.state.services.get("huggingface")
    if service is None:
        raise HTTPException(status_code=503, detail="HuggingFace integration not configured")
    if not service.model_present():
        raise HTTPException(status_code=409, detail="النموذج غير محمّل بعد — نزّله أولًا")
    try:
        return service.load_as_default()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/default")
def default_model(request: Request):
    """The model the UI should preselect (HF default when present)."""
    service = request.app.state.services.get("huggingface")
    if service is not None:
        status = service.status()
        return {
            "name": status["default_model"],
            "source": "huggingface" if status["model"]["present"] else "repository",
            "recommended": status["recommended"],
        }
    models = _svc(request)["models"].list()
    first_f0 = next((m.name for m in models if m.f0), None)
    return {"name": first_f0, "source": "repository", "recommended": None}


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
