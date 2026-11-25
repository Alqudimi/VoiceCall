"""File conversion routes (upload -> convert -> download)."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from rvc_platform.api.schemas import ConversionResponse
from rvc_platform.core.errors import ConversionError, ModelNotFoundError

router = APIRouter(prefix="/api/convert", tags=["conversion"])


def _svc(request: Request):
    return request.app.state.services


@router.get("/settings")
def settings(request: Request):
    svc = _svc(request)
    defaults = svc["config"].conversion
    loaded = svc["models"].loaded()
    return {
        "defaults": {
            "pitch_shift": defaults.pitch_shift,
            "f0_method": svc["config"].engine.f0_method,
            "index_rate": defaults.index_rate,
            "filter_radius": defaults.filter_radius,
            "rms_mix_rate": defaults.rms_mix_rate,
            "protect": defaults.protect,
            "denoise": defaults.denoise,
            "normalize": defaults.normalize,
            "output_sample_rate": defaults.output_sample_rate,
        },
        "engine": {
            "status": svc["engine"].status,
            "message": svc["engine"].status_message,
            "loaded_model": loaded.to_dict() if loaded else None,
        },
        "f0_methods": ["rmvpe", "fcpe", "pm"],
    }


@router.post("")
async def convert(
    request: Request,
    file: UploadFile = File(...),
    pitch_shift: int | None = Form(None),
    f0_method: str | None = Form(None),
    index_rate: float | None = Form(None),
    filter_radius: int | None = Form(None),
    rms_mix_rate: float | None = Form(None),
    protect: float | None = Form(None),
    output_sample_rate: int | None = Form(None),
    denoise: bool | None = Form(None),
    normalize: bool | None = Form(None),
):
    svc = _svc(request)
    try:
        stored = svc["conversion"].store_upload(file.filename or "upload.wav", await file.read())
    except ConversionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # absent fields fall back to the server-side configuration defaults
    # (engine.f0_method / conversion.* — no hard-coded client-side values)
    settings = svc["conversion"].build_settings(
        {
            "pitch_shift": pitch_shift,
            "f0_method": f0_method,
            "index_rate": index_rate,
            "filter_radius": filter_radius,
            "rms_mix_rate": rms_mix_rate,
            "protect": protect,
            "output_sample_rate": output_sample_rate,
            "denoise": denoise,
            "normalize": normalize,
        }
    )
    try:
        # run blocking conversion in a worker thread to keep the loop alive
        result = await asyncio.to_thread(
            svc["conversion"].convert, stored, settings, file.filename
        )
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConversionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ConversionResponse(
        output_path=result.output_path.name,
        duration_s=round(result.duration_seconds, 3),
        processing_s=round(result.processing_seconds, 3),
        realtime_factor=round(result.realtime_factor, 3),
        settings={
            "pitch_shift": settings.pitch_shift,
            "f0_method": settings.f0_method,
            "index_rate": settings.index_rate,
            "filter_radius": settings.filter_radius,
            "rms_mix_rate": settings.rms_mix_rate,
            "protect": settings.protect,
            "denoise": settings.denoise,
            "normalize": settings.normalize,
        },
    )


@router.get("/history")
def history(request: Request, limit: int = 50):
    """Recent conversion outputs with their settings and speed metrics."""
    limit = max(1, min(limit, 200))
    return _svc(request)["conversion"].history(limit)


@router.get("/results/{result_id}/download")
def download_result(request: Request, result_id: str):
    try:
        path = _svc(request)["conversion"].result_path(result_id)
    except ConversionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="audio/wav", filename=path.name)
