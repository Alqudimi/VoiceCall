"""ULTIMATE AI VOICE OPTIMIZER — REST routes.

All routes are additive; existing endpoints and panels are untouched.

    GET  /api/optimizer/capabilities            — methods, embedders, modes
    GET  /api/optimizer/status                  — live session dashboard state
    GET  /api/optimizer/sessions                — persisted session list
    GET  /api/optimizer/results[?session_id]    — full results (trials/chart)
    GET  /api/optimizer/results/audio           — A/B audio files (wav)
    POST /api/optimizer/reference-samples       — upload a reference sample
    GET  /api/optimizer/reference-samples       — list stored samples
    DELETE /api/optimizer/reference-samples/{sid}
    POST /api/optimizer/analyze-reference        — build the speaker profile
    GET  /api/optimizer/profiles                 — profiles (+reliability)
    DELETE /api/optimizer/profiles/{name}
    POST /api/optimizer/input                    — upload the input audio
    POST /api/optimizer/start                    — start a session
    POST /api/optimizer/pause|resume|cancel      — live controls
    POST /api/optimizer/apply                    — apply best settings to model
    GET  /api/optimizer/models/{name}/applied    — applied settings (convert UI)
    GET  /api/optimizer/report                   — export full JSON report
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from rvc_platform.core.errors import ModelNotFoundError

router = APIRouter(prefix="/api/optimizer", tags=["optimizer"])


def _svc(request: Request):
    return request.app.state.services["optimizer"]


@router.get("/capabilities")
def capabilities(request: Request):
    try:
        return _svc(request).capabilities()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/status")
def status(request: Request):
    return _svc(request).status()


@router.get("/sessions")
def sessions(request: Request):
    return {"sessions": _svc(request).store.list_sessions()}


@router.get("/results")
def results(request: Request, session_id: str | None = None):
    try:
        return _svc(request).results(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/results/audio")
def results_audio(request: Request, session_id: str, name: str):
    try:
        path = _svc(request).audio_path(session_id, name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="audio/wav", filename=path.name)


# ---------------------------------------------------------------- reference
@router.post("/reference-samples")
async def upload_reference(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(...),
    profile_name: str = Form("default"),
):
    try:
        meta = _svc(request).store_reference_sample(
            model_name, profile_name, file.filename or "sample.wav",
            await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"stored": meta}


@router.get("/reference-samples")
def list_reference_samples(request: Request, model_name: str,
                           profile_name: str = "default"):
    return {"samples": _svc(request).list_reference_samples(
        model_name, profile_name)}


@router.get("/reference-samples/audio")
def reference_sample_audio(request: Request, model_name: str,
                           profile_name: str, sample_id: str):
    """Stream one stored reference sample for A/B listening in the panel."""
    import mimetypes


    samples = _svc(request).list_reference_samples(model_name, profile_name)
    match = next((s for s in samples if s["id"] == sample_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="sample not found")
    directory = _svc(request).profiles.samples_dir(model_name, profile_name)
    path = directory / match["file"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="sample file missing")
    media = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(path, media_type=media, filename=path.name)


@router.delete("/reference-samples/{sample_id}")
def delete_reference_sample(request: Request, sample_id: str, model_name: str,
                            profile_name: str = "default"):
    ok = _svc(request).delete_reference_sample(model_name, profile_name,
                                               sample_id)
    if not ok:
        raise HTTPException(status_code=404, detail="sample not found")
    return {"deleted": sample_id}


@router.post("/analyze-reference")
def analyze_reference(request: Request, model_name: str,
                      profile_name: str = "default"):
    try:
        return _svc(request).build_profile(model_name, profile_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/profiles")
def list_profiles(request: Request, model_name: str):
    return {"profiles": _svc(request).list_profiles(model_name)}


@router.delete("/profiles/{profile_name}")
def delete_profile(request: Request, profile_name: str, model_name: str):
    ok = _svc(request).delete_profile(model_name, profile_name)
    if not ok:
        raise HTTPException(status_code=404, detail="profile not found")
    return {"deleted": profile_name}


# ------------------------------------------------------------------- input
@router.post("/input")
async def upload_input(request: Request, file: UploadFile = File(...)):
    try:
        meta = _svc(request).store_input_audio(
            file.filename or "input.wav", await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"stored": meta}


# ------------------------------------------------------------------ control
@router.post("/start")
async def start(request: Request, model_name: str = Form(...),
                input_path: str = Form(...),
                profile_name: str = Form("default"),
                mode: str = Form("advanced"),
                max_trials: int | None = Form(None),
                time_budget_s: float | None = Form(None),
                strategy: str = Form("auto"),
                pitch_policy: str = Form("preserve"),
                pitch_min: int = Form(-6), pitch_max: int = Form(6),
                embedder_prefer: str = Form("auto"),
                denoise: bool = Form(False), normalize: bool = Form(True),
                weights_json: str = Form(""),
                f0_methods_json: str = Form(""),
                validate_top_k: int | None = Form(None)):
    import json as _json

    weights: dict = {}
    f0_methods: list = []
    try:
        if weights_json:
            weights = dict(_json.loads(weights_json))
        if f0_methods_json:
            f0_methods = list(_json.loads(f0_methods_json))
    except _json.JSONDecodeError as exc:
        raise HTTPException(status_code=400,
                            detail=f"invalid JSON field: {exc}") from exc
    config = {
        "model_name": model_name,
        "input_path": input_path,
        "profile_name": profile_name,
        "mode": mode,
        "max_trials": max_trials,
        "time_budget_s": time_budget_s,
        "strategy": strategy,
        "pitch_policy": pitch_policy,
        "pitch_range": (min(pitch_min, pitch_max), max(pitch_min, pitch_max)),
        "embedder_prefer": embedder_prefer,
        "denoise": denoise,
        "normalize": normalize,
        "weights": weights,
        "f0_methods": f0_methods,
        "validate_top_k": validate_top_k,
    }
    try:
        return _svc(request).start(config)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/pause")
def pause(request: Request):
    return _svc(request).pause()


@router.post("/resume")
def resume(request: Request):
    return _svc(request).resume()


@router.post("/cancel")
def cancel(request: Request):
    return _svc(request).cancel()


# ------------------------------------------------------------- apply/report
@router.post("/apply")
def apply_best(request: Request, session_id: str | None = None):
    try:
        return _svc(request).apply_best(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models/{name}/applied")
def applied_settings(request: Request, name: str):
    data = _svc(request).applied_settings(name)
    return {"applied": data}


@router.get("/report")
def export_report(request: Request, session_id: str | None = None):
    try:
        report = _svc(request).export_report(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return report
