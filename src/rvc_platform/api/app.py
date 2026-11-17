"""FastAPI application factory — wires the full service graph.

Composition root: the ONLY place concrete implementations are chosen.
Everything downstream depends on interfaces (see core/interfaces.py).
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rvc_platform import __version__
from rvc_platform.api.routes import conversion, models, realtime, system, training, tunnel
from rvc_platform.application.conversion_service import ConversionService
from rvc_platform.application.model_service import ModelService
from rvc_platform.application.realtime_service import RealtimeService
from rvc_platform.application.system_service import SystemService
from rvc_platform.application.tunnel_service import TunnelService
from rvc_platform.application.training_service import TrainingService
from rvc_platform.engine.rvc.inference import RVCInferenceEngine
from rvc_platform.infrastructure.audio.pipeline import AudioProcessingEngine
from rvc_platform.infrastructure.config.loader import get_config
from rvc_platform.infrastructure.logging.setup import setup_logging
from rvc_platform.infrastructure.paths import get_paths
from rvc_platform.infrastructure.persistence.model_repository import ModelRepository
from rvc_platform.infrastructure.processes.manager import ProcessManager
from rvc_platform.infrastructure.security.auth import TokenAuth
from rvc_platform.utils.device import select_device

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    setup_logging()
    config = get_config()
    paths = get_paths().ensure()

    # ---- device & engines ----
    device_index = None
    if config.engine.device.startswith("cuda"):
        device_index = int(config.engine.device.split(":")[1])
    profile = select_device(device_index) if config.engine.device != "cpu" else select_device(None)
    if config.engine.precision == "fp32":
        profile_is_half = False
    else:
        profile_is_half = profile.is_half
    from rvc_platform.utils.device import DeviceProfile

    profile = DeviceProfile(
        device=profile.device, is_half=profile_is_half,
        gpu_name=profile.gpu_name, gpu_memory_gb=profile.gpu_memory_gb, sm=profile.sm,
    )
    logger.info("engine device=%s fp16=%s", profile.device, profile.is_half)

    audio_engine = AudioProcessingEngine()
    inference_engine = RVCInferenceEngine(profile, audio_engine)

    repository = ModelRepository(paths.models)
    model_service = ModelService(repository, inference_engine,
                                 max_upload_mb=config.server.max_upload_mb)
    conversion_service = ConversionService(
        inference_engine, audio_engine, config.conversion,
        engine_cfg=config.engine,               # f0_method default knob
        max_upload_mb=config.server.max_upload_mb,
    )

    process_manager = ProcessManager(paths.logs / "processes")
    training_service = TrainingService(process_manager)

    realtime_service = RealtimeService(profile, repository)
    tunnel_service = TunnelService(
        process_manager, config.server.port, retries=config.tunnel.retries,
        start_timeout_s=config.tunnel.start_timeout_s,
        binary=config.tunnel.binary,            # tunnel.binary knob
    )

    system_service = SystemService(
        inference_engine, profile, training_service, realtime_service, tunnel_service
    )

    token = config.server.effective_token(paths.logs / "auth.token") if config.server.auth_enabled else config.server.auth_token
    auth = TokenAuth(token, enabled=config.server.auth_enabled)
    if config.server.auth_enabled and config.server.auth_token == "":
        logger.warning("auth enabled with auto-generated token — see logs/auth.token")

    def enable_auth_for_tunnel() -> None:
        if not auth.enabled:
            auth.enabled = True
            logger.warning("tunnel active: API authentication is now REQUIRED")

    tunnel_service.auth_enabler = enable_auth_for_tunnel

    app = FastAPI(title="Iraqi Voice Studio", version=__version__, docs_url="/api/docs")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state = {
        "config": config,
        "paths": paths,
        "profile": profile,
        "engine": inference_engine,
        "audio": audio_engine,
        "models": model_service,
        "conversion": conversion_service,
        "training": training_service,
        "realtime": realtime_service,
        "tunnel": tunnel_service,
        "processes": process_manager,
        "system": system_service,
        "auth": auth,
    }
    app.state.services = state

    # ---- auth middleware ----
    # NOTE: returns a 401 response instead of raising HTTPException, because
    # exceptions raised in user middleware bypass the router's handlers and
    # surface as 500s (verified by tests/integration/test_api.py).
    @app.middleware("http")
    async def auth_guard(request: Request, call_next):
        if request.url.path.startswith("/api") and not request.url.path.startswith("/api/health"):
            if auth.enabled and not auth.check_header(
                request.headers.get("Authorization")
            ):
                return JSONResponse(
                    {"detail": "missing or invalid bearer token"}, status_code=401
                )
        return await call_next(request)

    # ---- uniform error envelope for unhandled failures ----
    # per-route errors raise HTTPException already; anything that slips past
    # is caught here so clients always get JSON (never an HTML 500 page).
    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception):
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            {"detail": f"internal error: {exc.__class__.__name__}",
             "path": request.url.path},
            status_code=500,
        )

    app.include_router(system.router)
    app.include_router(models.router)
    app.include_router(conversion.router)
    app.include_router(training.router)
    app.include_router(realtime.router)
    app.include_router(tunnel.router)

    # ---- static web SPA ----
    web_dir = Path(__file__).resolve().parent.parent / "web"
    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(web_dir / "index.html")

        @app.get("/favicon.ico")
        async def favicon() -> JSONResponse:
            return JSONResponse({}, status_code=204)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        try:
            tunnel_service.stop()
        except Exception:
            logger.exception("tunnel stop failed on shutdown")
        try:
            process_manager.stop_all()
        except Exception:
            logger.exception("process cleanup failed on shutdown")
        inference_engine.unload()

    return app
