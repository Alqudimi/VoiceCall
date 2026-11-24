"""HuggingFace default voice-model integration service.

Pulls the configured (possibly private) RVC voice model — by default
``eng7mi/Iraqi-v3`` — from HuggingFace into the local model repository
(``models/<local_name>.pth`` + ``.index``), tracks byte-accurate download
progress for the web UI, and optionally makes it the engine's default
loaded model so the dashboard boots "ready".

All secrets (the HF access token) and the model URL are injected through
environment variables (``RVC_HF_TOKEN`` / ``HF_TOKEN``, ``RVC_HF_MODEL_URL``,
``RVC_HF_REPO`` …) or ``configs/user.yaml`` — never hard-coded, never
returned by the status API (only a boolean ``token_configured``).
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from rvc_platform.application.downloads import _download_one, download_assets
from rvc_platform.infrastructure.config.loader import get_config
from rvc_platform.infrastructure.config.schema import HuggingFaceConfig
from rvc_platform.infrastructure.paths import get_paths

logger = logging.getLogger(__name__)

# inference-critical assets (HuBERT content encoder + RMVPE pitch model);
# the pretrained G/D checkpoints are training-only and intentionally NOT
# pulled here. rmvpe.onnx is only needed by the ONNX extras path.
_INFERENCE_ASSETS = [
    "hubert_base/config.json",
    "hubert_base/preprocessor_config.json",
    "hubert_base/pytorch_model.bin",
    "rmvpe.pt",
]


class HuggingFaceModelService:
    """Download + lifecycle management for the default HuggingFace voice model."""

    def __init__(self, model_service, repository) -> None:
        self.model_service = model_service   # ModelService (engine loader)
        self.repo = repository               # ModelRepository (local registry)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        # ---- progress state (read by the status API from other threads) ----
        self.state: str = "idle"             # idle | downloading | ready | error
        self.error: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.files: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ config
    @property
    def cfg(self) -> HuggingFaceConfig:
        return get_config().huggingface

    def _reset_files(self) -> None:
        paths = get_paths()
        cfg = self.cfg
        self.files = {
            "model": {
                "label": "أوزان النموذج (model.pth)",
                "url": cfg.resolve_url(cfg.model_file),
                "dest": paths.models / f"{cfg.local_name}.pth",
                "status": "pending",
                "received_bytes": 0,
                "total_bytes": 0,
                "error": None,
            },
            "index": {
                "label": "فهرس الاسترجاع (index.index)",
                "url": cfg.resolve_url(cfg.index_file),
                "dest": paths.models / f"{cfg.local_name}.index",
                "status": "pending",
                "received_bytes": 0,
                "total_bytes": 0,
                "error": None,
            },
        }

    # ------------------------------------------------------------------ paths
    def model_path(self) -> Path:
        return get_paths().models / f"{self.cfg.local_name}.pth"

    def index_path(self) -> Path:
        return get_paths().models / f"{self.cfg.local_name}.index"

    def model_present(self) -> bool:
        p = self.model_path()
        return p.is_file() and p.stat().st_size > 0

    def index_present(self) -> bool:
        p = self.index_path()
        return p.is_file() and p.stat().st_size > 0

    def token_configured(self) -> bool:
        return bool(self.cfg.token_from_env())

    def _auth_headers(self) -> dict[str, str]:
        token = self.cfg.token_from_env()
        return {"Authorization": f"Bearer {token}"} if token else {}

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        cfg = self.cfg
        model = self.model_path()
        index = self.index_path()
        loaded = self.model_service.loaded()
        # a default model exists when the HF model is present; otherwise fall
        # back to the first f0 model in the repository so the realtime page
        # always has something sensible to preselect.
        default_model = cfg.local_name if self.model_present() else None
        if default_model is None:
            for info in self.repo.list_models():
                if info.f0:
                    default_model = info.name
                    break
        return {
            "repo_id": cfg.repo_id,
            "local_name": cfg.local_name,
            "model_url": cfg.resolve_url(cfg.model_file),
            "index_url": cfg.resolve_url(cfg.index_file),
            "token_configured": self.token_configured(),
            "auto_download": cfg.auto_download,
            "auto_load": cfg.auto_load,
            "model": {
                "present": self.model_present(),
                "path": str(model),
                "size_mb": round(model.stat().st_size / (1024 * 1024), 1)
                if model.is_file() else 0,
            },
            "index": {
                "present": self.index_present(),
                "path": str(index),
                "size_mb": round(index.stat().st_size / (1024 * 1024), 1)
                if index.is_file() else 0,
            },
            "download": {
                "state": self.state,
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "files": {
                    key: {
                        "label": meta["label"],
                        "url": meta["url"],
                        "status": meta["status"],
                        "received_bytes": meta["received_bytes"],
                        "total_bytes": meta["total_bytes"],
                        "error": meta["error"],
                    }
                    for key, meta in self.files.items()
                },
            },
            "recommended": {
                "index_rate": cfg.recommended_index_rate,
                "f0_method": cfg.recommended_f0_method,
                "rms_mix_rate": cfg.recommended_rms_mix_rate,
                "pitch_shift": 0,
            },
            "default_model": default_model,
            "engine_loaded": loaded.name if loaded else None,
            "is_engine_default": bool(loaded and loaded.name == cfg.local_name),
        }

    # ------------------------------------------------------------------ download
    def download_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_download(self, ensure_assets: bool = True) -> dict[str, Any]:
        """Kick off the background download thread (idempotent)."""
        with self._lock:
            if self.download_active():
                return {"started": False, "message": "download already running"}
            if not self.token_configured():
                return {
                    "started": False,
                    "message": "RVC_HF_TOKEN غير مضبوط — أضف التوكن في .env "
                               "أو كمتغير بيئة ثم أعد المحاولة",
                }
            self._reset_files()
            self.state = "downloading"
            self.error = None
            self.started_at = time.time()
            self.finished_at = None
            self._thread = threading.Thread(
                target=self._download_worker,
                name="hf-model-download",
                daemon=True,
                kwargs={"ensure_assets": ensure_assets},
            )
            self._thread.start()
        return {"started": True, "message": "downloading in background; poll /api/models/hf/status"}

    # ------------------------------------------------------------------ worker
    def _download_worker(self, ensure_assets: bool = True) -> None:
        try:
            if ensure_assets:
                self._ensure_inference_assets()
            self._download_file("model")
            self._download_file("index")
            # register into the local repository (probe + registry metadata)
            try:
                self.repo.register(self.model_path(), f"{self.cfg.local_name}.pth")
            except Exception as exc:  # registry bookkeeping must not fail the run
                logger.warning("model registry registration failed: %s", exc)
            self.state = "ready"
            self.finished_at = time.time()
            logger.info(
                "HuggingFace voice model '%s' ready at %s",
                self.cfg.local_name, self.model_path(),
            )
            if self.cfg.auto_load:
                self.load_as_default()
        except Exception as exc:  # noqa: BLE001 — reported to the UI, never crash
            self.state = "error"
            self.error = str(exc)
            self.finished_at = time.time()
            logger.exception("HuggingFace model download failed")

    def _ensure_inference_assets(self) -> None:
        """Fetch HuBERT + RMVPE if missing (needed for actual conversion)."""
        try:
            missing = [
                remote for remote in _INFERENCE_ASSETS
                if not (get_paths().assets / get_config().downloads.files.get(remote, remote)).is_file()
            ]
            if missing:
                logger.info("fetching %d missing inference asset(s)", len(missing))
                download_assets(only=missing)
        except Exception as exc:
            logger.warning("asset pre-download failed (continuing): %s", exc)

    def _download_file(self, key: str) -> None:
        meta = self.files.get(key) or {}
        if not meta:
            return
        dest: Path = meta["dest"]
        if dest.is_file() and dest.stat().st_size > 0:
            meta["status"] = "skipped"
            meta["received_bytes"] = dest.stat().st_size
            meta["total_bytes"] = dest.stat().st_size
            return
        meta["status"] = "downloading"
        cfg = get_config().downloads

        def _progress(_name: str, fraction: float) -> None:
            meta["received_bytes"] = int(fraction * (meta.get("total_bytes") or 0))

        try:
            # HEAD first for a stable total size (resolve → signed CDN URL)
            import httpx

            with httpx.Client(follow_redirects=True, timeout=cfg.connect_timeout_s,
                              headers=self._auth_headers()) as client:
                head = client.head(meta["url"])
                size = int(head.headers.get("x-linked-size")
                           or head.headers.get("content-length") or 0)
            meta["total_bytes"] = size
            _download_one(
                meta["url"], dest,
                chunk_mb=cfg.chunk_mb, timeout=cfg.connect_timeout_s,
                progress=_progress, headers=self._auth_headers(),
            )
            meta["status"] = "done"
            meta["received_bytes"] = dest.stat().st_size
            meta["total_bytes"] = dest.stat().st_size
        except Exception as exc:
            meta["status"] = "failed"
            meta["error"] = str(exc)
            raise

    # ------------------------------------------------------------------ engine
    def load_as_default(self) -> dict[str, Any]:
        """Load the HF model into the shared engine (dashboard default model)."""
        if not self.model_present():
            return {"loaded": None, "message": "النموذج غير محمّل بعد"}
        info = self.model_service.load(self.cfg.local_name)
        logger.info("default engine model set to '%s'", info.name)
        return {"loaded": info.to_dict()}

    # ------------------------------------------------------------------ startup
    def ensure_default(self) -> None:
        """Startup hook: download if missing, then load as the default model.

        Never raises — every failure degrades to a status the UI can render.
        """
        if self.model_present() and self.index_present():
            if self.cfg.auto_load:
                try:
                    self.load_as_default()
                except Exception:
                    logger.exception("auto-load of default model failed")
            return
        if not self.cfg.auto_download:
            return
        if not self.token_configured():
            logger.warning(
                "HuggingFace default model '%s' missing and RVC_HF_TOKEN is not "
                "set — set it in .env to enable the automatic download",
                self.cfg.local_name,
            )
            return
        logger.info(
            "default voice model '%s' missing — downloading from %s in background",
            self.cfg.local_name, self.cfg.repo_id,
        )
        self.start_download(ensure_assets=True)
