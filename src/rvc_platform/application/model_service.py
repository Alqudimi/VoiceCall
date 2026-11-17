"""Model management service: repository + engine lifecycle in one place."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from rvc_platform.core.entities import ModelInfo
from rvc_platform.core.errors import ModelNotFoundError, SecurityError
from rvc_platform.engine.rvc import checkpoint as engine_ckpt
from rvc_platform.infrastructure.audio.pipeline import AudioProcessingEngine
from rvc_platform.infrastructure.persistence.model_repository import ModelRepository
from rvc_platform.engine.rvc.inference import RVCInferenceEngine
from rvc_platform.utils.device import DeviceProfile

logger = logging.getLogger(__name__)


class ModelService:
    """Facade: list/register/delete models; load/unload into the engine."""

    def __init__(self, repository: ModelRepository, engine: RVCInferenceEngine) -> None:
        self.repo = repository
        self.engine = engine

    # ------------------------------------------------------------------ listing
    def list(self) -> list[ModelInfo]:
        return self.repo.list_models()

    def get(self, name: str) -> ModelInfo:
        return self.repo.get(name)

    def info(self, name: str) -> dict[str, Any]:
        info = self.repo.get(name)
        detail = engine_ckpt.inspect_checkpoint(info.path)
        detail.update(
            {
                "name": info.name,
                "index": str(info.index_path) if info.index_path else None,
                "engine_loaded": self.engine.loaded_model is not None
                and self.engine.loaded_model.name == name,
            }
        )
        return detail

    # ------------------------------------------------------------------ upload
    def register_upload(self, source: Path, name: str | None = None) -> ModelInfo:
        """Security-checked registration of an uploaded .pth (+.index nearby)."""
        source = Path(source)
        if source.suffix.lower() != ".pth":
            raise SecurityError("only .pth model files can be registered")
        # validate it is actually an RVC checkpoint before copying anything
        engine_ckpt.inspect_checkpoint(source)
        name = name or source.name
        info = self.repo.register(source, name)
        sibling_index = source.with_suffix(".index")
        if sibling_index.is_file():
            self.repo.register_index(sibling_index, info.name)
            info = self.repo.get(info.name)
        return info

    def register_upload_payload(self, filename: str, payload: bytes) -> ModelInfo:
        """Stream-safe registration of raw upload bytes (API path)."""
        import re
        import uuid

        from rvc_platform.infrastructure.paths import get_paths

        if not payload:
            raise SecurityError("empty upload")
        if len(payload) > 300 * 1024 * 1024:
            raise SecurityError("model upload exceeds 300 MB")
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", filename or "model.pth")
        if not safe.endswith(".pth"):
            raise SecurityError("model uploads must be .pth files")
        tmp_dir = get_paths().temp
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"upload_{uuid.uuid4().hex[:8]}_{safe}"
        tmp.write_bytes(payload)
        return self.register_upload(tmp)

    # ------------------------------------------------------------------ engine
    def load(self, name: str) -> ModelInfo:
        info = self.repo.get(name)
        index_path = info.index_path if info.index_path and info.index_path.is_file() else None
        self.engine.load_model(info.path, index_path)
        return self.engine.loaded_model or info

    def unload(self) -> None:
        self.engine.unload()

    def loaded(self) -> ModelInfo | None:
        return self.engine.loaded_model

    # ------------------------------------------------------------------ delete
    def delete(self, name: str) -> None:
        if self.engine.loaded_model and self.engine.loaded_model.name == name:
            self.engine.unload()
        self.repo.delete(name)
