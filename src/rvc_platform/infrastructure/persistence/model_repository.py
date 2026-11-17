"""Model repository: discovers, registers and inspects RVC voice models.

A "voice model" is an RVC inference checkpoint (``*.pth``) plus an optional
FAISS retrieval index (``*.index``). ``models/model.json`` carries registry
metadata (friendly name, notes, trained-at) that survives re-scans.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from rvc_platform.core.entities import ModelInfo
from rvc_platform.core.errors import ModelNotFoundError, SecurityError
from rvc_platform.infrastructure.paths import get_paths

logger = logging.getLogger(__name__)

ALLOWED_MODEL_SUFFIX = ".pth"
ALLOWED_INDEX_SUFFIX = ".index"


class ModelRepository:
    def __init__(self, models_dir: Path | None = None) -> None:
        self.dir = models_dir or get_paths().models
        self.dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.dir / "registry.json"

    # ---- discovery ----
    def _read_registry(self) -> dict:
        try:
            return json.loads(self._registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"models": {}}

    def _write_registry(self, data: dict) -> None:
        tmp = self._registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._registry_path)

    def _safe_name(self, name: str) -> str:
        """Accept either a stem or a .pth filename; reject path tricks."""
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise SecurityError(f"invalid model name: {name!r}")
        if name.endswith(ALLOWED_MODEL_SUFFIX):
            base = name[: -len(ALLOWED_MODEL_SUFFIX)]
        else:
            base = name
        if not base or any(ch in base for ch in ("\x00",)):
            raise SecurityError(f"invalid model name: {name!r}")
        return base + ALLOWED_MODEL_SUFFIX

    def _probe(self, pth: Path, index: Path | None) -> ModelInfo:
        import torch

        try:
            cpt = torch.load(pth, map_location="cpu", weights_only=False)
            sr = int(cpt.get("sr", 40000))
            f0 = int(cpt.get("f0", 1)) == 1
            version = str(cpt.get("version", "v1"))
            info = str(cpt.get("info", ""))
            del cpt
        except Exception as exc:  # unreadable -> still listed, flagged
            logger.warning("cannot read checkpoint %s: %s", pth, exc)
            sr, f0, version, info = 0, False, "unknown", f"unreadable ({exc})"
        stat = pth.stat()
        return ModelInfo(
            name=pth.stem,
            path=pth,
            sample_rate=sr,
            f0=f0,
            version=version,
            info=info,
            size_mb=stat.st_size / (1024 * 1024),
            created_at=stat.st_mtime,
            index_path=index,
        )

    def _find_index(self, pth: Path) -> Path | None:
        candidates = sorted(self.dir.glob(f"{pth.stem}*.index"))
        return candidates[0] if candidates else None

    def list_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        for pth in sorted(self.dir.glob("*.pth")):
            if pth.name == "registry.json":
                continue
            models.append(self._probe(pth, self._find_index(pth)))
        return models

    def get(self, name: str) -> ModelInfo:
        name = self._safe_name(name)
        pth = self.dir / name
        if not pth.is_file():
            raise ModelNotFoundError(f"model not found: {name}")
        return self._probe(pth, self._find_index(pth))

    # ---- lifecycle ----
    def register(self, source: Path, name: str | None = None) -> ModelInfo:
        """Register a model file (e.g. exported by the trainer) from anywhere
        inside the project root; copies it into models/ and validates it."""
        source = source.resolve()
        if not source.is_file():
            raise ModelNotFoundError(f"source model missing: {source}")
        name = self._safe_name(name or source.name)
        target = (self.dir / name).resolve()
        if source != target:
            shutil.copy2(source, target)
        info = self._probe(target, self._find_index(target))
        registry = self._read_registry()
        registry["models"][target.stem] = {
            "registered_at": time.time(),
            "sample_rate": info.sample_rate,
            "version": info.version,
            "f0": info.f0,
        }
        self._write_registry(registry)
        return info

    def register_index(self, source: Path, model_name: str) -> Path:
        source = source.resolve()
        if not source.is_file() or source.suffix != ALLOWED_INDEX_SUFFIX:
            raise ModelNotFoundError(f"index file missing: {source}")
        self._safe_name(model_name)
        target = self.dir / f"{Path(model_name).stem}.index"
        shutil.copy2(source, target)
        return target

    def delete(self, name: str, with_index: bool = True) -> None:
        name = self._safe_name(name)
        pth = self.dir / name
        if not pth.exists():
            raise ModelNotFoundError(name)
        pth.unlink()
        if with_index:
            for idx in self.dir.glob(f"{Path(name).stem}*.index"):
                idx.unlink()
        registry = self._read_registry()
        registry["models"].pop(Path(name).stem, None)
        self._write_registry(registry)

    def note(self, name: str, notes: str) -> None:
        self._safe_name(name)
        registry = self._read_registry()
        registry["models"].setdefault(Path(name).stem, {})["notes"] = notes
        self._write_registry(registry)
