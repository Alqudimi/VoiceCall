"""HTTP request/response schemas (pydantic v2) — the API contract."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ConversionRequest(BaseModel):
    # form fields alongside the multipart file in /api/convert
    pitch_shift: int = Field(default=0, ge=-24, le=24)
    f0_method: str = Field(default="rmvpe", pattern="^(rmvpe|fcpe|pm)$")
    index_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    filter_radius: int = Field(default=3, ge=0, le=7)
    rms_mix_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    protect: float = Field(default=0.33, ge=0.0, le=0.5)
    output_sample_rate: int | None = Field(default=None, ge=16000, le=192000)
    denoise: bool = False
    normalize: bool = True


class ConversionResponse(BaseModel):
    output_path: str
    duration_s: float
    processing_s: float
    realtime_factor: float
    settings: dict[str, Any]


class TrainingPrepareRequest(BaseModel):
    experiment_name: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,64}$")
    dataset_path: str
    sample_rate: str = "40k"
    version: str = "v2"
    f0: bool = True
    f0_method: str = "rmvpe"
    speaker_id: int = Field(default=0, ge=0, le=109)


class TrainingStartRequest(BaseModel):
    experiment_name: str | None = None
    dataset_path: str | None = None
    sample_rate: str = "40k"
    epochs: int = Field(default=200, ge=1, le=10000)
    batch_size: int = Field(default=6, ge=1, le=64)
    learning_rate: float = 1e-4
    lr_decay: float = 0.999875
    save_every_epoch: int = Field(default=10, ge=1, le=200)
    save_every_weights: bool = True
    save_latest: bool = True
    save_best: bool = True
    val_split: float = Field(default=0.0, ge=0.0, le=0.5)
    cache_in_gpu: bool = False
    preload_base_models: bool = True
    gpu: int | None = None
    f0: bool = True
    f0_method: str = "rmvpe"
    version: str = "v2"
    speaker_id: int = 0
    resume: bool = True
    clean_start: bool = False
    hf_repo_id: str | None = Field(default=None, pattern=r"^[^/\s]+/[^/\s]+$")


class TrainingExportRequest(BaseModel):
    experiment: str
    epoch: int | None = None


class TunnelActionResponse(BaseModel):
    running: bool
    url: str | None = None
    message: str = ""


class SimpleStatus(BaseModel):
    ok: bool
    detail: str = ""
