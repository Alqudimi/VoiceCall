"""Typed application configuration (pydantic v2 models).

Layering: built-in defaults -> ``configs/default.yaml`` -> user YAML
(``RVC_PLATFORM_CONFIG`` or ``configs/user.yaml``) -> environment overrides
(``RVC_*``). Every section maps to a real consumer — there are no dead knobs.
"""
from __future__ import annotations

import os
import secrets
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8320, ge=1024, le=65535)
    auth_enabled: bool = False
    # when empty at runtime and auth_enabled=True a fresh token is generated,
    # printed to the terminal and written to logs/auth.token (never to git).
    auth_token: str = ""
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_upload_mb: int = Field(default=200, ge=1, le=2048)

    def effective_token(self, token_file: "os.PathLike | str") -> str:
        if self.auth_token:
            return self.auth_token
        token = secrets.token_urlsafe(24)
        with open(token_file, "a", encoding="utf-8") as fh:
            fh.write(token + "\n")
        return token


class EngineConfig(BaseModel):
    device: str = "auto"                      # "auto" | "cpu" | "cuda:0" ...
    f0_method: Literal["rmvpe", "fcpe", "pm"] = "rmvpe"
    precision: Literal["auto", "fp16", "fp32"] = "auto"
    cuda_graph: bool = False                  # capture/replay fast path (CUDA only)

    @field_validator("device")
    @classmethod
    def _device_shape(cls, v: str) -> str:
        v = v.strip().lower()
        if v in ("auto", "cpu") or v.startswith("cuda"):
            return v
        raise ValueError("device must be 'auto', 'cpu' or 'cuda:<index>'")


class ConversionDefaults(BaseModel):
    pitch_shift: int = Field(default=0, ge=-24, le=24)
    # official RVC default (webui.py): 0.75 — retrieval blending toward the
    # training-set timbre; 0.0 disabled it entirely (the "far from the
    # speaker" complaint). See RVC FAQ Q11 on index rate / timbre leakage.
    index_rate: float = Field(default=0.75, ge=0.0, le=1.0)
    filter_radius: int = Field(default=0, ge=0, le=7)   # official default: median filter OFF (legacy knob: >2 = kernel-3)
    rms_mix_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    protect: float = Field(default=0.33, ge=0.0, le=0.5)
    denoise: bool = False
    normalize: bool = True
    output_sample_rate: int | None = None


class RealtimeConfig(BaseModel):
    block_ms: int = Field(default=160, ge=60, le=500)       # audio block granularity
    lookback_ms: int = Field(default=200, ge=100, le=2500)  # F0 history window
                                                           # (upstream allows 2500)
    crossfade_ms: int = Field(default=8, ge=0, le=40)       # block-to-block blend
    silence_threshold_db: float = Field(default=-40.0, le=-20.0, ge=-70.0)
    keep_alive_s: float = Field(default=60.0, ge=5.0, le=600.0)
    f0_method: Literal["rmvpe", "fcpe", "pm"] = "rmvpe"
    max_sessions: int = Field(default=2, ge=1, le=16)       # VRAM guard per server


class TrainingDefaults(BaseModel):
    sample_rate: Literal["32k", "40k", "48k"] = "40k"
    version: Literal["v1", "v2"] = "v2"
    f0: bool = True
    f0_method: Literal["rmvpe", "fcpe", "pm"] = "rmvpe"
    # only the methods the extractor registry actually ships are accepted —
    # crepe/harvest/dio were evaluated and rejected (see docs/ARCHITECTURE.md);
    # refusing them at config time prevents a late failure mid-pipeline.
    epochs: int = Field(default=200, ge=1, le=10000)
    batch_size: int = Field(default=6, ge=1, le=64)
    learning_rate: float = 1e-4
    lr_decay: float = 0.999875
    segment: int | None = None                  # None -> derive from sample rate
    save_every_epoch: int = Field(default=10, ge=1, le=200)
    save_latest: bool = True
    save_every_weights: bool = True
    cache_in_gpu: bool = False
    preload_base_models: bool = True             # f0G/f0D v2 pretrained
    gpu: int | None = None


class TunnelConfig(BaseModel):
    enabled: bool = False
    binary: str = "cloudflared"
    retries: int = Field(default=3, ge=1, le=10)
    start_timeout_s: float = Field(default=30.0, ge=5.0, le=120.0)


class DownloadsConfig(BaseModel):
    huggingface_base: str = (
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
    )
    files: dict[str, str] = Field(
        default_factory=lambda: {
            "hubert_base/config.json": "hubert_base/config.json",
            "hubert_base/preprocessor_config.json": "hubert_base/preprocessor_config.json",
            "hubert_base/pytorch_model.bin": "hubert_base/pytorch_model.bin",
            "rmvpe.pt": "rmvpe/rmvpe.pt",
            "rmvpe.onnx": "rmvpe/rmvpe.onnx",
            "pretrained_v2/f0G40k.pth": "pretrained_v2/f0G40k.pth",
            "pretrained_v2/f0D40k.pth": "pretrained_v2/f0D40k.pth",
            "pretrained_v2/f0G48k.pth": "pretrained_v2/f0G48k.pth",
            "pretrained_v2/f0D48k.pth": "pretrained_v2/f0D48k.pth",
            "pretrained_v2/f0G32k.pth": "pretrained_v2/f0G32k.pth",
            "pretrained_v2/f0D32k.pth": "pretrained_v2/f0D32k.pth",
        }
    )
    chunk_mb: int = 8
    connect_timeout_s: float = 30.0


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dir_name: str = "logs"
    max_bytes: int = 10 * 1024 * 1024
    backups: int = 5


class PlatformConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    conversion: ConversionDefaults = Field(default_factory=ConversionDefaults)
    realtime: RealtimeConfig = Field(default_factory=RealtimeConfig)
    training: TrainingDefaults = Field(default_factory=TrainingDefaults)
    tunnel: TunnelConfig = Field(default_factory=TunnelConfig)
    downloads: DownloadsConfig = Field(default_factory=DownloadsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


ENV_OVERRIDES = {
    "RVC_SERVER_HOST": ("server", "host"),
    "RVC_SERVER_PORT": ("server", "port"),
    "RVC_AUTH_ENABLED": ("server", "auth_enabled"),
    "RVC_AUTH_TOKEN": ("server", "auth_token"),
    "RVC_DEVICE": ("engine", "device"),
    "RVC_F0_METHOD": ("engine", "f0_method"),
    "RVC_BLOCK_MS": ("realtime", "block_ms"),
    "RVC_TRAIN_SR": ("training", "sample_rate"),
    "RVC_TUNNEL_ENABLED": ("tunnel", "enabled"),
    "RVC_LOG_LEVEL": ("logging", "level"),
}
