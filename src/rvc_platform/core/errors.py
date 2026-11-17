"""Domain error taxonomy — explicit failure kinds instead of stringly-typed errors."""
from __future__ import annotations


class RVCPlatformError(Exception):
    """Base class for every platform-raised error."""


class ConfigurationError(RVCPlatformError):
    """Invalid / missing configuration or environment capability mismatch."""


class AssetMissingError(RVCPlatformError):
    """A pretrained asset (HuBERT / RMVPE / base G,D) is not installed."""

    def __init__(self, what: str, hint: str = "") -> None:
        super().__init__(f"Missing required asset: {what}. {hint}".strip())
        self.what = what
        self.hint = hint


class ModelNotFoundError(RVCPlatformError):
    """Requested voice model (.pth) or index does not exist."""


class CheckpointError(RVCPlatformError):
    """Checkpoint unreadable, incompatible, or resume-state mismatch."""


class DatasetError(RVCPlatformError):
    """Dataset folder invalid / empty / contains unusable audio."""


class TrainingError(RVCPlatformError):
    """Training process failure (launch, step, OOM, crash)."""


class ConversionError(RVCPlatformError):
    """Audio-to-audio conversion failure."""


class RealtimeError(RVCPlatformError):
    """Real-time streaming session failure."""


class TunnelError(RVCPlatformError):
    """Cloudflare tunnel launch / supervision failure."""


class SecurityError(RVCPlatformError):
    """Authentication, upload validation, or path traversal attempt."""


class ProcessError(RVCPlatformError):
    """Managed subprocess could not be started / is already running."""


class UnsupportedFeatureError(RVCPlatformError):
    """Feature explicitly unsupported in the current environment."""
