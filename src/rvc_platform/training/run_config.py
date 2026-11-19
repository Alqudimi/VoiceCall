"""Run configuration for training jobs + run-directory layout.

A training "run" lives in ``runs/<experiment>/`` with the upstream-compatible
layout (so vanilla RVC tooling can also read it):

    runs/<name>/
        config.json           model hyper-parameters actually used
        filelist.txt          gt|feature|f0|f0nsf|spk lines
        0_gt_wavs/            target-sr chunks
        1_16k_wavs/           16 kHz chunks
        2a_f0/  2b-f0nsf/     pitch arrays (.wav.npy)
        3_feature768/         HuBERT features (.npy)
        checkpoints/          G_*.pth, D_*.pth, state.json (resume state)
        exports/              periodic inference checkpoints
        train.log  progress.json  events.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from rvc_platform.core.errors import ConfigurationError, DatasetError
from rvc_platform.infrastructure.paths import get_paths
from rvc_platform.training.utils import HParams

SR_ALIASES = {"32k": 32_000, "40k": 40_000, "48k": 48_000}

TRAINING_F0_METHODS = ("rmvpe", "fcpe", "pm", "crepe")  # dataset extraction methods we ship (crepe: optional torchcrepe)


class TrainingRunConfig(BaseModel):
    """Everything needed to (re)start one training run — all persisted to JSON."""

    experiment_name: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,64}$")
    dataset_path: Path
    sample_rate: str = "40k"                     # 32k | 40k | 48k
    version: str = "v2"
    f0: bool = True
    f0_method: str = "rmvpe"
    epochs: int = Field(default=200, ge=1, le=10_000)
    batch_size: int = Field(default=6, ge=1, le=64)
    learning_rate: float = Field(default=1e-4, gt=0)
    lr_decay: float = Field(default=0.999875, gt=0.0, le=1.0)
    segment: int | None = None                  # None -> template value
    save_every_epoch: int = Field(default=10, ge=1, le=200)
    save_latest: bool = True                     # keep G_2333333/D_2333333 updated
    save_every_weights: bool = True              # periodic inference checkpoints
    save_best: bool = True                       # track + export the best-so-far model
    val_split: float = Field(default=0.05, ge=0.0, le=0.5)  # held-out fraction for best-model scoring (0 = train-mel fallback)
    # ---- researched quality levers (see UPGRADE_REPORT_AR.md) -----------------
    ema_decay: float = Field(default=0.999, ge=0.0, lt=1.0)  # generator EMA (0 disables; BigVGAN-style)
    c_stft: float = Field(default=2.0, ge=0.0)  # multi-resolution STFT loss weight (0 = upstream objective)
    use_msd: bool = True                        # extra multi-scale (raw) discriminator scales 2/4 (HiFi-GAN MSD)
    spk_weight: float = Field(default=1.0, ge=0.0)  # speaker-similarity weight in the composite best-model metric
    early_stop_patience: int = Field(default=0, ge=0, le=10_000)  # epochs without a new best before cooperative stop (0 = off)
    f0_eval: bool = True                         # F0-RMSE diagnostic in validation (needs rmvpe asset)
    # ---- dataset-quality preprocessing options (any dataset, Colab-safe) -----
    denoise: bool = False                        # pre-slicing spectral-gate denoise (noisereduce; guarded no-op)
    loudness: bool = True                        # per-chunk RMS loudness normalisation (common training targets)
    cache_in_gpu: bool = False
    pretrained_g: Path | None = None             # explicit override
    pretrained_d: Path | None = None
    preload_base_models: bool = True
    gpu: int | None = None
    speaker_id: int = Field(default=0, ge=0, le=109)
    clean_start: bool = False                    # ignore existing checkpoints
    resume: bool = True                          # auto-resume when possible
    resume_from: Path | None = None             # explicit G_*.pth to resume from
    hf_repo_id: str | None = None                # private model repo, e.g. user/voice-model

    @field_validator("hf_repo_id")
    @classmethod
    def _hf_repo(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        if len(value.split("/")) != 2 or any(not part for part in value.split("/")):
            raise ValueError("hf_repo_id must use the format 'owner/repository'")
        return value

    @field_validator("sample_rate")
    @classmethod
    def _sr(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in SR_ALIASES:
            raise ValueError(f"sample_rate must be one of {list(SR_ALIASES)}")
        return v

    @field_validator("f0_method")
    @classmethod
    def _method(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in TRAINING_F0_METHODS:
            raise ValueError(
                f"f0_method must be one of {TRAINING_F0_METHODS} "
                "(harvest/dio intentionally not shipped — see docs; "
                "crepe needs the optional torchcrepe quality group)"
            )
        return v

    # ------------------------------------------------------------------ paths
    @property
    def run_dir(self) -> Path:
        return get_paths().runs / self.experiment_name

    @property
    def sr_int(self) -> int:
        return SR_ALIASES[self.sample_rate]

    def layout(self) -> dict[str, Path]:
        base = self.run_dir
        dirs = {
            "gt_wavs": base / "0_gt_wavs",
            "wavs_16k": base / "1_16k_wavs",
            "f0": base / "2a_f0",
            "f0nsf": base / "2b-f0nsf",
            "feature": base / ("3_feature768" if self.version == "v2" else "3_feature256"),
            "checkpoints": base / "checkpoints",
            "exports": base / "exports",
        }
        dirs["filelist"] = base / "filelist.txt"
        dirs["config"] = base / "config.json"
        dirs["train_log"] = base / "train.log"
        dirs["progress"] = base / "progress.json"
        dirs["events"] = base / "events.jsonl"
        dirs["state"] = dirs["checkpoints"] / "state.json"
        return dirs

    def ensure_layout(self) -> dict[str, Path]:
        dirs = self.layout()
        for key, path in dirs.items():
            if key in ("filelist", "config", "train_log", "progress", "events", "state"):
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)
        return dirs

    # ------------------------------------------------------------------ model json
    def resolve_pretrained(self) -> tuple[Path, Path]:
        """Default: official v2 pretrained f0G/f0D matching the sample rate."""
        if self.f0_method and not self.f0:
            raise ConfigurationError("f0 extraction settings contradict f0 model flag")
        root = get_paths().pretrained
        if self.pretrained_g and self.pretrained_d:
            return Path(self.pretrained_g), Path(self.pretrained_d)
        if self.pretrained_g or self.pretrained_d:
            raise ConfigurationError("provide both pretrained_g and pretrained_d or neither")
        prefix = "f0" if self.f0 else ""
        g = root / f"{prefix}G{self.sample_rate}.pth"
        d = root / f"{prefix}D{self.sample_rate}.pth"
        return g, d

    def to_hparams(self, template: dict | None = None) -> HParams:
        """Build the hps object the vendored trainer/dataset expects."""
        if template is None:
            template = load_template(self.sample_rate)
        train = dict(template["train"])
        if self.batch_size:
            train["batch_size"] = self.batch_size
        if self.learning_rate:
            train["learning_rate"] = self.learning_rate
        if self.lr_decay:
            train["lr_decay"] = self.lr_decay
        if self.segment:
            train["segment_size"] = self.segment
        if self.epochs:
            train["epochs"] = self.epochs
        # researched quality levers -> train hyper-parameters (0/False = exact upstream behaviour)
        train["ema_decay"] = self.ema_decay
        train["c_stft"] = self.c_stft
        train["use_msd"] = 1 if self.use_msd else 0
        train["spk_weight"] = self.spk_weight
        train["early_stop_patience"] = self.early_stop_patience
        train["f0_eval"] = 1 if self.f0_eval else 0
        hps = HParams.from_template(
            {"train": train, "data": dict(template["data"]), "model": dict(template["model"])},
            version=self.version,
            sample_rate=self.sr_int,
            if_f0=1 if self.f0 else 0,
            exp_name=self.experiment_name,
            model_dir=str(self.run_dir),
            total_epoch=self.epochs,
            save_every_epoch=self.save_every_epoch,
            save_every_weights="1" if self.save_every_weights else "0",
            if_latest=1 if self.save_latest else 0,
            if_cache_data_in_gpu=self.cache_in_gpu,
            speaker_id=self.speaker_id,
            gpus=str(self.gpu if self.gpu is not None else 0),
        )
        hps.data["training_files"] = str(self.layout()["filelist"])
        return hps

    def save(self) -> Path:
        cfg_path = self.run_dir / "run_config.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return cfg_path

    @classmethod
    def load(cls, run_dir: Path) -> "TrainingRunConfig":
        path = Path(run_dir) / "run_config.json"
        if not path.is_file():
            raise ConfigurationError(f"no run_config.json in {run_dir}")
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def validate_dataset(self) -> None:
        ds = Path(self.dataset_path)
        if not ds.is_dir():
            raise DatasetError(f"dataset folder missing: {ds}")
        files = [
            p for p in ds.rglob("*")
            if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm", ".aac")
        ]
        if len(files) < 1:
            raise DatasetError(f"no audio files in {ds}")


def load_template(sample_rate: str) -> dict:
    """Model hyper-parameter template: project override first, packaged copy
    (guaranteed to exist with the installed package) as fallback."""
    sr = sample_rate.lower().strip()
    if sr not in SR_ALIASES:
        raise ConfigurationError(f"unknown sample rate template: {sample_rate}")
    candidates = [
        get_paths().root / "configs" / "training" / f"{sr}.json",
        Path(__file__).resolve().parent / "templates" / f"{sr}.json",
    ]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise ConfigurationError(f"missing model template: {candidates[0]}")
