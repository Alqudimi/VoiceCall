"""Training-side utilities: wav loading, filelists, hparams container, logger."""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from rvc_platform.core.errors import DatasetError


def load_wav_to_torch(filename: str) -> tuple[torch.Tensor, int]:
    """Read a float32 wav (written by our preprocessor as float32 PCM)."""
    sampling_rate, data = wavfile.read(filename)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        pass
    else:
        data = data.astype(np.float32)
    return torch.from_numpy(data).float(), sampling_rate


def load_filepaths_and_text(filename: str, split: str = "|") -> list[list[str]]:
    with open(filename, "r", encoding="utf-8") as fh:
        filepaths_and_text = [line.strip().split(split) for line in fh if line.strip()]
    if not filepaths_and_text:
        raise DatasetError(f"empty filelist: {filename}")
    return filepaths_and_text


class AttrDict(dict):
    """dict with attribute access — keeps upstream ``hps.data.x`` call sites
    and our ``hps.data["x"]`` sites both working on the same object."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


@dataclass
class HParams:
    """Attribute-style nested config (drop-in for the upstream hps object)."""

    data: dict = field(default_factory=lambda: AttrDict())
    model: dict = field(default_factory=lambda: AttrDict())
    train: dict = field(default_factory=lambda: AttrDict())
    version: str = "v2"
    sample_rate: int = 40_000
    if_f0: int = 1
    exp_name: str = "run"
    model_dir: str = "runs/run"
    pretrainG: str = ""
    pretrainD: str = ""
    total_epoch: int = 200
    save_every_epoch: int = 10
    save_every_weights: str = "1"
    if_latest: int = 1
    if_cache_data_in_gpu: bool = False
    speaker_id: int = 0
    speaker_info: list = field(default_factory=list)
    training_files: str = "filelist.txt"
    gpus: str = "0"

    def __getattr__(self, name: str):  # unknown keys -> None (upstream compat)
        return None

    @classmethod
    def from_template(cls, template: dict, **overrides) -> "HParams":
        hps = cls(
            data=AttrDict(template["data"]),
            model=AttrDict(template["model"]),
            train=AttrDict(template["train"]),
        )
        for key, value in overrides.items():
            setattr(hps, key, value)
        return hps


def get_run_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)
    return logger


def stable_dataset_signature(files: list[Path]) -> dict:
    """Cheap content signature of a dataset (sizes + mtimes) for resume safety."""
    return {
        "files": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "mtime_sum": sum(f.stat().st_mtime for f in files),
    }


def atomic_json_dump(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
