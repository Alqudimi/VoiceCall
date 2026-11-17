"""End-to-end dataset preparation pipeline (orchestrator).

    check -> preprocess -> f0 -> features -> filelist -> index -> config

Used by CLI (`make prepare-dataset`) and the API; each stage is idempotent
and can be re-run independently.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from rvc_platform.core.errors import DatasetError
from rvc_platform.engine.rvc.index import build_index, newest_added_index
from rvc_platform.training import feature_extract
from rvc_platform.training.preprocess import DatasetChecker, DatasetPreprocessor
from rvc_platform.training.run_config import TrainingRunConfig
from rvc_platform.training.utils import atomic_json_dump
from rvc_platform.utils.device import select_device

logger = logging.getLogger(__name__)

StageProgress = Callable[[str, float, str], None]


class DataPipeline:
    def __init__(self, config: TrainingRunConfig) -> None:
        self.config = config
        self.layout = config.ensure_layout()

    # ------------------------------------------------------------------ stages
    def check(self) -> dict:
        report = DatasetChecker().check(Path(self.config.dataset_path))
        if not report.ok:
            raise DatasetError(f"dataset unusable: {report.to_dict()}")
        atomic_json_dump(report.to_dict(), self.layout["filelist"].parent / "dataset_report.json")
        return report.to_dict()

    def preprocess(self, progress: StageProgress | None = None) -> dict:
        def _cb(frac: float, name: str) -> None:
            if progress:
                progress("preprocess", frac, name)

        pp = DatasetPreprocessor(self.config.sr_int, self.layout["gt_wavs"].parent,
                                 progress=_cb)
        return pp.run(Path(self.config.dataset_path), progress=_cb)

    def extract_f0(self, progress: StageProgress | None = None) -> dict:
        profile = select_device(None)
        device = profile.device
        is_half = False if device == "cpu" else profile.is_half
        def _cb(frac: float, name: str) -> None:
            if progress:
                progress("f0", frac, name)

        return feature_extract.extract_f0(
            self.layout["gt_wavs"].parent, self.config.f0_method, device, is_half,
            progress=_cb,
        )

    def extract_features(self, progress: StageProgress | None = None) -> dict:
        profile = select_device(None)
        device = profile.device
        is_half = False if device == "cpu" else profile.is_half
        def _cb(frac: float, name: str) -> None:
            if progress:
                progress("features", frac, name)

        return feature_extract.extract_features(
            self.layout["gt_wavs"].parent, self.config.version, device, is_half,
            progress=_cb,
        )

    def build_index(self, progress: StageProgress | None = None) -> dict:
        run_dir = self.layout["feature"].parent
        added = build_index(
            self.layout["feature"], run_dir, self.config.experiment_name,
            self.config.version,
            progress=lambda f: progress("index", f, "") if progress else None,
        )
        return {"index_path": str(added)}

    def write_filelist(self) -> dict:
        path = feature_extract.write_filelist(
            self.layout["gt_wavs"].parent, self.config.version,
            self.config.f0, self.config.speaker_id,
        )
        return {"filelist": str(path)}

    def persist_config(self) -> dict:
        cfg_path = self.config.save()
        hps = self.config.to_hparams()

        model_json = {
            "train": hps.train,
            "data": hps.data,
            "model": hps.model,
        }
        atomic_json_dump(model_json, self.layout["config"])
        return {"run_config": str(cfg_path), "model_config": str(self.layout["config"])}

    # ------------------------------------------------------------------ all
    def prepare_all(self, progress: StageProgress | None = None) -> dict:
        report: dict = {"check": self.check()}
        report["preprocess"] = self.preprocess(progress)
        report["f0"] = self.extract_f0(progress)
        report["features"] = self.extract_features(progress)
        report["filelist"] = self.write_filelist()
        report["index"] = self.build_index(progress)
        report.update(self.persist_config())
        return report

    def index_path(self) -> Path | None:
        run_dir = self.layout["feature"].parent
        return newest_added_index(run_dir, self.config.experiment_name, self.config.version)
