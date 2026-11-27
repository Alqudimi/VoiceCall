#!/usr/bin/env python3
"""Run ONE data-pipeline stage in the foreground (resumable, chunkable).

Usage: python run_stage.py <check|preprocess|f0|features|filelist|index|all_small>
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, "/home/z/my-project/VoiceCall/src")

from rvc_platform.training.run_config import TrainingRunConfig
from rvc_platform.training.data_pipeline import DataPipeline
from rvc_platform.infrastructure.paths import get_paths

STAGE = sys.argv[1] if len(sys.argv) > 1 else "check"

RUN = get_paths().runs / "iraqi_voice"
if (RUN / "run_config.json").is_file():
    cfg = TrainingRunConfig.load(RUN)
else:
    # identical to the `prepare-dataset` CLI invocation
    cfg = TrainingRunConfig(
        experiment_name="iraqi_voice",
        dataset_path="/home/z/my-project/VoiceCall/data/raw/iraqi_voice",
        sample_rate="48k", version="v2", f0_method="rmvpe",
        denoise=False, loudness=True, speech_filter_ratio=0.25,
    )
    cfg.save()          # persist immediately (survive interrupted runs)
pipeline = DataPipeline(cfg)

started = time.time()
stages = {
    "check": pipeline.check,
    "preprocess": lambda: pipeline.preprocess(),
    "f0": lambda: pipeline.extract_f0(),
    "features": lambda: pipeline.extract_features(),
    "filelist": pipeline.write_filelist,
    "index": lambda: pipeline.build_index(),
}
if STAGE not in stages:
    raise SystemExit(f"unknown stage: {STAGE}")

report = stages[STAGE]()
print(json.dumps({STAGE: report, "wall_seconds": round(time.time() - started, 1)},
                 ensure_ascii=False, default=str, indent=2))
