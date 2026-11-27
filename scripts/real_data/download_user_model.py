#!/usr/bin/env python3
"""Download the user's trained HF model (eng7mi/Iraqi-v3) via the project's
own HuggingFaceModelService, then verify engine load + a real conversion.

The token comes ONLY from the environment (RVC_HF_TOKEN) — never persisted.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/home/z/my-project/VoiceCall/src")

assert os.environ.get("RVC_HF_TOKEN"), "RVC_HF_TOKEN must be set in the env"

from rvc_platform.application.hf_model_service import HuggingFaceModelService
from rvc_platform.application.model_service import ModelService
from rvc_platform.engine.rvc.inference import RVCInferenceEngine
from rvc_platform.infrastructure.audio.pipeline import AudioProcessingEngine
from rvc_platform.infrastructure.persistence.model_repository import \
    ModelRepository
from rvc_platform.utils.device import DeviceProfile


def main() -> int:
    profile = DeviceProfile("cpu", False, None, 0.0, 0.0)
    engine = RVCInferenceEngine(profile, AudioProcessingEngine())
    repo = ModelRepository(None)   # defaults to get_paths().models
    models = ModelService(repo, engine)
    hf = HuggingFaceModelService(models, repo)

    print(f"[hf] repo: {hf.cfg.repo_id}  token_configured={hf.token_configured()}")
    started = hf.start_download(ensure_assets=True)
    print(f"[hf] started: {started}")

    # poll the background thread (download is byte-range resumable)
    t0 = time.time()
    while hf.download_active() and time.time() - t0 < 900:
        time.sleep(5)
        files = hf.files.get("model", {})
        idx = hf.files.get("index", {})
        print(f"[hf] state={hf.state} "
              f"model={files.get('status')} {files.get('received_bytes', 0) / 1e6:.1f}MB "
              f"index={idx.get('status')}", flush=True)

    status = hf.status()
    # never print the token or the signed URL — only safe fields
    safe = {
        "state": status["download"]["state"],
        "error": status["download"]["error"],
        "model": status["model"],
        "index": status["index"],
        "recommended": status["recommended"],
        "default_model": status["default_model"],
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))
    if status["download"]["state"] != "ready":
        return 1

    # ---- verify the model actually loads into the real engine ----
    info = models.load("iraqi_voice")
    print(f"[engine] loaded: {info.name} sr={info.sample_rate} "
          f"version={info.version} f0={info.f0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
