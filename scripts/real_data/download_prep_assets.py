#!/usr/bin/env python3
"""Download only the assets needed for dataset preparation (not training)."""
import sys
sys.path.insert(0, "/home/z/my-project/VoiceCall/src")

from rvc_platform.application.downloads import download_assets

NEEDED = [
    "hubert_base/config.json",
    "hubert_base/preprocessor_config.json",
    "hubert_base/pytorch_model.bin",
    "rmvpe.pt",
]

report = download_assets(
    progress=lambda name, frac: print(f"{name}: {frac * 100:.0f}%", flush=True),
    only=NEEDED,
)
print("REPORT:", report)
