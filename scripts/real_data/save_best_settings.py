#!/usr/bin/env python3
"""Save the benchmark's best settings as the model's applied (restorable)
settings through the project's own store, plus a standalone deliverable file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/z/my-project/VoiceCall")
sys.path.insert(0, str(ROOT / "src"))

from rvc_platform.optimizer.profile import ReferenceProfileStore  # noqa: E402

report = json.loads(
    (ROOT / "data" / "optimizer" / "benchmark" / "phase2_benchmark_real.json")
    .read_text(encoding="utf-8"))
best_params = report["best_params"]
assert best_params, "best params missing from benchmark report"

payload = {
    "model": "iraqi_voice",
    "source": "REAL Phase-02 optimizer benchmark "
              "(eng7mi/Iraqi-v3, real data, session opt_20261126_233854)",
    "strategy": "balanced",
    "params": best_params,
    "scores": {
        "baseline_composite": report["session_a"]["baseline_composite"],
        "best_composite": report["session_a"]["best_composite"],
        "final_test_composite": report["session_a"]["final_test_composite"],
        "final_test_improvement": report["session_a"]["final_test_improvement"],
        "winner_stability": report["session_a"]["winner_stability"],
    },
    "axes_baseline": report["baseline_axes"],
    "axes_best": report["best_axes"],
    "saved_at": report["generated_at"],
    "restorable": True,
}

store = ReferenceProfileStore()
path = store.save_applied("iraqi_voice", payload)
print(f"[applied] saved via project store -> {path}")

out = Path("/home/z/my-project/download/voice_analysis/best_settings_iraqi_voice.json")
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
               encoding="utf-8")
print(f"[file] standalone deliverable -> {out}")
print(json.dumps(payload, ensure_ascii=False, indent=2))
