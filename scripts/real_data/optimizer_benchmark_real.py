#!/usr/bin/env python3
"""REAL Phase-02 optimizer benchmark — every component real, no stand-ins.

Difference from `optimizer_benchmark.py` (which used deterministic test
stand-ins for the model/HuBERT/embedders): this benchmark runs the REAL
production stack end-to-end on the REAL voice dataset:

- model: the user's own trained RVC v2/48k model (HuggingFace eng7mi/Iraqi-v3,
  best/model.pth + best/index.index) — real trained weights + real FAISS index
- content encoder: real HuBERT base (official asset)
- F0: real RMVPE (official asset) and Praat pm — both real extractors
- speaker identity: real WavLM x-vector + real ECAPA-TDNN (speechbrain),
  two INDEPENDENT speaker-verification models with per-model calibration
- audio: real recordings from the real dataset with a clean protocol:
  * 5 enrollment files (reference profile) — DISJOINT from inputs
  * 3 input files -> file-level search/validation/test split (no leakage)

Honest disclosures (kept in the report):
- inputs are the SAME SPEAKER the model was trained on (identity round-trip),
  because no second-speaker recording exists in this environment; absolute
  similarity therefore measures round-trip identity preservation, and the
  optimizer's RELATIVE improvement is the measured quantity;
- engine runs on CPU (no GPU in this environment).

Determinism: stable_seed patched to 424242 (same convention as the Phase-02
benchmark) so sessions are reproducible; trial conversions persist in the
experiment memory (fixed env dir) so an interrupted run resumes cheaply.

Run: python optimizer_benchmark_real.py [--trials 10] [--fresh]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/home/z/my-project/VoiceCall")

# -- isolated-but-persistent benchmark env (real assets stay shared) ---------
ENV = ROOT / "data" / "optimizer" / "benchmark_real_env"
for d in ("data", "models", "runs"):
    (ENV / d).mkdir(parents=True, exist_ok=True)
os.environ["RVC_DATA_DIR"] = str(ENV / "data")
os.environ["RVC_MODELS_DIR"] = str(ENV / "models")
os.environ["RVC_RUNS_DIR"] = str(ENV / "runs")
# NOTE: RVC_ASSETS_DIR is intentionally NOT overridden -> real HuBERT + RMVPE

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rvc_platform.optimizer.analysis import AudioAnalyzer  # noqa: E402

REAL_DATASET = ROOT / "data" / "raw" / "iraqi_voice"
REAL_MODEL = ROOT / "models" / "iraqi_voice.pth"
REAL_INDEX = ROOT / "models" / "iraqi_voice.index"

# clean 8-file protocol (all disjoint):
ENROLLMENT = ["voice_0004.wav", "voice_0005.wav", "voice_0008.wav",
              "voice_0010.wav", "voice_0011.wav"]          # reference profile
INPUTS = ["voice_0002.wav", "voice_0013.wav", "voice_0016.wav"]  # optimizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--fresh", action="store_true",
                        help="wipe the benchmark env (full re-run)")
    parser.add_argument("--skip-b", action="store_true",
                        help="run only session A (for chunked runs)")
    args = parser.parse_args()

    if args.fresh and ENV.exists():
        shutil.rmtree(ENV)
        for d in ("data", "models", "runs"):
            (ENV / d).mkdir(parents=True, exist_ok=True)

    import numpy as np  # noqa: F401

    from rvc_platform.application.model_service import ModelService
    from rvc_platform.engine.rvc.inference import RVCInferenceEngine
    from rvc_platform.infrastructure.audio.pipeline import AudioProcessingEngine
    from rvc_platform.infrastructure.config import loader as cfg_loader
    from rvc_platform.infrastructure.config.schema import (
        ConversionDefaults, EngineConfig, OptimizerConfig,
    )
    from rvc_platform.infrastructure.persistence.model_repository import \
        ModelRepository
    from rvc_platform.optimizer import service as service_mod
    from rvc_platform.optimizer.embeddings import SpeakerEmbedderManager
    from rvc_platform.optimizer.service import OptimizerService
    from rvc_platform.utils.device import DeviceProfile

    # reproducible sessions (Phase-02 stable-seed convention)
    service_mod.stable_seed = lambda *a, **k: 424242

    profile_dev = DeviceProfile("cpu", False, None, 0.0, 0.0)
    engine = RVCInferenceEngine(profile_dev, AudioProcessingEngine())
    repo = ModelRepository(None)
    models = ModelService(repo, engine)
    cfg_loader.reset_config()
    cfg = type("C", (), {
        "conversion": ConversionDefaults(),
        "engine": EngineConfig(f0_method="rmvpe"),
        "optimizer": OptimizerConfig(),
    })()

    # ---- real speaker embedders (WavLM + ECAPA, lazily shared) --------------
    embedder = SpeakerEmbedderManager("cpu", prefer="auto")
    emb_status = embedder.status
    print(f"[embedders] wavlm={emb_status.wavlm} ecapa={emb_status.ecapa} "
          f"dsp={emb_status.dsp} reliability={emb_status.reliability}")
    if not (emb_status.wavlm and emb_status.ecapa):
        print("[embedders] FATAL: dual neural embedders required for this "
              "benchmark", file=sys.stderr)
        return 2

    svc = OptimizerService(engine, models, cfg)
    svc._embedder = embedder
    svc._analyzer = AudioAnalyzer(embedder, device="cpu")

    # ---- real trained model + real index ------------------------------------
    if not (REAL_MODEL.is_file() and REAL_INDEX.is_file()):
        print("[model] FATAL: real model/index missing — download first",
              file=sys.stderr)
        return 2
    env_models = ENV / "models"
    if not (env_models / "iraqi_voice.pth").is_file():
        shutil.copy2(REAL_MODEL, env_models / "iraqi_voice.pth")
    if not (env_models / "iraqi_voice.index").is_file():
        shutil.copy2(REAL_INDEX, env_models / "iraqi_voice.index")
    models.register_upload(env_models / "iraqi_voice.pth", "iraqi_voice.pth")
    models.load("iraqi_voice")
    info = models.get("iraqi_voice")
    print(f"[engine] model={info.name} sr={info.sample_rate} "
          f"version={info.version} f0={info.f0} index={info.index_path}")

    # ---- real reference profile from 5 disjoint enrollment files ------------
    enroll_paths = [REAL_DATASET / f for f in ENROLLMENT]
    missing = [p.name for p in enroll_paths if not p.is_file()]
    if missing:
        print(f"[profile] FATAL: enrollment files missing: {missing}",
              file=sys.stderr)
        return 2
    if svc.profiles.load("iraqi_voice", "default") is None:
        t0 = time.time()
        built = svc.build_profile("iraqi_voice", "default",
                                  files=enroll_paths)
        print(f"[profile] built from {len(enroll_paths)} real files in "
              f"{time.time() - t0:.1f}s "
              f"(self-consistency: {built['reliability'].get('self_consistency')})")
    else:
        print("[profile] reusing persisted real profile")

    input_paths = [REAL_DATASET / f for f in INPUTS]
    for p in input_paths:
        if not p.is_file():
            print(f"[inputs] FATAL: {p} missing", file=sys.stderr)
            return 2

    def run_session(tag: str) -> dict:
        t0 = time.time()
        started = svc.start({
            "model_name": "iraqi_voice",
            "input_path": str(input_paths[0]),
            "input_paths": [str(p) for p in input_paths],
            "profile_name": "default",
            "mode": "quick",
            "max_trials": args.trials,
            "strategy": "builtin",
            "f0_methods": ["rmvpe", "pm"],
            "multi_objective": True,
            "weights": {},
        })
        sid = started["session_id"]
        print(f"[session {tag}] started {sid}")
        last = ""
        while True:
            st = svc.status()
            line = (f"phase={st.get('phase')} trials={st.get('trials_done')}"
                    f"/{st.get('max_trials')} {st.get('message', '')[:60]}")
            if line != last:
                print(f"  [{tag}] {line}", flush=True)
                last = line
            if st["phase"] in ("completed", "failed", "stopped"):
                break
            time.sleep(2.0)
        elapsed = time.time() - t0
        results = svc.results(sid)
        return {"session_id": sid, "elapsed_s": round(elapsed, 2),
                "phase": st["phase"], "results": results}

    # ---------------- session A (fresh search) -------------------------------
    print(f"[bench] session A: {args.trials} trials, real model + real "
          f"pipeline, 3 real input files (file-level split)")
    a = run_session("A")
    ra = a["results"]
    if a["phase"] != "completed":
        print(f"[bench] session A ended in phase={a['phase']} "
              f"(partial results kept for inspection)")
        return 3

    rb = None
    if not args.skip_b:
        print("[bench] session B: identical repeat (deterministic seed) — "
              "measuring the cross-session conversion cache")
        b = run_session("B")
        rb = b["results"]

    # ---------------- assemble the measured report ----------------------------
    baseline = ra["baseline"]["evaluation"]["axes"]
    baseline_comp = ra["baseline"]["evaluation"]["composite"]
    best = ra["best"] or {}
    best_comp = best.get("composite")
    final_test = ((ra["robustness"] or {}).get("final_test")) or {}
    ft_comp = final_test.get("test_composite")

    composites = [t["composite"] for t in ra["trials"]
                  if t.get("composite") is not None and t["number"] > 0]
    trials_to_90 = None
    if composites and best_comp:
        target = 0.90 * (best_comp - (baseline_comp or 0)) + (baseline_comp or 0)
        for i, c in enumerate(composites, start=1):
            if c >= target:
                trials_to_90 = i
                break

    validation = ra["validation"] or []
    winner_stability = None
    for v in validation:
        if v.get("number") == best.get("number"):
            winner_stability = (v.get("robustness") or {}).get("stability")

    report = {
        "generated_at": time.time(),
        "setup": {
            "model": "REAL trained RVC v2/48k — eng7mi/Iraqi-v3 "
                     "best/model.pth (user's own training)",
            "index": "REAL FAISS index — best/index.index from the same repo",
            "hubert": "REAL official HuBERT base asset",
            "f0": "REAL RMVPE (official asset) + REAL Praat pm",
            "embedders": "REAL WavLM x-vector + REAL ECAPA-TDNN (speechbrain) "
                         "— independent, each with its own calibration",
            "device": "CPU (no GPU in this environment)",
            "trials": args.trials,
            "enrollment": ENROLLMENT,
            "inputs": INPUTS,
            "split": "file-level: 5 enrollment files (profile) are DISJOINT "
                     "from the 3 input files (search/validation/test)",
            "seed": 424242,
            "disclosure": "inputs are the SAME SPEAKER the model was trained "
                          "on (identity round-trip — no second-speaker "
                          "recording exists here); absolute similarity "
                          "measures round-trip identity preservation, the "
                          "optimizer's RELATIVE improvement is the measured "
                          "quantity",
        },
        "session_a": {
            "session_id": a["session_id"],
            "elapsed_s": a["elapsed_s"],
            "baseline_composite": baseline_comp,
            "best_composite": best_comp,
            "final_test_composite": ft_comp,
            "improvement_over_baseline": (
                round(best_comp - baseline_comp, 4)
                if best_comp is not None and baseline_comp is not None
                else None),
            "final_test_improvement": (
                round(ft_comp - baseline_comp, 4)
                if ft_comp is not None and baseline_comp is not None
                else None),
            "trials_to_90pct_of_gain": trials_to_90,
            "winner_stability": winner_stability,
            "strategies": {
                k: (v or {}).get("composite")
                for k, v in (ra["best_by_strategy"] or {}).items()
                if isinstance(v, dict)},
        },
        "baseline_axes": baseline,
        "best_axes": best.get("axes"),
        "best_params": best.get("params"),
        "insights_status": (ra["insights"] or {}).get("status"),
        "diagnoses_count": ((ra["diagnostics"] or {}).get("n_diagnoses")),
    }
    if rb is not None:
        replays = rb.get("n_cache_replays") or 0
        speedup = round(a["elapsed_s"] / max(1e-9, b["elapsed_s"]), 2)
        report["session_b_cache"] = {
            "elapsed_s": b["elapsed_s"],
            "cached_trial_replays": replays,
            "wall_time_speedup": speedup,
        }

    out_dir = Path("/home/z/my-project/download/voice_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase2_benchmark_real.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "data" / "optimizer" / "benchmark"
     ).mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "optimizer" / "benchmark" / "phase2_benchmark_real.json"
     ).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(json.dumps({k: report[k] for k in report
                      if k in ("session_a", "session_b_cache")}, indent=2))
    print(f"\n[bench] report: {out_dir / 'phase2_benchmark_real.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
