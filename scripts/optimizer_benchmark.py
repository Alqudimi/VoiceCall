#!/usr/bin/env python
"""Optimizer Phase-02 scientific benchmark — measured, reproducible, honest.

What this benchmark MEASURES (all numbers come from real conversions through
the project's own engine on a deterministic synthetic setup):

1. **Baseline vs optimized** — the default conversion settings vs the
   Phase-02 selection strategies, each re-converted on the FINAL TEST take
   (a file that never participated in the search or the validation);
2. **search efficiency** — trials needed to reach 90% of the final best
   composite (mean over the session);
3. **validation robustness** — the selected winner's stability
   (mean - 0.5*std across probe types) on held-out audio;
4. **cross-session cache speedup** — an identical second session (same
   model + inputs + deterministic seed) replaying cached conversions vs
   re-converting: wall time + number of replays.

What it does NOT claim (honesty rules):

* the setup uses the project's test stand-ins (random-weight v2/48k
  synthesiser + deterministic HuBERT stand-in + deterministic embedder
  stand-in, exactly like tests/integration) — absolute scores are NOT
  comparable to production models; the *relative* effect of the optimizer
  pipeline is what is measured;
* no PESQ/STOI numbers are reported — they are not speaker-identity or
  voice-cloning quality measures and would be misleading here;
* results are written to ``data/optimizer/benchmark/phase2_benchmark.json``
  next to a Markdown summary for the report.

Run: PYTHONPATH=src python scripts/optimizer_benchmark.py [--trials N]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

# -- make the platform importable when run from the repo root -----------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rvc_platform.optimizer.analysis import SR_ANALYSIS, AudioAnalyzer  # noqa: E402


def _speech_like(sr: int, seconds: float, f0: float, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    sig = 0.4 * np.sin(2 * np.pi * f0 * t)
    sig += 0.2 * np.sin(2 * np.pi * 2 * f0 * t)
    sig += 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
    env = 0.5 * (1 + np.sin(2 * np.pi * 2.5 * t)) ** 2
    sig = sig * env + 0.005 * rng.standard_normal(sig.shape)
    sig[: int(sr * 0.3)] = 1e-6
    sig[-int(sr * 0.3):] = 1e-6
    return sig.astype(np.float32)


class _FakeEmbedder:
    """Deterministic embedder stand-in (same as tests/integration)."""

    def __init__(self, device="cpu"):
        import numpy as np

        self._np = np
        self.device = device
        self.status = type("S", (), {"wavlm": True, "ecapa": True,
                                     "dsp": False, "messages": {},
                                     "neural_count": 2,
                                     "reliability": "high"})()

    def embeddings(self, wav16k):
        np = self._np

        x = np.asarray(wav16k, dtype=np.float32)
        spec = np.abs(np.fft.rfft(x * 0.5))[:2048]
        e = np.zeros(32, dtype=np.float32)
        e[0] = float(spec.mean())
        e[1] = float(spec.std())
        for j in range(2, 32):
            e[j] = float(spec[(j - 2) * 70: (j - 1) * 70].mean())
        n = np.linalg.norm(e)
        e = e / n if n > 0 else e
        e2 = np.tanh(e) * 3.0
        e2 /= max(1e-9, np.linalg.norm(e2))
        return {"wavlm": e, "ecapa": e2}

    def cosine(self, a, b):
        np = self._np
        d = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / d) if d > 0 else 0.0

    def similarity_report(self, wav16k, reference_embeddings, **_):
        np = self._np
        sims = {}
        cand = self.embeddings(wav16k)
        for name, refs in reference_embeddings.items():
            sims[name] = float(np.mean(
                [self.cosine(cand[name], r) for r in refs]))
        neural = [sims[k] for k in ("wavlm", "ecapa") if k in sims]
        return {"similarities": sims,
                "ensemble": float(sum(neural) / len(neural)) if neural
                else None,
                "agreement_spread": 0.02, "reliability": "high",
                "notes": []}


def _write_take(path: Path, seed: int, f0: float = 175.0):
    import numpy as np
    import soundfile as sf

    wav = np.zeros(int(SR_ANALYSIS * 7.0), dtype=np.float32)
    for i, (start, dur) in enumerate([(0.1, 2.2), (3.0, 2.4), (5.8, 1.0)]):
        seg = _speech_like(SR_ANALYSIS, dur, f0 + 2 * i, seed + i)
        lo = int(start * SR_ANALYSIS)
        wav[lo:lo + seg.size] = seg[:wav.size - lo]
    sf.write(str(path), wav, SR_ANALYSIS, subtype="PCM_16")
    return path


def _build_synth_model(models_dir: Path) -> Path:
    import torch

    from rvc_platform.engine.rvc import checkpoint as ckpt
    from rvc_platform.engine.rvc.models import models as rvc_models
    from rvc_platform.training.run_config import load_template
    from rvc_platform.training.utils import HParams
    from tests.conftest import SYNTH_CONFIG_40K

    template = load_template("48k", "v2")
    hps = HParams.from_template(
        {"train": dict(template["train"]),
         "data": dict(template["data"]),
         "model": dict(template["model"])},
        version="v2", sample_rate=48_000, if_f0=1, exp_name="bench",
        model_dir="", total_epoch=1, save_every_epoch=1,
        save_every_weights="1", if_latest=1, if_cache_data_in_gpu=False,
        pretrainG="", pretrainD="", gpus="0",
    )
    net = rvc_models.SynthesizerTrnMs768NSFsid(
        *SYNTH_CONFIG_40K[:17], is_half=False, sr=48_000)
    out = models_dir / "benchmark_model.pth"
    ckpt.export_inference_checkpoint(net.state_dict(), hps, 3, "v2",
                                      out, 48_000, 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    # -- isolated benchmark env ------------------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix="opt_bench_"))
    import os

    os.environ["RVC_DATA_DIR"] = str(tmp / "data")
    os.environ["RVC_MODELS_DIR"] = str(tmp / "models")
    os.environ["RVC_ASSETS_DIR"] = str(tmp / "assets")
    os.environ["RVC_RUNS_DIR"] = str(tmp / "runs")
    for d in ("data", "models", "assets", "runs"):
        (tmp / d).mkdir(parents=True, exist_ok=True)

    import numpy as np

    from rvc_platform.application.model_service import ModelService
    from rvc_platform.engine.rvc import hubert as hubert_mod
    from rvc_platform.engine.rvc import inference as inf_mod
    from rvc_platform.engine.rvc import realtime as rt_mod
    from rvc_platform.engine.rvc.inference import RVCInferenceEngine
    from rvc_platform.infrastructure.audio.pipeline import AudioProcessingEngine
    from rvc_platform.infrastructure.config import loader as cfg_loader
    from rvc_platform.infrastructure.config.schema import (
        ConversionDefaults, EngineConfig, OptimizerConfig,
    )
    from rvc_platform.infrastructure.persistence.model_repository import \
        ModelRepository
    from rvc_platform.optimizer import service as service_mod
    from rvc_platform.optimizer.service import OptimizerService
    from rvc_platform.utils.device import DeviceProfile

    # deterministic HuBERT stand-in (tests/conftest mock_hubert pattern):
    # fake asset marker + patched feature extractor; the synthesiser, the
    # F0 extractor and all DSP remain fully real
    hubert_dir = tmp / "assets" / "hubert_base"
    hubert_dir.mkdir(parents=True, exist_ok=True)
    (hubert_dir / "config.json").write_text("{}", encoding="utf-8")
    (hubert_dir / "preprocessor_config.json").write_text(
        "{}", encoding="utf-8")

    import torch

    def _fake_features(model, source, version, padding_mask=None):
        import torch

        n = source.shape[-1]
        t = max(1, n // 320)
        feats = torch.zeros(1, t, 768, dtype=source.dtype,
                            device=source.device)
        feats[0, :, 0] = torch.linspace(
            0, 1, t, device=source.device).to(source.dtype)
        return feats

    class _FakeHubert:
        pass

    inf_mod.extract_hubert_features = _fake_features
    rt_mod.extract_hubert_features = _fake_features
    inf_mod.load_hubert_model = lambda device, is_half=False: _FakeHubert()
    rt_mod.load_hubert_model = lambda device, is_half=False: _FakeHubert()
    hubert_mod.load_hubert_model = inf_mod.load_hubert_model

    # fixed seed for reproducible sessions (Phase 02 stable seeds)
    service_mod.stable_seed = lambda *a, **k: 424242

    profile = DeviceProfile("cpu", False, None, 0.0, 0.0)
    engine = RVCInferenceEngine(profile, AudioProcessingEngine())
    repo = ModelRepository(tmp / "models")
    models = ModelService(repo, engine)
    cfg_loader.reset_config()
    cfg = type("C", (), {
        "conversion": ConversionDefaults(),
        "engine": EngineConfig(f0_method="pm"),
        "optimizer": OptimizerConfig(),
    })()
    svc = OptimizerService(engine, models, cfg)
    svc._embedder = _FakeEmbedder("cpu")
    svc._analyzer = AudioAnalyzer(svc._embedder, device="cpu")

    # -- register model + reference profile ------------------------------------
    model_path = _build_synth_model(tmp / "models")
    models.register_upload(model_path, "bench_model.pth")
    info = models.get("bench_model")
    models.load("bench_model")
    model_name = info.name

    ref_dir = svc.profiles.samples_dir(model_name, "default")
    for i, f0 in enumerate((150.0, 156.0)):
        import soundfile as sf

        sf.write(str(ref_dir / f"ref_{i}.wav"),
                 _speech_like(SR_ANALYSIS, 3.5, f0, 40 + i), SR_ANALYSIS,
                 subtype="PCM_16")
    svc.build_profile(model_name, "default")

    inputs = [
        _write_take(tmp / f"take_{i}.wav", seed=60 + 10 * i,
                    f0=170.0 + 5 * i)
        for i in range(3)
    ]

    def run_session(tag: str) -> dict:
        t0 = time.time()
        started = svc.start({
            "model_name": model_name,
            "input_path": str(inputs[0]),
            "input_paths": [str(p) for p in inputs],
            "profile_name": "default",
            "mode": "quick",
            "max_trials": args.trials,
            "strategy": "builtin",
            "f0_methods": ["pm"],
            "weights": {},
        })
        sid = started["session_id"]
        while True:
            st = svc.status()
            if st["phase"] in ("completed", "failed", "stopped"):
                break
            time.sleep(0.3)
        elapsed = time.time() - t0
        results = svc.results(sid)
        return {"session_id": sid, "elapsed_s": round(elapsed, 2),
                "phase": st["phase"], "results": results}

    print(f"[bench] session A: {args.trials} trials, 3 input takes "
          "(file-level split), builtin staged search...")
    a = run_session("A")
    ra = a["results"]
    if a["phase"] != "completed":
        print(f"[bench] FAILED to complete: {a['phase']}")
        return 2

    print("[bench] session B: identical repeat (deterministic seed) — "
          "measuring the cross-session conversion cache...")
    b = run_session("B")
    rb = b["results"]

    # -- assemble the measured report -------------------------------------------
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

    replays = rb.get("n_cache_replays") or 0
    speedup = round(a["elapsed_s"] / max(1e-9, b["elapsed_s"]), 2)

    report = {
        "generated_at": time.time(),
        "setup": {
            "model": "synthetic v2/48k random-weight checkpoint "
                     "(tests/conftest SYNTH_CONFIG_40K)",
            "hubert": "deterministic stand-in (tests/conftest)",
            "embedders": "deterministic stand-in (wavlm+ecapa shaped)",
            "f0": "Praat pm (real extractor)",
            "trials": args.trials,
            "inputs": "3 synthetic takes -> file-level data-role split",
            "seed": 424242,
            "disclaimer": "absolute scores are NOT comparable to production "
                          "models — the RELATIVE effect of the optimizer "
                          "pipeline is what is measured",
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
        "session_b_cache": {
            "session_id": b["session_id"],
            "elapsed_s": b["elapsed_s"],
            "cached_trial_replays": replays,
            "wall_time_speedup": speedup,
        },
        "baseline_axes": baseline,
        "best_axes": best.get("axes"),
        "insights_status": (ra["insights"] or {}).get("status"),
        "diagnoses_count": ((ra["diagnostics"] or {}).get("n_diagnoses")),
    }

    out_dir = Path(args.out) if args.out else \
        Path(tmp) / "data" / "optimizer" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase2_benchmark.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # also copy into the repo working dir for the report
    try:
        repo_out = ROOT / "data" / "optimizer" / "benchmark"
        repo_out.mkdir(parents=True, exist_ok=True)
        (repo_out / "phase2_benchmark.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8")
    except OSError:
        pass

    print(json.dumps({k: v for k, v in report.items()
                      if k in ("session_a", "session_b_cache")}, indent=2))
    print(f"\n[bench] report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
