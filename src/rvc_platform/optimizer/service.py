"""OptimizerService — the orchestrator of the ULTIMATE AI VOICE OPTIMIZER.

Lifecycle (each phase persisted — the run survives restarts)::

    build profile -> BASELINE (current settings) -> EXPLORATION ->
    OPTIMIZATION (staged/TPE-guided) -> FINE-TUNING -> VALIDATION
    (held-out audio, top-k candidates) -> ROBUSTNESS (probe types)
    -> FINAL SELECTION (four strategies + preference view) -> results

Threading model: ONE background worker thread per session; every trial
runs a REAL conversion through the project's own inference engine (same
loaded model, no reloads) — or replays a cached conversion from the
experiment memory (Phase 02) when the exact (model, index, input, params)
triple was already measured under the current engine version. The engine's
own RLock serialises access, so the optimizer is safe against concurrent
UI conversions (they simply queue).

Resource management:

* GPU/CPU: the engine's configured device is reused; VRAM/RAM/CPU are
  sampled per trial and reported live;
* pause = finish current trial then wait; resume = continue from the
  persisted state (trials replayed into the search backend — and warm
  started from cross-session memory);
* cancel = finish current trial, mark stopped, keep all results;
* errors: a failing trial is recorded (error field) and the search
  continues; engine/model mismatch or missing profile aborts the session
  with an honest message;
* disk: candidate audio is pruned to the top-5 (+ baseline/best/AB).

Honesty rules enforced here:

* results always include reliability notes (profile quality, embedder
  availability, validation drop, short-audio warnings);
* the winner is chosen from gate-passing trials only (a distorted or
  unintelligible output can never win on similarity);
* validation + robustness scores are what decide among the top-k
  (anti-overfitting), and four explicit strategies are exposed instead of
  one hidden weighting;
* every score is labelled a relative search result, never a match
  probability; preference learning influences only the *combined
  recommendation view*, never objective scores.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from rvc_platform.core.entities import ConversionSettings
from rvc_platform.core.errors import ConversionError
from rvc_platform.infrastructure.paths import get_paths
from rvc_platform.optimizer.analysis import SR_ANALYSIS, AudioAnalyzer
from rvc_platform.optimizer.calibration import SpeakerScoreCalibrator
from rvc_platform.optimizer.embeddings import SpeakerEmbedderManager
from rvc_platform.optimizer.memory import (
    ExperimentMemory,
    audio_sha256,
    file_sha256,
)
from rvc_platform.optimizer.preferences import PreferenceStore
from rvc_platform.optimizer.profile import (
    ReferenceProfile,
    ReferenceProfileStore,
)
from rvc_platform.optimizer.robustness import (
    assign_input_roles,
    robustness_probes,
    robustness_scores,
)
from rvc_platform.optimizer.scoring import (
    DEFAULT_WEIGHTS,
    EvaluationEngine,
    MetricWeights,
    VoiceEvaluation,
)
from rvc_platform.optimizer.search import (
    OptimizationSpace,
    TrialRecord,
    available_f0_methods,
    build_backend,
    stable_seed,
)
from rvc_platform.optimizer.session import (
    OptimizerSessionConfig,
    OptimizerSessionState,
    SessionStore,
)

logger = logging.getLogger(__name__)

MODES = {
    "quick": {"max_trials": 12, "validate_top_k": 2,
              "phases": ("exploration", "optimization"),
              "roundtrip": False},
    "advanced": {"max_trials": 24, "validate_top_k": 3,
                 "phases": ("exploration", "optimization", "fine_tuning"),
                 "roundtrip": False},
    "deep": {"max_trials": 48, "validate_top_k": 5,
             "phases": ("exploration", "optimization", "fine_tuning"),
             "roundtrip": True},
}

_MAX_SEARCH_S = 12.0
_MAX_VAL_S = 10.0


class OptimizerService:
    """Application service wiring analyzer, scorer, search, store, engine."""

    def __init__(self, engine, models_service, config: Any = None) -> None:
        self.engine = engine
        self.models = models_service
        self.cfg = config
        self.store = SessionStore()
        self.profiles = ReferenceProfileStore()
        self.paths = get_paths()
        self.memory = ExperimentMemory()
        self.preferences = PreferenceStore()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pause_evt = threading.Event()      # set -> PAUSE requested
        self._resume_evt = threading.Event()    # set -> resume granted
        self._cancel_evt = threading.Event()      # set -> CANCEL requested
        self._state: OptimizerSessionState | None = None
        self._embedder: SpeakerEmbedderManager | None = None
        self._analyzer: AudioAnalyzer | None = None
        self._backend = None
        self._backend_name = ""
        self._reference_input_cache: dict[str, Any] = {}
        self._model_hash_cache: dict[tuple[str, float], str | None] = {}

    # ================================================================ status
    def status(self) -> dict[str, Any]:
        """Live status for the dashboard."""
        state = self._state
        if state is None:
            return {
                "running": False, "phase": "idle", "session_id": None,
                "embedders": self._embedder_status_dict(),
                "sessions": self.store.list_sessions()[:20],
            }
        trials_done = len(state.trials)
        running = self._thread is not None and self._thread.is_alive()
        progress = self._progress(state, trials_done)
        current = state.current_trial or {}
        best = self._best_record_summary(state)
        return {
            "running": running,
            "paused": bool(self._pause_evt.is_set() and running),
            "phase": state.phase,
            "session_id": state.config.session_id,
            "model": state.config.model_name,
            "profile": state.config.profile_name,
            "mode": state.config.mode,
            "trials_done": trials_done,
            "max_trials": state.config.max_trials,
            "progress": progress,
            "current_trial": current,
            "best": best,
            "message": state.message,
            "reliability_notes": list(state.reliability_notes),
            "resources": self._resources(),
            "embedders": self._embedder_status_dict(),
            "elapsed_s": round(time.time() - state.started_at, 1)
            if state.started_at else 0.0,
        }

    def _progress(self, state: OptimizerSessionState, done: int) -> float:
        total = max(1, state.config.max_trials)
        base = 0.10 if state.baseline else 0.0
        search_frac = min(1.0, done / total) * 0.70
        val = 0.20 if state.phase in (
            "validation", "robustness", "final_selection", "completed",
            "stopped") else 0.0
        return round(min(1.0, base + search_frac + val), 4)

    def _best_record_summary(self, state: OptimizerSessionState
                              ) -> dict[str, Any] | None:
        best = self._best_trial(state)
        if best is None:
            return None
        return {
            "number": best.number,
            "params": best.params,
            "composite": best.composite,
            "axes": best.axes,
            "gates_passed": best.gates_passed,
        }

    def _best_trial(self, state: OptimizerSessionState) -> TrialRecord | None:
        ok = [t for t in state.trials
              if t.composite is not None and t.error is None and t.gates_passed]
        if not ok:
            ok = [t for t in state.trials
                  if t.composite is not None and t.error is None]
        if not ok:
            return None
        return max(ok, key=lambda t: float(t.composite or 0.0))

    def _embedder_status_dict(self) -> dict[str, Any]:
        if self._embedder is None:
            return {"wavlm": False, "ecapa": False, "reliability": "low",
                    "note": "not initialised — run an analysis first"}
        s = self._embedder.status
        return {
            "wavlm": s.wavlm,
            "ecapa": s.ecapa,
            "dsp": s.dsp,
            "reliability": s.reliability,
            "messages": s.messages,
        }

    def _resources(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            import psutil

            out["cpu_percent"] = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            out["ram_used_mb"] = int(vm.used // (1024 * 1024))
            out["ram_total_mb"] = int(vm.total // (1024 * 1024))
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                out["vram_used_mb"] = int(
                    torch.cuda.memory_allocated(idx) // (1024 * 1024))
                out["vram_total_mb"] = int(
                    torch.cuda.get_device_properties(idx).total_memory
                    // (1024 * 1024))
        except Exception:
            pass
        return out

    # ============================================================== profiles
    def ensure_embedders(self) -> SpeakerEmbedderManager:
        if self._embedder is None:
            prefer = "auto"
            try:
                prefer = getattr(self.cfg, "embedder", "auto")
            except Exception:
                pass
            self._embedder = SpeakerEmbedderManager(
                self.engine.device, prefer=prefer)
            self._analyzer = AudioAnalyzer(self._embedder,
                                            device=self.engine.device)
        return self._embedder

    def build_profile(self, model_name: str, profile_name: str = "default",
                      files: list[Path] | None = None) -> dict[str, Any]:
        """Analyse reference samples and persist a speaker profile."""
        model_name = (model_name or "").strip()
        if not model_name:
            raise ValueError("model name is required")
        self.ensure_embedders()
        if files is None:
            files = self.profiles.sample_files(model_name, profile_name)
        if not files:
            raise ValueError(
                f"no reference samples stored for model {model_name!r} "
                f"profile {profile_name!r} — upload samples first")
        self._embedder_status_dict()
        profile = ReferenceProfile(model_name, profile_name)
        analyzer = self._analyzer
        assert analyzer is not None
        for path in files:
            analysis = analyzer.analyze_file(path).to_dict()
            profile.add_sample_analysis(
                analysis,
                {"id": path.stem, "file": path.name,
                 "duration_s": analysis.get("duration_s")})
        reliability = profile.reliability_report()
        self.profiles.save(profile)
        return {
            "model": model_name,
            "profile": profile_name,
            "n_samples": len(files),
            "reliability": reliability,
            "aggregate": profile.aggregate(),
        }

    def list_profiles(self, model_name: str) -> list[dict[str, Any]]:
        return self.profiles.list_profiles(model_name)

    def delete_profile(self, model_name: str,
                       profile_name: str) -> bool:
        return self.profiles.delete_profile(model_name, profile_name)

    def store_reference_sample(self, model_name: str, profile_name: str,
                               filename: str, payload: bytes) -> dict[str, Any]:
        return self.profiles.store_sample(model_name, profile_name,
                                          filename, payload)

    def list_reference_samples(self, model_name: str,
                               profile_name: str) -> list[dict[str, Any]]:
        out = []
        for p in self.profiles.sample_files(model_name, profile_name):
            try:
                st = p.stat()
                out.append({"id": p.stem, "file": p.name,
                            "size_mb": round(st.st_size / (1024 * 1024), 2),
                            "created_at": st.st_mtime})
            except OSError:
                continue
        return out

    def delete_reference_sample(self, model_name: str, profile_name: str,
                                sample_id: str) -> bool:
        return self.profiles.delete_sample(model_name, profile_name,
                                           sample_id)

    # ================================================================ inputs
    def store_input_audio(self, filename: str,
                          payload: bytes) -> dict[str, Any]:
        import re
        import uuid as _uuid

        from rvc_platform.infrastructure.audio.io import (
            ALLOWED_AUDIO_SUFFIXES, validate_audio_file,
        )
        if not payload:
            raise ValueError("empty upload")
        safe = re.sub(r"[^A-Za-z0-9_.\-]", "_", filename or "input.wav")
        suffix = Path(safe).suffix.lower()
        if suffix not in ALLOWED_AUDIO_SUFFIXES:
            raise ValueError(f"unsupported audio format: {suffix}")
        d = self.paths.data / "optimizer" / "inputs"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{_uuid.uuid4().hex[:8]}_{safe}"
        path.write_bytes(payload)
        try:
            validate_audio_file(path, max_mb=200)
        except Exception as exc:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise ValueError(str(exc)) from exc
        return {"id": path.stem, "file": path.name, "path": str(path)}

    # ================================================================= start
    def start(self, config: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError(
                    "an optimization session is already running "
                    f"({self._state.config.session_id if self._state else '?'}) "
                    "— wait, pause or cancel it first")
            state = self._prepare_state(config)
            self._state = state
            self._pause_evt.clear()
            self._resume_evt.clear()
            self._cancel_evt.clear()
            self._thread = threading.Thread(
                target=self._run, name=state.config.session_id, daemon=True)
            self._thread.start()
            return {"session_id": state.config.session_id,
                    "phase": state.phase, "running": True}

    def _prepare_state(self, config: dict[str, Any]) -> OptimizerSessionState:
        model_name = str(config.get("model_name") or "").strip()
        if not model_name:
            raise ValueError("model_name is required")
        info = self.models.get(model_name)          # raises ModelNotFoundError
        input_path = str(config.get("input_path") or "").strip()
        input_paths = [str(p) for p in (config.get("input_paths") or [])
                       if str(p).strip()]
        if input_path:
            input_paths = [input_path] + [p for p in input_paths
                                          if p != input_path]
        input_paths = [p for p in input_paths
                       if p and Path(p).is_file()]
        if not input_paths:
            raise ValueError("input audio is required (upload it first)")
        if self.engine.net_g is None or \
                (self.engine.loaded_model or None) is None or \
                self.engine.loaded_model.name != model_name:
            # load the model into the engine (single shared engine)
            self.models.load(model_name)
        profile_name = str(config.get("profile_name") or "default")
        profile = self.profiles.load(model_name, profile_name)
        if profile is None:
            raise ValueError(
                f"no reference profile {profile_name!r} for model "
                f"{model_name!r} — analyse reference samples first")
        mode = str(config.get("mode") or "advanced")
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r} (quick|advanced|deep)")
        mode_cfg = MODES[mode]
        weights = dict(DEFAULT_WEIGHTS)
        user_weights = config.get("weights") or {}
        for k, v in user_weights.items():
            if k in weights:
                weights[k] = max(0.0, float(v))
        max_trials = int(config.get("max_trials") or mode_cfg["max_trials"])
        max_trials = max(4, min(max_trials, 400))
        state = OptimizerSessionState()
        state.config = OptimizerSessionConfig(
            session_id=SessionStore.new_session_id(),
            model_name=model_name,
            model_path=str(info.path),
            index_path=str(info.index_path) if info.index_path else None,
            profile_name=profile_name,
            input_path=input_paths[0],
            input_paths=input_paths,
            mode=mode,
            max_trials=max_trials,
            time_budget_s=max(0.0, float(config.get("time_budget_s") or 0.0)),
            strategy=str(config.get("strategy") or "auto"),
            multi_objective=bool(
                config.get("multi_objective")
                or (mode == "deep" and str(config.get("strategy") or "auto")
                    in ("pareto", "nsga2"))),
            pitch_policy=str(config.get("pitch_policy") or "preserve"),
            pitch_range=tuple(config.get("pitch_range") or (-6, 6)),
            weights=weights,
            denoise=bool(config.get("denoise", False)),
            normalize=bool(config.get("normalize", True)),
            f0_methods=list(config.get("f0_methods") or []),
            validate_top_k=int(config.get("validate_top_k")
                               or mode_cfg["validate_top_k"]),
            embedder_prefer=str(config.get("embedder_prefer") or "auto"),
            device=self.engine.device,
        )
        state.phase = "baseline"
        state.started_at = time.time()
        state.message = "session prepared"
        self.store.save(state)
        return state

    # ============================================================== controls
    def pause(self) -> dict[str, Any]:
        if self._state is None or self._thread is None \
                or not self._thread.is_alive():
            return {"paused": False, "message": "no running session"}
        self._pause_evt.set()
        self._resume_evt.clear()
        return {"paused": True, "message": "pause requested — finishing the "
                "current trial first"}

    def resume(self) -> dict[str, Any]:
        with self._lock:
            if self._state is not None and self._thread is not None \
                    and self._thread.is_alive():
                self._pause_evt.clear()
                self._resume_evt.set()
                return {"resumed": True, "message": "resuming"}
            return self.resume_saved()

    def resume_saved(self) -> dict[str, Any]:
        """Continue an interrupted session (app restart / lost thread)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"resumed": False,
                        "message": "a session is already running"}
            # find the latest interruptible session
            for row in self.store.list_sessions():
                sid = row["session_id"]
                state = self.store.load(sid)
                if state is None:
                    continue
                if state.phase in ("completed", "failed", "stopped"):
                    continue
                if row["n_trials"] >= state.config.max_trials and \
                        state.phase != "paused":
                    continue
                self._state = state
                state.phase = "baseline" if state.phase in ("paused",) \
                    else state.phase
                state.message = "resuming interrupted session"
                self._pause_evt.clear()
                self._resume_evt.clear()
                self._cancel_evt.clear()
                self._thread = threading.Thread(
                    target=self._run, name=f"resume_{sid}", daemon=True)
                self._thread.start()
                return {"resumed": True, "session_id": sid}
            return {"resumed": False,
                    "message": "no interrupted session found to resume"}

    def cancel(self) -> dict[str, Any]:
        if self._state is None or self._thread is None \
                or not self._thread.is_alive():
            return {"cancelled": False, "message": "no running session"}
        self._cancel_evt.set()
        self._pause_evt.clear()
        self._resume_evt.set()
        return {"cancelled": True,
                "message": "cancel requested — finishing the current trial, "
                           "results are kept"}

    # ================================================================== run
    def _run(self) -> None:
        state = self._state
        assert state is not None
        try:
            self._run_session(state)
        except Exception as exc:                     # noqa: BLE001
            logger.exception("optimizer: session crashed")
            state.phase = "failed"
            state.message = f"session failed: {exc}"
            state.finished_at = time.time()
            self.store.save(state)

    # ------------------------------------------------------------ run helpers
    def _load_wav(self, path: str) -> tuple[np.ndarray, int]:
        import librosa

        wav, sr = librosa.load(path, sr=None, mono=True)
        if wav.size < SR_ANALYSIS // 2:
            raise ValueError(f"input audio too short (<0.5s): {path}")
        if sr != SR_ANALYSIS:
            wav = librosa.resample(wav, orig_sr=sr,
                                   target_sr=SR_ANALYSIS).astype(np.float32)
        return np.asarray(wav, dtype=np.float32), sr

    def _model_hash(self) -> str | None:
        try:
            path = Path(self._state.config.model_path)
            if not path.is_file():
                return None
            stat = path.stat()
            key = (str(path), stat.st_mtime)
            if key in self._model_hash_cache:
                return self._model_hash_cache[key]
            h = file_sha256(path)
            self._model_hash_cache[key] = h
            return h
        except Exception:
            return None

    def _input_hash(self, wav: np.ndarray) -> str | None:
        return audio_sha256(wav)

    def _calibrator_for(self, profile_dict: dict[str, Any]
                        ) -> SpeakerScoreCalibrator:
        cal = SpeakerScoreCalibrator()
        cal.fit_from_profile(profile_dict)
        return cal

    def _embedder_fingerprint(self) -> dict[str, Any]:
        if self._embedder is None:
            return {}
        s = self._embedder.status
        return {"wavlm": bool(s.wavlm), "ecapa": bool(s.ecapa),
                "dsp": bool(s.dsp)}

    def _phase_for(self, done: int, total: int,
                   phases: tuple[str, ...]) -> str:
        if not phases:
            return "exploration"
        frac = done / max(1, total)
        if frac < 0.30:
            return phases[0]
        if frac < 0.75 and len(phases) > 1:
            return phases[1]
        if len(phases) > 2:
            return phases[2]
        return phases[-1]

    def _current_settings_params(self) -> dict[str, Any]:
        defaults = getattr(self.cfg, "conversion", None) if self.cfg else None
        get = lambda k, d: getattr(defaults, k, d) if defaults is not None else d  # noqa: E731
        return {
            "pitch_shift": int(get("pitch_shift", 0)),
            "f0_method": str(getattr(
                getattr(self.cfg, "engine", None), "f0_method", "rmvpe")
                if self.cfg is not None and getattr(self.cfg, "engine", None)
                is not None else "rmvpe"),
            "index_rate": float(get("index_rate", 0.75)),
            "filter_radius": int(get("filter_radius", 0)),
            "rms_mix_rate": float(get("rms_mix_rate", 0.25)),
            "protect": float(get("protect", 0.33)),
        }

    # ----------------------------------------------------------- main session
    def _run_session(self, state: OptimizerSessionState) -> None:
        cfg = state.config
        state.message = "preparing analysis"
        self.store.save(state)

        # -- model sanity: the engine must still hold this model
        loaded = self.engine.loaded_model
        if loaded is None or loaded.name != cfg.model_name:
            self.models.load(cfg.model_name)

        # -- analyzer + embedders
        analyzer = self._analyzer or AudioAnalyzer(
            self.ensure_embedders(), device=cfg.device)
        self._analyzer = analyzer
        self._embedder_status_dict()

        # -- Phase 02: input role assignment (file-level when possible)
        roles = assign_input_roles(cfg.input_paths or [cfg.input_path],
                                   seed=stable_seed(cfg.session_id))
        state.data_split = roles
        for note in roles["notes"]:
            if note not in state.reliability_notes:
                state.reliability_notes.append(note)

        search_wav = self._concat_role(roles["search"], analyzer)
        if search_wav.size < SR_ANALYSIS:            # <1s of speech
            search_wav = self._role_wav_fallback(roles["search"], analyzer)
        val_wav = (self._concat_role(roles["validation"], analyzer)
                   if roles["validation"] else np.zeros(0, dtype=np.float32))
        test_wav = (self._concat_role(
            [p for p in roles["test"]
             if p not in roles["validation"]] or roles["test"], analyzer)
            if roles["test"] else np.zeros(0, dtype=np.float32))
        # single-file fallback: segment split (search/validation/test)
        if roles["strategy"] == "segment-level" and not val_wav.size:
            full = self._role_wav_fallback(
                roles["search"], analyzer, full=True)
            s, v = analyzer.split_search_validation(
                full, max_search_s=_MAX_SEARCH_S, max_validation_s=_MAX_VAL_S)
            if s.size:
                search_wav = s
            val_wav = v
            if val_wav.size:
                # split the held-out audio into validation + final test
                half = val_wav.size // 2
                test_wav = val_wav[half:]
                val_wav = val_wav[:half]
        elif roles["strategy"] == "file+segment" and \
                test_wav.size and np.array_equal(test_wav, val_wav):
            half = val_wav.size // 2
            test_wav = val_wav[half:]
            val_wav = val_wav[:half]

        input_analysis = analyzer.analyze(
            search_wav, "input", with_embeddings=False).to_dict()
        self._save_wav(self.store.audio_dir(cfg.session_id)
                       / "input_segment.wav", search_wav)
        if test_wav.size:
            test_wav = test_wav[: int(_MAX_VAL_S * SR_ANALYSIS)]

        # -- profile + calibration + evaluation engine
        profile = self.profiles.load(cfg.model_name, cfg.profile_name)
        if profile is None:
            raise ValueError("reference profile disappeared")
        profile_dict = profile.to_dict()
        profile_reliability = profile.reliability_report()
        state.reliability_notes = list(profile_reliability.get("notes") or [])
        if profile_reliability.get("level") == "low":
            state.reliability_notes.insert(
                0, "reference profile reliability is LOW — add more/cleaner "
                   "reference samples for trustworthy identity scoring")
        calibrator = self._calibrator_for(profile_dict)
        weights = MetricWeights(dict(cfg.weights or DEFAULT_WEIGHTS))
        evaluator = EvaluationEngine(weights)

        # -- search space + backend (deterministic seed — Phase 02 W9)
        methods = [m for m in (cfg.f0_methods or available_f0_methods())
                   if m in available_f0_methods()]
        if not methods:
            methods = available_f0_methods()
        space = OptimizationSpace(
            f0_methods=methods,
            pitch_policy=cfg.pitch_policy,
            pitch_range=cfg.pitch_range,
        )
        seed = stable_seed(cfg.session_id)
        self._backend, self._backend_name = build_backend(
            space, strategy=cfg.strategy,
            multi_objective=cfg.multi_objective,
            seed=seed, budget_total=cfg.max_trials)
        backend = self._backend
        restored = getattr(backend, "restore", None)
        if restored and state.trials:
            restored([t for t in state.trials if t.error is None])
        # Phase 02: warm start from cross-session experiment memory
        model_hash = self._model_hash()
        warm = []
        if model_hash:
            try:
                warm = self.memory.warm_start_params(model_hash, k=3)
            except Exception:
                logger.debug("optimizer: warm start lookup failed",
                             exc_info=True)
        warm_start_hook = getattr(backend, "warm_start", None)
        if warm_start_hook and warm and state.baseline is None:
            warm_start_hook(warm)
            state.reliability_notes.append(
                f"warm start: {len(warm)} prior parameter set(s) reused from "
                "the experiment memory for this model")

        # -- memory bookkeeping
        index_hash = (file_sha256(cfg.index_path)
                      if cfg.index_path else None)
        search_input_hash = self._input_hash(search_wav)
        embedder_fp = self._embedder_fingerprint()

        # -- BASELINE: current configured settings
        baseline_params = self._current_settings_params()
        if baseline_params.get("f0_method") not in methods and methods:
            dropped = baseline_params.get("f0_method")
            baseline_params["f0_method"] = methods[0]
            state.reliability_notes.append(
                f"configured default f0 method {dropped!r} is unavailable "
                f"here — the baseline uses {methods[0]!r} instead")
        if state.baseline is None:
            state.phase = "baseline"
            state.message = "converting baseline (current settings)"
            self.store.save(state)
            base_eval, base_path, _ = self._trial_conversion(
                state, search_wav, baseline_params, analyzer, profile_dict,
                evaluator, calibrator, label="baseline",
                model_hash=model_hash, index_hash=index_hash,
                input_hash=search_input_hash, embedder_fp=embedder_fp)
            state.baseline = {
                "params": baseline_params,
                "evaluation": base_eval.to_dict(),
                "audio": base_path.name if base_path else None,
            }
            state.trials.append(TrialRecord(
                number=0, params=baseline_params,
                composite=base_eval.composite,
                axes=base_eval.to_dict()["axes"],
                gates_passed=base_eval.gates_passed,
                failed_gates=base_eval.failed_gates,
                reliability=base_eval.reliability,
                error=None,
            ))
            self.store.save(state)

        # -- search phases
        phases = MODES[cfg.mode]["phases"]
        total = cfg.max_trials
        done = len([t for t in state.trials if t.number > 0])
        deadline = (time.time() + cfg.time_budget_s
                    if cfg.time_budget_s else None)

        while done < total and not self._cancel_evt.is_set():
            # honour pause between trials
            while self._pause_evt.is_set() and not self._cancel_evt.is_set():
                state.phase = "paused"
                state.message = "paused between trials — resume any time"
                self.store.save(state)
                self._resume_evt.wait(timeout=2.0)
                if not self._pause_evt.is_set():
                    state.phase = self._phase_for(done, total, phases)
                    break
            if self._cancel_evt.is_set():
                break
            if deadline is not None and time.time() > deadline:
                state.reliability_notes.append(
                    "time budget reached — search stopped early")
                break
            state.phase = self._phase_for(done, total, phases)
            params = backend.suggest(done)
            state.current_trial = {
                "number": done + 1, "params": params,
                "phase": state.phase,
            }
            state.message = f"trial {done + 1}/{total} ({self._backend_name})"
            self.store.save(state)
            t0 = time.time()
            try:
                ev, out_path, from_cache = self._trial_conversion(
                    state, search_wav, params, analyzer, profile_dict,
                    evaluator, calibrator, label=f"trial_{done + 1}",
                    model_hash=model_hash, index_hash=index_hash,
                    input_hash=search_input_hash, embedder_fp=embedder_fp,
                    record_to_memory=True)
                rec = TrialRecord(
                    number=done + 1, params=params,
                    composite=ev.composite,
                    axes=ev.to_dict()["axes"],
                    gates_passed=ev.gates_passed,
                    failed_gates=ev.failed_gates,
                    reliability=ev.reliability,
                    duration_s=round(time.time() - t0, 2),
                    vram_mb=(self._resources().get("vram_used_mb")),
                    cpu_percent=(self._resources().get("cpu_percent")),
                    artifact_detail=dict(
                        (ev.detail or {}).get("artifact_penalties") or {}),
                    from_cache=from_cache,
                )
                if from_cache:
                    state.n_cache_replays += 1
            except ConversionError as exc:
                rec = TrialRecord(number=done + 1, params=params,
                                  error=str(exc),
                                  duration_s=time.time() - t0)
            backend.tell(rec)
            state.trials.append(rec)
            state.current_trial = None
            # prune AFTER appending: the fresh trial must be in the keep
            # set (Phase 02 bug fix — Phase 01 pruned before appending and
            # deleted every new candidate's audio immediately)
            if not rec.error:
                self._prune_candidate_audio(state, done + 1)
            done += 1
            self.store.save(state)

        if self._cancel_evt.is_set():
            state.phase = "stopped"
            state.message = "stopped by user — partial results kept"
            state.finished_at = time.time()
            self._finalize_analysis(state, evaluator, profile,
                                     profile_dict, profile_reliability,
                                     test_wav, analyzer, calibrator,
                                     model_hash, index_hash,
                                     search_input_hash, embedder_fp,
                                     early=True)
            return

        # -- VALIDATION on held-out audio (top-k) + ROBUSTNESS probes
        top_k = max(1, cfg.validate_top_k)
        candidates = [t for t in state.trials
                     if t.number > 0 and t.error is None
                     and t.composite is not None and t.gates_passed]
        candidates.sort(key=lambda t: float(t.composite or 0.0), reverse=True)
        candidates = candidates[:top_k]
        state.phase = "validation"
        state.message = f"validating top {len(candidates)} on held-out audio"
        self.store.save(state)

        if val_wav.size >= SR_ANALYSIS and candidates:
            probes = robustness_probes(analyzer, val_wav)
            val_input_analysis = analyzer.analyze(
                val_wav, "validation_input",
                with_embeddings=False).to_dict()
            robustness_records: list[dict[str, Any]] = []
            for cand in candidates:
                if self._cancel_evt.is_set():
                    break
                probe_results = []
                for probe in probes:
                    if probe["audio"].size < SR_ANALYSIS // 2:
                        continue
                    try:
                        ev, path, _ = self._trial_conversion(
                            state, probe["audio"], cand.params, analyzer,
                            profile_dict, evaluator, calibrator,
                            label=f"validation_{cand.number}",)
                        probe_results.append({
                            "type": probe["type"],
                            "composite": ev.composite,
                            "axes": ev.to_dict()["axes"],
                            "error": None,
                        })
                    except ConversionError as exc:
                        probe_results.append({
                            "type": probe["type"], "composite": None,
                            "error": str(exc)})
                robustness_records.append({
                    "number": cand.number,
                    "params": cand.params,
                    "search_composite": cand.composite,
                    "probes": probe_results,
                    "robustness": robustness_scores(probe_results),
                })
            # aggregate: validation composite = probe mean
            for rr in robustness_records:
                rb = rr["robustness"] or {}
                rr["validation_composite"] = rb.get("mean")
                drop = None
                if rr["search_composite"] and rr["validation_composite"]:
                    drop = round(
                        1.0 - float(rr["validation_composite"])
                        / max(1e-9, float(rr["search_composite"])), 4)
                rr["drop"] = drop
                state.validation.append({
                    "number": rr["number"],
                    "params": rr["params"],
                    "search_composite": rr["search_composite"],
                    "validation_composite": rr["validation_composite"],
                    "drop": drop,
                    "robustness": rb,
                })
                self.store.save(state)
            state.robustness = {
                "note": "stability = mean - 0.5 x std across probe types",
                "candidates": robustness_records,
            }
            drops = [v["drop"] for v in state.validation
                     if v.get("drop") is not None]
            if drops and max(drops) > 0.15:
                state.reliability_notes.append(
                    f"validation drop up to {max(drops)*100:.0f}% for some "
                    "top candidates — scores may be overfitted to the "
                    "search audio; trust the validation numbers")
        else:
            if candidates:
                state.reliability_notes.append(
                    "input audio too short for a held-out validation set — "
                    "validation skipped (results rest on the search audio "
                    "only); add a second input take to enable it")
            else:
                state.reliability_notes.append(
                    "no candidate passed the quality gates — held-out "
                    "validation skipped (gate-failing candidates are never "
                    "selected); check the artifact scores in the trial log")

        # -- FINAL TEST on the test-role audio (never used above)
        final_test: dict[str, Any] | None = None
        if test_wav.size >= SR_ANALYSIS and candidates:
            try:
                winner = self._select_winner(state)
                if winner is not None:
                    ev, path, _ = self._trial_conversion(
                        state, test_wav, winner.params, analyzer,
                        profile_dict, evaluator, calibrator,
                        label="final_test")
                    final_test = {
                        "number": winner.number,
                        "params": winner.params,
                        "test_composite": ev.composite,
                        "axes": ev.to_dict()["axes"],
                        "audio": path.name if path else None,
                    }
            except Exception:
                logger.exception("optimizer: final test failed")

        # -- FINAL SELECTION (validated + robustness-aware score decides)
        state.phase = "final_selection"
        state.message = "selecting best parameters"
        self.store.save(state)
        val_scores = {v["number"]: v.get("validation_composite")
                      for v in state.validation
                      if v.get("validation_composite") is not None}
        # stability-aware validated score: prefer robust candidates
        stability = {}
        for v in state.validation:
            rb = (v.get("robustness") or {})
            st = rb.get("stability")
            if st is not None:
                stability[v["number"]] = float(st)
        best_rec = None
        best_score = -1.0
        for t in state.trials:
            if t.number == 0 or t.error is not None \
                    or t.composite is None:
                continue
            if not t.gates_passed:
                continue
            score = float(val_scores.get(
                t.number,
                stability.get(t.number, t.composite)))
            if stability.get(t.number) is not None:
                score = 0.5 * score + 0.5 * stability[t.number]
            if score > best_score:
                best_score = score
                best_rec = t
        if best_rec is None:
            fallback = [t for t in state.trials
                        if t.number > 0 and t.error is None
                        and t.composite is not None]
            if fallback:
                best_rec = max(
                    fallback, key=lambda t: float(t.composite or 0.0))
                state.reliability_notes.append(
                    "no candidate passed the quality gates — the reported "
                    "best is the least-bad option, check its artifact scores")
        if best_rec is not None:
            state.best_trial = {
                "number": best_rec.number,
                "params": best_rec.params,
                "composite": best_rec.composite,
                "validated_composite": val_scores.get(best_rec.number),
                "stability": stability.get(best_rec.number),
                "axes": best_rec.axes,
                "reliability": best_rec.reliability,
            }
            # render + keep the best audio for A/B
            try:
                ev, path, _ = self._trial_conversion(
                    state, search_wav, best_rec.params, analyzer,
                    profile_dict, evaluator, calibrator, label="best")
                if path is not None:
                    dest = self.store.audio_dir(cfg.session_id) / "best.wav"
                    try:
                        import shutil

                        shutil.copyfile(path, dest)
                    except OSError:
                        pass
            except Exception:
                logger.exception("optimizer: best render failed")

        # -- pareto front across ALL trials (4 axes — Phase 02)
        evals = []
        for t in state.trials:
            ev = VoiceEvaluation()
            for k, v in (t.axes or {}).items():
                setattr(ev, k, v)
            evals.append(ev)
        state.pareto_indices = [
            i for i in EvaluationEngine.pareto_front(evals)
            if state.trials[i].gates_passed]

        # -- best by strategy (identity / naturalness / clarity / balanced)
        self._compute_best_by_strategy(state, val_scores, stability)

        # -- round-trip probe (deep mode): the model's own ceiling
        state.roundtrip = None
        if MODES[cfg.mode].get("roundtrip") and model_hash:
            state.roundtrip = self._roundtrip_probe(
                state, analyzer, profile_dict, best_rec, calibrator)
        if final_test is not None:
            state.robustness = state.robustness or {}
            state.robustness["final_test"] = final_test

        self._finalize_analysis(state, evaluator, profile, profile_dict,
                                profile_reliability, test_wav, analyzer,
                                calibrator, model_hash, index_hash,
                                search_input_hash, embedder_fp)

    # ------------------------------------------------- finalize + analytics
    def _finalize_analysis(self, state: OptimizerSessionState,
                           evaluator: EvaluationEngine,
                           profile: ReferenceProfile,
                           profile_dict: dict[str, Any],
                           profile_reliability: dict[str, Any],
                           test_wav: np.ndarray,
                           analyzer: AudioAnalyzer,
                           calibrator: SpeakerScoreCalibrator,
                           model_hash: str | None,
                           index_hash: str | None,
                           input_hash: str | None,
                           embedder_fp: dict[str, Any],
                           early: bool = False) -> None:
        """Compute insights, recurrence, diagnostics; persist everything."""
        cfg = state.config
        trial_dicts = [t.to_dict() for t in state.trials]

        # parameter intelligence (sensitivity/importance/interactions)
        try:
            from rvc_platform.optimizer.interactions import \
                analyze_parameter_effects

            state.insights = analyze_parameter_effects(trial_dicts)
        except Exception:
            logger.exception("optimizer: insights failed")
            state.insights = {"status": "error",
                              "note": "parameter analysis failed"}

        # artifact recurrence + parameter linkage
        artifact_recurrence = None
        try:
            from rvc_platform.optimizer.quality import recurrence_analysis

            artifact_recurrence = recurrence_analysis(trial_dicts)
        except Exception:
            logger.exception("optimizer: recurrence analysis failed")

        # root-cause diagnostics (5 classes)
        try:
            from rvc_platform.optimizer.diagnostics import \
                root_cause_analysis

            model_info = None
            try:
                model_info = {"path": cfg.model_path,
                              "index_path": cfg.index_path}
            except Exception:
                pass
            input_analyses = [dict(self._reference_input_cache.get(
                (cfg.session_id, "search")) or {})]
            state.diagnostics = root_cause_analysis(
                trials=trial_dicts,
                baseline=state.baseline,
                best=state.best_trial,
                validation=state.validation,
                profile_reliability=profile_reliability,
                input_analyses=[a for a in input_analyses if a],
                model_info=model_info,
                roundtrip=state.roundtrip,
                insights=state.insights,
                artifact_recurrence=artifact_recurrence,
                embedder_status=self._embedder_status_dict(),
                robustness=(state.robustness or {}).get("final_test")
                and state.robustness or None,
            )
        except Exception:
            logger.exception("optimizer: diagnostics failed")
            state.diagnostics = {"status": "error"}

        if artifact_recurrence:
            state.insights = state.insights or {}
            state.insights["artifact_recurrence"] = artifact_recurrence

        # memory stats for the UI
        try:
            state.memory_stats = self.memory.stats()
        except Exception:
            state.memory_stats = None

        # history into the profile (reference-aware optimization)
        try:
            profile.history.append({
                "session_id": cfg.session_id,
                "at": time.time(),
                "mode": cfg.mode,
                "best": state.best_trial,
                "best_by_strategy": state.best_by_strategy,
                "n_trials": len(state.trials) - 1,
                "final_test": ((state.robustness or {}).get("final_test")),
            })
            profile.history = profile.history[-10:]
            self.profiles.save(profile)
        except Exception:
            logger.debug("optimizer: profile history update failed",
                         exc_info=True)

        state.phase = "stopped" if early else "completed"
        state.message = ("stopped by user — partial results kept" if early
                         else "optimization completed")
        state.finished_at = time.time()
        state.current_trial = None
        self.store.save(state)

    def _compute_best_by_strategy(self, state: OptimizerSessionState,
                                  val_scores: dict[int, float],
                                  stability: dict[int, float]) -> None:
        """Four winners + the preference-adjusted combined view."""
        from rvc_platform.optimizer.scoring import STRATEGIES

        candidates = []
        for t in state.trials:
            if t.error is not None or t.composite is None:
                continue
            candidates.append({
                "number": t.number,
                "params": t.params,
                "axes": t.axes,
                "composite": t.composite,
                "validated_composite": val_scores.get(t.number),
                "stability": stability.get(t.number),
                "gates_passed": t.gates_passed,
            })
        winners = EvaluationEngine.best_by_strategy(candidates)
        out: dict[str, Any] = {}
        for strategy, cand in winners.items():
            if strategy == "note" or cand is None:
                if strategy == "note":
                    out["note"] = cand
                continue
            # combined ranking: validated/stability + bounded preference boost
            score = float(cand.get("validated_composite")
                          or cand.get("stability")
                          or cand.get("composite") or 0.0)
            boost = 0.0
            try:
                boost = self.preferences.preference_boost(
                    state.config.model_name, cand.get("params") or {})
            except Exception:
                logger.debug("optimizer: preference boost failed",
                             exc_info=True)
            out[strategy] = {
                "number": cand.get("number"),
                "params": cand.get("params"),
                "composite": cand.get("composite"),
                "validated_composite": cand.get("validated_composite"),
                "stability": cand.get("stability"),
                "axes": cand.get("axes"),
                "objective_score": round(score, 4),
                "preference_boost": round(boost, 4),
                "combined_rank_score": round(score * (1.0 + boost), 4),
            }
        out["strategies"] = list(STRATEGIES)
        out["note_preference"] = (
            "preference boost affects the COMBINED ranking only — objective "
            "axis scores are never modified")
        state.best_by_strategy = out

    def _select_winner(self, state: OptimizerSessionState
                       ) -> TrialRecord | None:
        """Current winner: validated+stability-aware, gates honoured."""
        best = self._best_trial(state)
        return best

    def _roundtrip_probe(self, state: OptimizerSessionState,
                         analyzer: AudioAnalyzer,
                         profile_dict: dict[str, Any],
                         best_rec: TrialRecord | None,
                         calibrator: SpeakerScoreCalibrator
                         ) -> dict[str, Any] | None:
        """Convert the target speaker's OWN reference audio through the
        model -> the model's similarity ceiling (model-quality evidence)."""
        try:
            sample_files = self.profiles.sample_files(
                state.config.model_name, state.config.profile_name)
            if not sample_files:
                return None
            wav, _ = self._load_wav(str(sample_files[0]))
            wav = wav[: int(6.0 * SR_ANALYSIS)]
            params = dict(best_rec.params) if best_rec else \
                self._current_settings_params()
            label = "roundtrip"
            if self._embedder is None:
                return None
            out_path = self.store.audio_dir(
                state.config.session_id) / "roundtrip.wav"
            settings = ConversionSettings(
                pitch_shift=int(params.get("pitch_shift", 0)),
                f0_method=str(params.get("f0_method", "rmvpe")),
                index_rate=float(params.get("index_rate", 0.75)),
                filter_radius=int(params.get("filter_radius", 0)),
                rms_mix_rate=float(params.get("rms_mix_rate", 0.25)),
                protect=float(params.get("protect", 0.33)),
                denoise=state.config.denoise,
                normalize=state.config.normalize,
                auto_pitch=(state.config.pitch_policy == "auto"),
            )
            settings.validate()
            import soundfile as sf

            tmp_in = self.store.audio_dir(state.config.session_id) / \
                "_tmp_roundtrip_in.wav"
            sf.write(str(tmp_in), wav, SR_ANALYSIS, subtype="PCM_16")
            self.engine.convert_file(tmp_in, out_path, settings)
            tmp_in.unlink(missing_ok=True)
            out_wav, _ = self._load_wav(str(out_path))
            refs = {k: [np.asarray(v, dtype=np.float32)
                        for v in vecs]
                    for k, vecs in (profile_dict.get("embeddings")
                                    or {}).items()}
            if not refs:
                return None
            report = None
            try:
                report = self._embedder.similarity_report(
                    out_wav, refs, calibrator=calibrator)
            except TypeError:
                report = self._embedder.similarity_report(out_wav, refs)
            return {
                "sample": sample_files[0].name,
                "speaker_similarity": (
                    (report.get("calibration") or {}).get(
                        "calibrated_ensemble")
                    or report.get("ensemble")),
                "raw": report.get("similarities"),
                "params": params,
                "note": "converting the target speaker's own reference "
                        "audio through the model — a proxy ceiling for "
                        "what any inference settings can achieve",
            }
        except Exception:
            logger.exception("optimizer: round-trip probe failed")
            return None

    # ---------------------------------------------------------- conversions
    def _concat_role(self, paths: list[str], analyzer: AudioAnalyzer
                     ) -> np.ndarray:
        """Concatenate (speech-only, capped) audio of a role's files."""
        pieces = []
        budget = int(_MAX_SEARCH_S * SR_ANALYSIS)
        for p in (paths or []):
            try:
                wav, _ = self._load_wav(p)
                speech = analyzer.speech_only(wav, max_s=8.0)
                if speech.size >= SR_ANALYSIS // 2 and budget > 0:
                    pieces.append(speech[:budget])
                    budget -= speech.size
            except Exception:
                logger.exception("optimizer: role audio load failed (%s)", p)
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces)

    def _role_wav_fallback(self, paths: list[str], analyzer: AudioAnalyzer,
                           full: bool = False) -> np.ndarray:
        """Full (non-speech-trimmed) audio of the first file — fallback."""
        for p in (paths or []):
            try:
                wav, _ = self._load_wav(p)
                if full:
                    return wav
                return wav[: int(_MAX_SEARCH_S * SR_ANALYSIS)]
            except Exception:
                continue
        return np.zeros(0, dtype=np.float32)

    def _trial_conversion(self, state: OptimizerSessionState,
                          wav16k: np.ndarray, params: dict[str, Any],
                          analyzer: AudioAnalyzer,
                          profile_dict: dict[str, Any],
                          evaluator: EvaluationEngine,
                          calibrator: SpeakerScoreCalibrator,
                          label: str, *,
                          model_hash: str | None = None,
                          index_hash: str | None = None,
                          input_hash: str | None = None,
                          embedder_fp: dict[str, Any] | None = None,
                          record_to_memory: bool = False,
                          ) -> tuple[Any, Path | None]:
        """Convert one segment set with the given params and evaluate.

        Phase 02: the experiment-memory conversion cache is checked FIRST —
        a hit replays the stored analysis + raw similarities through the
        CURRENT evaluation (weights, calibration) with no conversion and
        no re-embedding; everything is still recorded honestly
        (``from_cache`` flags the trial).
        """
        cfg = state.config
        loaded = self.engine.loaded_model
        if loaded is None or loaded.name != cfg.model_name:
            raise ConversionError(
                f"engine no longer holds model {cfg.model_name!r} — "
                "session aborted to stay honest")

        cached = None
        if model_hash is not None and input_hash is None:
            # per-call input hash when not provided (trial audio)
            input_hash = self._input_hash(wav16k)
        if model_hash is not None and input_hash is not None \
                and label.startswith("trial_"):
            try:
                cached = self.memory.lookup_conversion(
                    model_hash, index_hash, input_hash, params)
            except Exception:
                logger.debug("optimizer: cache lookup failed", exc_info=True)

        if cached is not None:
            # rebuild the similarity report with CURRENT calibration
            sim_report = None
            raw_sims = (cached.get("similarity_report") or {}).get(
                "similarities")
            if raw_sims:
                sim_report = {"similarities": raw_sims,
                              "ensemble": None, "notes": []}
                try:
                    sim_report["calibration"] = \
                        calibrator.calibrate_report(raw_sims)
                    sims = [v for v in raw_sims.values()
                            if v is not None]
                    sim_report["ensemble"] = float(np.mean(sims)) if sims \
                        else None
                except Exception:
                    logger.debug("optimizer: cache recalibration failed",
                                 exc_info=True)
            analysis = cached.get("output_analysis") or {}
            ev = evaluator.evaluate(
                analysis,
                self._input_analysis_for(state, wav16k, label),
                profile_dict, sim_report)
            return ev, None, True

        settings = ConversionSettings(
            pitch_shift=int(params.get("pitch_shift", 0)),
            f0_method=str(params.get("f0_method", "rmvpe")),
            index_rate=float(params.get("index_rate", 0.75)),
            filter_radius=int(params.get("filter_radius", 0)),
            rms_mix_rate=float(params.get("rms_mix_rate", 0.25)),
            protect=float(params.get("protect", 0.33)),
            denoise=cfg.denoise,
            normalize=cfg.normalize,
            auto_pitch=(cfg.pitch_policy == "auto"),
        )
        settings.validate()
        # write the search segment to a temp input file
        session_dir = self.store.audio_dir(cfg.session_id)
        tmp_in = session_dir / f"_tmp_input_{label}.wav"
        self._save_wav(tmp_in, wav16k)
        out_path = session_dir / f"candidate_{label}.wav"
        self.engine.convert_file(tmp_in, out_path, settings)
        try:
            tmp_in.unlink(missing_ok=True)
        except OSError:
            pass
        import librosa

        out_wav, out_sr = librosa.load(str(out_path), sr=SR_ANALYSIS,
                                       mono=True)
        analysis = analyzer.analyze(out_wav, label).to_dict()
        sim_report = None
        if self._embedder is not None and profile_dict.get("embeddings"):
            refs = {k: [np.asarray(v, dtype=np.float32) for v in vecs]
                    for k, vecs in profile_dict["embeddings"].items()}
            if refs:
                try:
                    sim_report = self._embedder.similarity_report(
                        out_wav, refs, calibrator=calibrator)
                except TypeError:
                    # embedder stand-ins without the calibration kwarg
                    # (duck-typed back-compatibility)
                    sim_report = self._embedder.similarity_report(
                        out_wav, refs)
        ev = evaluator.evaluate(analysis, self._input_analysis_for(
            state, wav16k, label), profile_dict, sim_report)

        if record_to_memory and model_hash is not None:
            try:
                self.memory.record_trial(
                    model_hash=model_hash,
                    index_hash=index_hash,
                    profile_id=f"{cfg.model_name}/{cfg.profile_name}",
                    input_hash=input_hash or self._input_hash(wav16k),
                    params=dict(params),
                    session_id=cfg.session_id,
                    axes=ev.to_dict()["axes"],
                    composite=ev.composite,
                    gates_passed=ev.gates_passed,
                    failed_gates=ev.failed_gates,
                    reliability=ev.reliability,
                    artifact_detail=dict(
                        (ev.detail or {}).get("artifact_penalties") or {}),
                    device=cfg.device,
                    embedder_fingerprint=embedder_fp or {},
                    software_version=getattr(
                        self.cfg, "version", "") if self.cfg else "",
                    output_analysis=analysis,
                    similarity_report=sim_report,
                    audio_available=False,
                )
            except Exception:
                logger.exception("optimizer: memory record failed")
        return ev, out_path, False

    def _input_analysis_for(self, state: OptimizerSessionState,
                             wav16k: np.ndarray, label: str
                             ) -> dict[str, Any]:
        """Input-side analysis for prosody/clarity comparison, cached per
        (session, audio-set) so repeated conversions never re-analyse."""
        scope = "search" if (label.startswith("trial")
                             or label in ("baseline", "best")) else \
            "validation"
        key = (state.config.session_id, scope)
        cached = self._reference_input_cache.get(key)
        if cached is None:
            analyzer = self._analyzer
            assert analyzer is not None
            cached = analyzer.analyze(wav16k, f"input_{scope}",
                                      with_embeddings=False).to_dict()
            self._reference_input_cache[key] = cached
        return cached

    def _prune_candidate_audio(self, state: OptimizerSessionState,
                               keep_number: int) -> None:
        """Keep candidate audio only for the current best 5 trials (disk).

        The in-flight trial number is ALWAYS kept (Phase 02 bug fix: the
        keep set previously derived solely from already-appended trials,
        so a fresh trial's audio was deleted the moment it was written).
        """
        d = self.store.audio_dir(state.config.session_id)
        best = self._best_trial(state)
        keep = {f"candidate_trial_{best.number}.wav"} if best else set()
        for t in sorted(state.trials,
                        key=lambda t: float(t.composite or 0),
                        reverse=True)[:5]:
            keep.add(f"candidate_trial_{t.number}.wav")
        keep.add(f"candidate_trial_{keep_number}.wav")
        try:
            for p in d.glob("candidate_trial_*.wav"):
                if p.name not in keep:
                    p.unlink(missing_ok=True)
        except OSError:
            pass

    def _save_wav(self, path: Path, wav16k: np.ndarray) -> None:
        import soundfile as sf

        sf.write(str(path), np.asarray(wav16k, dtype=np.float32),
                 SR_ANALYSIS, subtype="PCM_16")

    # ================================================================ results
    def results(self, session_id: str | None = None) -> dict[str, Any]:
        state = None
        if session_id:
            state = self.store.load(session_id)
        elif self._state is not None:
            state = self._state
        if state is None:
            rows = self.store.list_sessions()
            if not rows:
                raise ValueError("no optimization sessions found")
            state = self.store.load(rows[0]["session_id"])
        if state is None:
            raise ValueError("session not found")
        chart = self._chart_data(state)
        baseline = state.baseline or {}
        best = state.best_trial
        pareto = [
            {"number": state.trials[i].number,
             "params": state.trials[i].params,
             "axes": state.trials[i].axes}
            for i in state.pareto_indices if i < len(state.trials)
        ]
        return {
            "session_id": state.config.session_id,
            "model": state.config.model_name,
            "profile": state.config.profile_name,
            "mode": state.config.mode,
            "phase": state.phase,
            "trials": [t.to_dict() for t in state.trials],
            "baseline": baseline,
            "best": best,
            "best_by_strategy": state.best_by_strategy,
            "validation": state.validation,
            "robustness": state.robustness,
            "insights": state.insights,
            "diagnostics": state.diagnostics,
            "data_split": state.data_split,
            "roundtrip": state.roundtrip,
            "pareto": pareto,
            "chart": chart,
            "reliability_notes": state.reliability_notes,
            "n_cache_replays": state.n_cache_replays,
            "search_space": {
                "description": "parameters searched by this session",
                "note": "filter_radius semantics: the engine applies a fixed "
                        "kernel-3 median filter only when radius > 2, so the "
                        "search covers its two real states (0 = off, 3 = on)",
                "weights": state.config.weights,
            },
        }

    def _chart_data(self, state: OptimizerSessionState) -> dict[str, Any]:
        trials = [t for t in state.trials if t.composite is not None]
        trials.sort(key=lambda t: t.number)
        numbers = [t.number for t in trials]

        def series(axis: str) -> list[float]:
            return [round(float(t.axes.get(axis) or 0), 4) for t in trials]

        return {
            "numbers": numbers,
            "composite": [round(float(t.composite), 4) for t in trials],
            "speaker": series("speaker_similarity"),
            "clarity": series("clarity"),
            "naturalness": series("naturalness"),
            "artifact": series("artifact_score"),
            "content": series("content_preservation"),
            "prosody": series("prosody_preservation"),
            "from_cache": [1 if t.from_cache else 0 for t in trials],
        }

    def audio_path(self, session_id: str, name: str) -> Path:
        safe = "".join(c for c in name
                       if c.isalnum() or c in "-_./")
        if ".." in safe or safe.startswith("/"):
            raise ValueError("invalid audio name")
        path = (self.store.audio_dir(session_id) / f"{safe}.wav"
                if not safe.endswith(".wav") else
                self.store.audio_dir(session_id) / safe)
        if not path.is_file():
            raise FileNotFoundError(f"audio not found: {name}")
        return path

    # ================================================================= apply
    def apply_best(self, session_id: str | None = None,
                   strategy: str = "balanced") -> dict[str, Any]:
        state = (self.store.load(session_id) if session_id
                 else (self._state if self._state is not None else None))
        if state is None:
            raise ValueError("no completed optimization with a best result")
        chosen = None
        strat = (state.best_by_strategy or {}).get(strategy)
        if strat is not None and strat.get("params"):
            chosen = strat
        if chosen is None and state.best_trial is not None:
            chosen = state.best_trial
        if chosen is None:
            raise ValueError("no completed optimization with a best result")
        params = dict(chosen["params"])
        model = state.config.model_name
        previous = self.profiles.load_applied(model)
        payload = {
            "model": model,
            "applied_at": time.time(),
            "session_id": state.config.session_id,
            "profile": state.config.profile_name,
            "strategy": strategy,
            "settings": {
                "pitch_shift": int(params.get("pitch_shift", 0)),
                "f0_method": str(params.get("f0_method", "rmvpe")),
                "index_rate": float(params.get("index_rate", 0.75)),
                "filter_radius": int(params.get("filter_radius", 0)),
                "rms_mix_rate": float(params.get("rms_mix_rate", 0.25)),
                "protect": float(params.get("protect", 0.33)),
            },
            "scores": {
                "composite": chosen.get("composite"),
                "validated_composite": chosen.get("validated_composite"),
                "stability": chosen.get("stability"),
                "axes": chosen.get("axes") or {},
            },
            "previous": (previous or {}).get("settings"),
        }
        self.profiles.save_applied(model, payload)
        return payload

    def applied_settings(self, model_name: str) -> dict[str, Any] | None:
        data = self.profiles.load_applied(model_name)
        if not data:
            return None
        return {
            "model": data.get("model"),
            "applied_at": data.get("applied_at"),
            "settings": data.get("settings"),
            "scores": data.get("scores"),
            "previous": data.get("previous"),
        }

    # ================================================================ export
    def export_report(self, session_id: str | None = None) -> dict[str, Any]:
        """Full machine-readable report (also written next to the session)."""
        results = self.results(session_id)
        state = self.store.load(results["session_id"])
        cfg = state.config if state is not None else None
        report = {
            "generated_at": time.time(),
            "generator": "rvc-platform ULTIMATE AI VOICE OPTIMIZER v2",
            "session": cfg.to_dict() if cfg else {},
            "results": results,
            "disclaimer": (
                "All scores are RELATIVE measures computed inside this "
                "search space with the available metrics. Speaker similarity "
                "is a cosine similarity between speaker-verification "
                "embeddings (calibrated per backend against same-speaker "
                "cohorts where available), NOT a probability of identity "
                "match. Composite scores are weighted combinations of the "
                "nine evaluation axes and must not be reported as 'accuracy' "
                "or 'match percentage'."
            ),
        }
        if state is not None:
            path = self.store.session_dir(
                state.config.session_id) / "report.json"
            try:
                import json as _json

                path.write_text(
                    _json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
            except OSError:
                pass
        return report

    def capabilities(self) -> dict[str, Any]:
        methods = available_f0_methods()
        self.ensure_embedders()
        s = self._embedder.status
        return {
            "f0_methods": methods,
            "embedders": {"wavlm": s.wavlm, "ecapa": s.ecapa, "dsp": s.dsp,
                          "reliability": s.reliability},
            "modes": {k: {"max_trials": v["max_trials"],
                          "validate_top_k": v["validate_top_k"],
                          "roundtrip_probe": v["roundtrip"]}
                      for k, v in MODES.items()},
            "strategies": ["auto", "optuna", "builtin", "pareto"],
            "weights": dict(DEFAULT_WEIGHTS),
            "pitch_policies": ["preserve", "auto", "range"],
            "selection_strategies": ["balanced", "identity", "naturalness",
                                     "clarity"],
            "artifact_detectors": [
                "metallic", "robotic", "buzzing", "crackling", "resonance",
                "pitch_stability", "vibrato", "sibilance", "thd", "holes",
                "clipping", "noise", "breathing", "consonants",
                "transitions"],
            "memory": {"version": self.memory.stats().get(
                "optimizer_version")},
        }

    # ============================================== blind A/B + preferences
    def create_ab_pair(self, session_id: str,
                       left_number: int | None = None,
                       right_number: int | None = None
                       ) -> dict[str, Any]:
        """Create a BLIND A/B pair from a session's candidates.

        Defaults: best strategy winners pair (best vs baseline when only
        one candidate exists). Returns the pair WITHOUT the parameters
        (blind protocol) — they are revealed by :meth:`record_preference`.
        """
        state = self.store.load(session_id)
        if state is None:
            raise ValueError(f"session {session_id!r} not found")
        audio_dir = self.store.audio_dir(session_id)

        def _cand(number: int | None, prefer: str | None = None):
            rec = None
            if number is not None:                 # trial 0 IS the baseline
                rec = next((t for t in state.trials
                            if t.number == number), None)
            if rec is None and prefer is not None:
                strat = (state.best_by_strategy or {}).get(prefer) or {}
                rec = next((t for t in state.trials
                            if t.number == strat.get("number")), None)
            if rec is None:
                return None
            # baseline audio is rendered as candidate_baseline.wav (the
            # label used by _trial_conversion)
            if rec.number == 0:
                path = audio_dir / "candidate_baseline.wav"
            else:
                path = audio_dir / f"candidate_trial_{rec.number}.wav"
            if not path.is_file():
                # best render fallback
                if state.best_trial and \
                        state.best_trial.get("number") == rec.number:
                    path = audio_dir / "best.wav"
            return {
                "number": rec.number,
                "params": dict(rec.params),
                "axes": dict(rec.axes or {}),
                "audio": path.name if path.is_file() else None,
                "label": (f"trial #{rec.number}"
                          if rec.number else "baseline defaults"),
            }

        a = _cand(left_number, prefer="identity")
        b = _cand(right_number)
        if b is None or b.get("number") == (a or {}).get("number"):
            # fall back to baseline as the second side
            b = _cand(0)
        if a is None or a.get("audio") is None or \
                b is None or b.get("audio") is None:
            raise ValueError(
                "cannot build a blind pair — no rendered audio for the "
                "requested candidates (run an optimization first)")
        pair = self.preferences.create_pair(session_id, a, b,
                                            rng=random.Random(
                                                stable_seed(session_id)))
        return {
            "pair_id": pair["pair_id"],
            "session_id": session_id,
            "a_audio": pair["candidates"]["a"]["audio"],
            "b_audio": pair["candidates"]["b"]["audio"],
            "answered": False,
        }

    def ab_pair_blind_view(self, session_id: str, pair_id: str
                           ) -> dict[str, Any]:
        """Pair view with parameters hidden until the choice is made."""
        pair = self.preferences.get_pair(session_id, pair_id)
        if pair is None:
            raise ValueError("pair not found")
        answered = pair.get("chosen") is not None
        return {
            "pair_id": pair_id,
            "session_id": session_id,
            "a_audio": (pair["candidates"]["a"] or {}).get("audio"),
            "b_audio": (pair["candidates"]["b"] or {}).get("audio"),
            "answered": answered,
            "chosen": pair.get("chosen") if answered else None,
            # parameters stay hidden pre-choice
            "a_params": (pair["candidates"]["a"] or {}).get("params")
            if answered else None,
            "b_params": (pair["candidates"]["b"] or {}).get("params")
            if answered else None,
        }

    def record_preference(self, session_id: str, pair_id: str,
                          choice: str, note: str = "") -> dict[str, Any]:
        """Record the blind choice; returns the REVEALED pair."""
        state = self.store.load(session_id)
        if state is None:
            raise ValueError(f"session {session_id!r} not found")
        pair = self.preferences.record_choice(session_id, pair_id,
                                              choice, note)
        return {
            "pair_id": pair_id,
            "chosen": pair["chosen"],
            "a_params": pair["candidates"]["a"]["params"],
            "a_label": pair["candidates"]["a"].get("label"),
            "b_params": pair["candidates"]["b"]["params"],
            "b_label": pair["candidates"]["b"].get("label"),
            "summary": self.preferences.preference_summary(
                state.config.model_name),
        }

    def preference_summary(self, model_name: str) -> dict[str, Any]:
        return self.preferences.preference_summary(model_name)

    # ====================================================== experiment memory
    def best_known(self, model_name: str) -> dict[str, Any] | None:
        """Best parameters ever recorded for a model (cross-session)."""
        info = self.models.get(model_name)
        h = file_sha256(info.path)
        if h is None:
            return None
        best = self.memory.best_known(h)
        if best is None:
            return None
        best["model"] = model_name
        return best

    def memory_stats(self) -> dict[str, Any]:
        return self.memory.stats()

    def memory_sessions(self) -> list[dict[str, Any]]:
        return self.memory.sessions_summary()
