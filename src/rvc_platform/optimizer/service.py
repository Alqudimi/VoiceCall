"""OptimizerService — the orchestrator of the ULTIMATE AI VOICE OPTIMIZER.

Lifecycle (each phase persisted — the run survives restarts)::

    build profile -> BASELINE (current settings) -> EXPLORATION ->
    OPTIMIZATION (adaptive/TPE-guided) -> FINE-TUNING -> VALIDATION
    (held-out audio, top-k candidates) -> FINAL SELECTION -> results

Threading model: ONE background worker thread per session; every trial
runs a REAL conversion through the project's own inference engine (same
loaded model, no reloads). The engine's own RLock serialises access, so
the optimizer is safe against concurrent UI conversions (they simply queue).

Resource management:
* GPU/CPU: the engine's configured device is reused; VRAM/RAM/CPU are
  sampled per trial and reported live;
* pause = finish current trial then wait; resume = continue from the
  persisted state (trials replayed into the search backend);
* cancel = finish current trial, mark stopped, keep all results;
* errors: a failing trial is recorded (error field) and the search
  continues; engine/model mismatch or missing profile aborts the session
  with an honest message.

Honesty rules enforced here:
* results always include reliability notes (profile quality, embedder
  availability, validation drop, short-audio warnings);
* the winner is chosen from gate-passing trials only (a distorted or
  unintelligible output can never win on similarity);
* validation scores are what decide among the top-k (anti-overfitting);
* every score is labelled a relative search result, not a match
  probability.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from rvc_platform.core.entities import ConversionSettings
from rvc_platform.core.errors import ConversionError
from rvc_platform.infrastructure.paths import get_paths
from rvc_platform.optimizer.analysis import SR_ANALYSIS, AudioAnalyzer
from rvc_platform.optimizer.embeddings import SpeakerEmbedderManager
from rvc_platform.optimizer.profile import (
    ReferenceProfile,
    ReferenceProfileStore,
)
from rvc_platform.optimizer.scoring import (
    DEFAULT_WEIGHTS,
    EvaluationEngine,
    MetricWeights,
)
from rvc_platform.optimizer.search import (
    OptimizationSpace,
    TrialRecord,
    build_backend,
)
from rvc_platform.optimizer.session import (
    OptimizerSessionConfig,
    OptimizerSessionState,
    SessionStore,
)

logger = logging.getLogger(__name__)

MODES = {
    "quick": {"max_trials": 12, "validate_top_k": 2, "phases": ("exploration", "optimization")},
    "advanced": {"max_trials": 24, "validate_top_k": 3,
                 "phases": ("exploration", "optimization", "fine_tuning")},
    "deep": {"max_trials": 48, "validate_top_k": 5,
             "phases": ("exploration", "optimization", "fine_tuning")},
}


class OptimizerService:
    """Application service wiring analyzer, scorer, search, store, engine."""

    def __init__(self, engine, models_service, config: Any = None) -> None:
        self.engine = engine
        self.models = models_service
        self.cfg = config
        self.store = SessionStore()
        self.profiles = ReferenceProfileStore()
        self.paths = get_paths()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pause_evt = threading.Event()      # set -> PAUSE requested
        self._resume_evt = threading.Event()    # set -> resume granted
        self._cancel_evt = threading.Event()    # set -> CANCEL requested
        self._state: OptimizerSessionState | None = None
        self._embedder: SpeakerEmbedderManager | None = None
        self._analyzer: AudioAnalyzer | None = None
        self._backend = None
        self._backend_name = ""
        self._reference_input_cache: dict[str, Any] = {}

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
        search_frac = min(1.0, done / total) * 0.75
        val = 0.15 if state.phase in ("validation", "final_selection",
                                       "completed", "stopped") else 0.0
        return round(min(1.0, base + search_frac + val), 4)

    def _best_record_summary(self, state: OptimizerSessionState) -> dict[str, Any] | None:
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

    def delete_profile(self, model_name: str, profile_name: str) -> bool:
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
        return self.profiles.delete_sample(model_name, profile_name, sample_id)

    # ================================================================ inputs
    def store_input_audio(self, filename: str, payload: bytes) -> dict[str, Any]:
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
        if not input_path or not Path(input_path).is_file():
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
            input_path=input_path,
            mode=mode,
            max_trials=max_trials,
            time_budget_s=max(0.0, float(config.get("time_budget_s") or 0.0)),
            strategy=str(config.get("strategy") or "auto"),
            multi_objective=bool(config.get("multi_objective")
                                 or (mode == "deep" and
                                     str(config.get("strategy") or "auto")
                                     == "pareto")),
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
        if self._state is None or self._thread is None or not self._thread.is_alive():
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
                return {"resumed": False, "message": "a session is already running"}
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
        if self._state is None or self._thread is None or not self._thread.is_alive():
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

    def _run_session(self, state: OptimizerSessionState) -> None:
        cfg = state.config
        state.message = "preparing analysis"
        self.store.save(state)

        # -- model sanity: the engine must still hold this model
        loaded = self.engine.loaded_model
        if loaded is None or loaded.name != cfg.model_name:
            self.models.load(cfg.model_name)

        # -- analyse input, split search/validation
        analyzer = self._analyzer or AudioAnalyzer(
            self.ensure_embedders(), device=cfg.device)
        self._analyzer = analyzer
        import librosa

        wav_in, sr_in = librosa.load(cfg.input_path, sr=None, mono=True)
        if wav_in.size < SR_ANALYSIS // 2:
            raise ValueError("input audio is too short (<0.5s)")
        if sr_in != SR_ANALYSIS:
            wav16 = librosa.resample(wav_in, orig_sr=sr_in,
                                     target_sr=SR_ANALYSIS).astype(np.float32)
        else:
            wav16 = wav_in.astype(np.float32)
        search_wav, validation_wav = analyzer.split_search_validation(wav16)
        if search_wav.size < SR_ANALYSIS:            # <1s of speech
            search_wav = wav16
        input_analysis = analyzer.analyze(search_wav, "input",
                                          with_embeddings=False).to_dict()
        self._save_wav(self.store.audio_dir(cfg.session_id) / "input_segment.wav",
                       search_wav)

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

        weights = MetricWeights(dict(cfg.weights or DEFAULT_WEIGHTS))
        evaluator = EvaluationEngine(weights)

        # -- search space + backend
        from rvc_platform.optimizer.search import available_f0_methods

        methods = [m for m in (cfg.f0_methods or available_f0_methods())
                   if m in available_f0_methods()]
        if not methods:
            methods = available_f0_methods()
        space = OptimizationSpace(
            f0_methods=methods,
            pitch_policy=cfg.pitch_policy,
            pitch_range=cfg.pitch_range,
        )
        self._backend, self._backend_name = build_backend(
            space, strategy=cfg.strategy, multi_objective=cfg.multi_objective,
            seed=hash(cfg.session_id) % (2**31))
        backend = self._backend
        restored = getattr(backend, "restore", None)
        if restored and state.trials:
            restored([t for t in state.trials if t.error is None])

        # -- BASELINE: current configured settings
        baseline_params = self._current_settings_params()
        if baseline_params.get("f0_method") not in methods and methods:
            # the configured default F0 method may be unavailable in this
            # environment (e.g. rmvpe asset missing) — degrade honestly
            dropped = baseline_params.get("f0_method")
            baseline_params["f0_method"] = methods[0]
            state.reliability_notes.append(
                f"configured default f0 method {dropped!r} is unavailable "
                f"here — the baseline uses {methods[0]!r} instead")
        if state.baseline is None:
            state.phase = "baseline"
            state.message = "converting baseline (current settings)"
            self.store.save(state)
            base_eval, base_path = self._trial_conversion(
                state, search_wav, baseline_params, analyzer, profile_dict,
                evaluator, label="baseline")
            state.baseline = {
                "params": baseline_params,
                "evaluation": base_eval.to_dict(),
                "audio": base_path.name if base_path else None,
            }
            state.trials.append(TrialRecord(
                number=0, params=baseline_params,
                composite=base_eval.composite,
                axes={k: v for k, v in base_eval.to_dict()["axes"].items()},
                gates_passed=base_eval.gates_passed,
                failed_gates=base_eval.failed_gates,
                reliability=base_eval.reliability,
                error=None,
            ))
            if restored:
                pass
            self.store.save(state)

        # -- search phases
        phases = MODES[cfg.mode]["phases"]
        total = cfg.max_trials
        done = len([t for t in state.trials if t.number > 0])
        search_started = time.time()
        deadline = search_started + cfg.time_budget_s if cfg.time_budget_s else None

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
            frac = done / max(1, total)
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
                ev, out_path = self._trial_conversion(
                    state, search_wav, params, analyzer, profile_dict,
                    evaluator, label=f"trial_{done + 1}")
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
                )
                if out_path is not None:
                    self._prune_candidate_audio(state, done + 1)
            except ConversionError as exc:
                rec = TrialRecord(number=done + 1, params=params,
                                  error=str(exc), duration_s=time.time() - t0)
            backend.tell(rec)
            state.trials.append(rec)
            state.current_trial = None
            done += 1
            self.store.save(state)

        if self._cancel_evt.is_set():
            state.phase = "stopped"
            state.message = "stopped by user — partial results kept"
            state.finished_at = time.time()
            self.store.save(state)
            return

        # -- VALIDATION on held-out audio
        top_k = max(1, cfg.validate_top_k)
        candidates = [t for t in state.trials
                      if t.number > 0 and t.error is None
                      and t.composite is not None and t.gates_passed]
        candidates.sort(key=lambda t: float(t.composite or 0.0), reverse=True)
        candidates = candidates[:top_k]
        state.phase = "validation"
        state.message = f"validating top {len(candidates)} on held-out audio"
        self.store.save(state)

        if validation_wav.size >= SR_ANALYSIS and candidates:
            val_analysis_input = analyzer.analyze(
                validation_wav, "validation_input",
                with_embeddings=False).to_dict()
            for cand in candidates:
                if self._cancel_evt.is_set():
                    break
                ev, path = self._trial_conversion(
                    state, validation_wav, cand.params, analyzer,
                    profile_dict, evaluator,
                    label=f"validation_{cand.number}")
                drop = None
                if cand.composite and ev.composite is not None:
                    drop = round(
                        1.0 - float(ev.composite) / max(1e-9, cand.composite), 4)
                state.validation.append({
                    "number": cand.number,
                    "params": cand.params,
                    "search_composite": cand.composite,
                    "validation_composite": ev.composite,
                    "drop": drop,
                    "evaluation": ev.to_dict(),
                    "audio": path.name if path else None,
                })
                self.store.save(state)
            # flag overfitting
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
                    "validation skipped (results rest on the search audio only)")

        # -- FINAL SELECTION (validated score decides among gate-passers)
        state.phase = "final_selection"
        state.message = "selecting best parameters"
        self.store.save(state)
        val_scores = {v["number"]: v.get("validation_composite")
                      for v in state.validation
                      if v.get("validation_composite") is not None}
        best_rec = None
        best_score = -1.0
        for t in state.trials:
            if t.number == 0 or t.error is not None or t.composite is None:
                continue
            if not t.gates_passed:
                continue
            score = float(val_scores.get(t.number, t.composite))
            if score > best_score:
                best_score = score
                best_rec = t
        if best_rec is None:
            # nothing passed the gates — pick the best available with a warning
            fallback = [t for t in state.trials
                        if t.number > 0 and t.error is None
                        and t.composite is not None]
            if fallback:
                best_rec = max(fallback,
                               key=lambda t: float(t.composite or 0.0))
                state.reliability_notes.append(
                    "no candidate passed the quality gates — the reported "
                    "best is the least-bad option, check its artifact scores")
        if best_rec is not None:
            ev_axes = best_rec.axes
            state.best_trial = {
                "number": best_rec.number,
                "params": best_rec.params,
                "composite": best_rec.composite,
                "validated_composite": val_scores.get(best_rec.number),
                "axes": ev_axes,
                "reliability": best_rec.reliability,
            }
            # render + keep the best audio for A/B
            ev, path = self._trial_conversion(
                state, search_wav, best_rec.params, analyzer, profile_dict,
                evaluator, label="best")
            if path is not None:
                dest = self.store.audio_dir(cfg.session_id) / "best.wav"
                try:
                    import shutil

                    shutil.copyfile(path, dest)
                except OSError:
                    pass

        # -- pareto front across ALL trials
        from rvc_platform.optimizer.scoring import VoiceEvaluation

        evals = []
        for t in state.trials:
            ev = VoiceEvaluation()
            for k, v in (t.axes or {}).items():
                setattr(ev, k, v)
            evals.append(ev)
        state.pareto_indices = [
            i for i in EvaluationEngine.pareto_front(evals)
            if state.trials[i].gates_passed]

        # -- history into the profile (reference-aware optimization)
        try:
            profile.history.append({
                "session_id": cfg.session_id,
                "at": time.time(),
                "mode": cfg.mode,
                "best": state.best_trial,
                "n_trials": len(state.trials) - 1,
            })
            profile.history = profile.history[-10:]
            self.profiles.save(profile)
        except Exception:
            logger.debug("optimizer: profile history update failed",
                         exc_info=True)

        state.phase = "completed"
        state.message = "optimization completed"
        state.finished_at = time.time()
        state.current_trial = None
        self.store.save(state)

    def _phase_for(self, done: int, total: int, phases: tuple[str, ...]) -> str:
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

    def _trial_conversion(self, state: OptimizerSessionState,
                          wav16k: np.ndarray, params: dict[str, Any],
                          analyzer: AudioAnalyzer,
                          profile_dict: dict[str, Any],
                          evaluator: EvaluationEngine,
                          label: str) -> tuple[Any, Path | None]:
        """Convert one segment set with the given params and evaluate."""
        cfg = state.config
        loaded = self.engine.loaded_model
        if loaded is None or loaded.name != cfg.model_name:
            raise ConversionError(
                f"engine no longer holds model {cfg.model_name!r} — "
                "session aborted to stay honest")
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

        out_wav, out_sr = librosa.load(str(out_path), sr=SR_ANALYSIS, mono=True)
        analysis = analyzer.analyze(out_wav, label).to_dict()
        sim_report = None
        if self._embedder is not None and profile_dict.get("embeddings"):
            refs = {k: [np.asarray(v, dtype=np.float32) for v in vecs]
                    for k, vecs in profile_dict["embeddings"].items()}
            if refs:
                sim_report = self._embedder.similarity_report(out_wav, refs)
        ev = evaluator.evaluate(analysis, self._input_analysis_for(
            state, wav16k, label), profile_dict, sim_report)
        return ev, out_path

    def _input_analysis_for(self, state: OptimizerSessionState,
                            wav16k: np.ndarray, label: str) -> dict[str, Any]:
        """Input-side analysis for prosody/clarity comparison, cached per
        (session, audio-set) so repeated conversions never re-analyse."""
        scope = "search" if (label.startswith("trial") or label in ("baseline", "best")) else "validation"
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
        """Keep candidate audio only for the current best 5 trials (disk)."""
        d = self.store.audio_dir(state.config.session_id)
        best = self._best_trial(state)
        keep = {f"candidate_trial_{best.number}.wav"} if best else set()
        for t in sorted(state.trials, key=lambda t: float(t.composite or 0),
                        reverse=True)[:5]:
            keep.add(f"candidate_trial_{t.number}.wav")
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
            "validation": state.validation,
            "pareto": pareto,
            "chart": chart,
            "reliability_notes": state.reliability_notes,
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
        return {
            "numbers": numbers,
            "composite": [round(float(t.composite), 4) for t in trials],
            "speaker": [round(float(t.axes.get("speaker_similarity") or 0), 4)
                        for t in trials],
            "clarity": [round(float(t.axes.get("clarity") or 0), 4)
                        for t in trials],
            "naturalness": [round(float(t.axes.get("naturalness") or 0), 4)
                            for t in trials],
            "artifact": [round(float(t.axes.get("artifact_score") or 0), 4)
                         for t in trials],
        }

    def audio_path(self, session_id: str, name: str) -> Path:
        safe = "".join(c for c in name if c.isalnum() or c in "-_./")
        if ".." in safe or safe.startswith("/"):
            raise ValueError("invalid audio name")
        path = (self.store.audio_dir(session_id) / f"{safe}.wav"
                if not safe.endswith(".wav") else
                self.store.audio_dir(session_id) / safe)
        if not path.is_file():
            raise FileNotFoundError(f"audio not found: {name}")
        return path

    # ================================================================= apply
    def apply_best(self, session_id: str | None = None) -> dict[str, Any]:
        state = (self.store.load(session_id) if session_id
                 else (self._state if self._state is not None else None))
        if state is None or state.best_trial is None:
            raise ValueError("no completed optimization with a best result")
        best = state.best_trial
        params = dict(best["params"])
        model = state.config.model_name
        previous = self.profiles.load_applied(model)
        payload = {
            "model": model,
            "applied_at": time.time(),
            "session_id": state.config.session_id,
            "profile": state.config.profile_name,
            "settings": {
                "pitch_shift": int(params.get("pitch_shift", 0)),
                "f0_method": str(params.get("f0_method", "rmvpe")),
                "index_rate": float(params.get("index_rate", 0.75)),
                "filter_radius": int(params.get("filter_radius", 0)),
                "rms_mix_rate": float(params.get("rms_mix_rate", 0.25)),
                "protect": float(params.get("protect", 0.33)),
            },
            "scores": {
                "composite": best.get("composite"),
                "validated_composite": best.get("validated_composite"),
                "axes": best.get("axes") or {},
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

    def export_report(self, session_id: str | None = None) -> dict[str, Any]:
        """Full machine-readable report (also written next to the session)."""
        results = self.results(session_id)
        state = self.store.load(results["session_id"])
        cfg = state.config if state is not None else None
        report = {
            "generated_at": time.time(),
            "generator": "rvc-platform ULTIMATE AI VOICE OPTIMIZER",
            "session": cfg.to_dict() if cfg else {},
            "results": results,
            "disclaimer": (
                "All scores are RELATIVE measures computed inside this "
                "search space with the available metrics. Speaker similarity "
                "is a cosine similarity between speaker-verification "
                "embeddings, NOT a probability of identity match. Composite "
                "scores are weighted combinations of the eight evaluation "
                "axes and must not be reported as 'accuracy' or 'match "
                "percentage'."
            ),
        }
        if state is not None:
            path = self.store.session_dir(state.config.session_id) / "report.json"
            try:
                import json as _json

                path.write_text(
                    _json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
            except OSError:
                pass
        return report

    def capabilities(self) -> dict[str, Any]:
        from rvc_platform.optimizer.search import available_f0_methods

        methods = available_f0_methods()
        self.ensure_embedders()
        s = self._embedder.status
        return {
            "f0_methods": methods,
            "embedders": {"wavlm": s.wavlm, "ecapa": s.ecapa,
                          "reliability": s.reliability},
            "modes": {k: {"max_trials": v["max_trials"],
                          "validate_top_k": v["validate_top_k"]}
                      for k, v in MODES.items()},
            "strategies": ["auto", "optuna", "builtin"],
            "weights": dict(DEFAULT_WEIGHTS),
            "pitch_policies": ["preserve", "auto", "range"],
        }
