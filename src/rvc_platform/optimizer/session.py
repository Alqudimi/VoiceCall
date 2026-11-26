"""Optimization session state + crash-safe persistence.

A session is the unit of resumability: every completed trial is appended to
``data/optimizer/sessions/<id>/state.json`` (atomic write, whole file) the
moment its evaluation finishes, so an app restart / power loss never loses
more than the trial in flight. On boot the service marks unfinished sessions
``interrupted``; ``resume`` continues them exactly where they stopped
(replaying the trial log into the search backend).

Files per session::

    <id>/state.json     — config + phase + trials + validation + best
    <id>/audio/         — baseline.wav, best.wav, candidate_<n>.wav,
                          validation_<n>.wav, input_segment.wav
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rvc_platform.optimizer.search import TrialRecord

logger = logging.getLogger(__name__)

PHASES = (
    "baseline",        # convert with current settings -> the reference point
    "exploration",     # first broad sweep of the space
    "optimization",    # model-guided sampling of promising regions
    "fine_tuning",     # local refinement around the best candidates
    "validation",      # held-out re-test of top candidates
    "final_selection", # pick + record the winner
    "completed",
    "failed",
    "stopped",
    "paused",
)


@dataclass
class OptimizerSessionConfig:
    """Everything that defines one optimization run."""

    session_id: str = ""
    model_name: str = ""
    model_path: str = ""
    index_path: str | None = None
    profile_name: str = "default"
    input_path: str = ""                # stored uploaded input audio
    mode: str = "advanced"               # quick | advanced | deep
    max_trials: int = 24
    time_budget_s: float = 0.0          # 0 = unlimited (trials-only budget)
    strategy: str = "auto"              # auto | optuna | builtin
    multi_objective: bool = False       # NSGA-II/Pareto strategy (deep mode)
    pitch_policy: str = "preserve"      # preserve | auto | range
    pitch_range: tuple[int, int] = (-6, 6)
    weights: dict[str, float] = field(default_factory=dict)
    denoise: bool = False               # fixed conversion pre-pass settings
    normalize: bool = True
    f0_methods: list[str] = field(default_factory=list)
    search_segments: int = 1            # audio segments used per trial (speed)
    validate_top_k: int = 3
    embedder_prefer: str = "auto"       # auto | wavlm | ecapa | both
    device: str = "cpu"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "index_path": self.index_path,
            "profile_name": self.profile_name,
            "input_path": self.input_path,
            "mode": self.mode,
            "max_trials": self.max_trials,
            "time_budget_s": self.time_budget_s,
            "strategy": self.strategy,
            "multi_objective": self.multi_objective,
            "pitch_policy": self.pitch_policy,
            "pitch_range": list(self.pitch_range),
            "weights": dict(self.weights),
            "denoise": self.denoise,
            "normalize": self.normalize,
            "f0_methods": list(self.f0_methods),
            "search_segments": self.search_segments,
            "validate_top_k": self.validate_top_k,
            "embedder_prefer": self.embedder_prefer,
            "device": self.device,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizerSessionConfig":
        return cls(
            session_id=str(data.get("session_id", "")),
            model_name=str(data.get("model_name", "")),
            model_path=str(data.get("model_path", "")),
            index_path=data.get("index_path"),
            profile_name=str(data.get("profile_name", "default")),
            input_path=str(data.get("input_path", "")),
            mode=str(data.get("mode", "advanced")),
            max_trials=int(data.get("max_trials", 24)),
            time_budget_s=float(data.get("time_budget_s", 0.0)),
            strategy=str(data.get("strategy", "auto")),
            multi_objective=bool(data.get("multi_objective", False)),
            pitch_policy=str(data.get("pitch_policy", "preserve")),
            pitch_range=tuple(data.get("pitch_range", (-6, 6))),
            weights=dict(data.get("weights") or {}),
            denoise=bool(data.get("denoise", False)),
            normalize=bool(data.get("normalize", True)),
            f0_methods=list(data.get("f0_methods") or []),
            search_segments=int(data.get("search_segments", 1)),
            validate_top_k=int(data.get("validate_top_k", 3)),
            embedder_prefer=str(data.get("embedder_prefer", "auto")),
            device=str(data.get("device", "cpu")),
            created_at=float(data.get("created_at", time.time())),
        )


@dataclass
class OptimizerSessionState:
    """Mutable live state of a session (persisted after every trial)."""

    config: OptimizerSessionConfig = field(default_factory=OptimizerSessionConfig)
    phase: str = "baseline"
    started_at: float = 0.0
    updated_at: float = 0.0
    finished_at: float | None = None
    trials: list[TrialRecord] = field(default_factory=list)
    baseline: dict[str, Any] | None = None       # baseline params + evaluation
    best_trial: dict[str, Any] | None = None     # final winner summary
    validation: list[dict[str, Any]] = field(default_factory=list)
    pareto_indices: list[int] = field(default_factory=list)
    current_trial: dict[str, Any] | None = None  # live in-flight trial (info only)
    message: str = ""
    reliability_notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "phase": self.phase,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "trials": [t.to_dict() for t in self.trials],
            "baseline": self.baseline,
            "best_trial": self.best_trial,
            "validation": self.validation,
            "pareto_indices": self.pareto_indices,
            "current_trial": self.current_trial,
            "message": self.message,
            "reliability_notes": self.reliability_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizerSessionState":
        return cls(
            config=OptimizerSessionConfig.from_dict(data.get("config") or {}),
            phase=str(data.get("phase", "baseline")),
            started_at=float(data.get("started_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
            finished_at=data.get("finished_at"),
            trials=[TrialRecord.from_dict(t)
                    for t in (data.get("trials") or [])],
            baseline=data.get("baseline"),
            best_trial=data.get("best_trial"),
            validation=list(data.get("validation") or []),
            pareto_indices=list(data.get("pareto_indices") or []),
            current_trial=data.get("current_trial"),
            message=str(data.get("message", "")),
            reliability_notes=list(data.get("reliability_notes") or []),
        )


class SessionStore:
    """Crash-safe session persistence (atomic whole-file writes)."""

    def __init__(self) -> None:
        from rvc_platform.infrastructure.paths import get_paths

        self.root = get_paths().data / "optimizer" / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self._io_lock = threading.Lock()

    def session_dir(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_") or "s"
        d = self.root / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def audio_dir(self, session_id: str) -> Path:
        d = self.session_dir(session_id) / "audio"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, state: OptimizerSessionState) -> None:
        state.updated_at = time.time()
        path = self.session_dir(state.config.session_id) / "state.json"
        tmp = path.with_suffix(".json.tmp")
        with self._io_lock:
            try:
                tmp.write_text(
                    json.dumps(state.to_dict(), ensure_ascii=False),
                    encoding="utf-8")
                tmp.replace(path)                      # atomic on POSIX
            except OSError:
                logger.exception("optimizer: session save failed")
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def load(self, session_id: str) -> OptimizerSessionState | None:
        path = self.session_dir(session_id) / "state.json"
        if not path.is_file():
            return None
        try:
            return OptimizerSessionState.from_dict(
                json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, ValueError):
            logger.exception("optimizer: session %s unreadable", session_id)
            return None

    def list_sessions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return out
        for d in sorted(self.root.iterdir(), key=lambda p: p.name, reverse=True):
            state = self.load(d.name)
            if state is None:
                continue
            done = len([t for t in state.trials])
            out.append({
                "session_id": state.config.session_id,
                "model": state.config.model_name,
                "profile": state.config.profile_name,
                "mode": state.config.mode,
                "phase": state.phase,
                "n_trials": done,
                "max_trials": state.config.max_trials,
                "best_composite": (state.best_trial or {}).get("composite"),
                "started_at": state.started_at,
                "updated_at": state.updated_at,
            })
        return out

    @staticmethod
    def new_session_id() -> str:
        return f"opt_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
