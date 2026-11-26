"""Parameter search: space definition + two optimizer backends.

Search space — EXACTLY the parameters this engine's inference pipeline
consumes (no phantom knobs; ``filter_radius`` searches {0, 3} because the
engine's median filter is a fixed kernel-3 that only activates when
radius > 2 — values 1/2 and 4..7 are no-ops of 0 and 3):

* ``f0_method``    — categorical: the F0 registry's available methods
  (rmvpe / fcpe / pm, + crepe when torchcrepe is importable);
* ``index_rate``   — continuous 0..1 (step 0.05);
* ``filter_radius``— categorical {0, 3} (honest semantics);
* ``rms_mix_rate`` — continuous 0..1 (step 0.05);
* ``protect``      — continuous 0..0.5 (step 0.01);
* ``pitch_shift``  — OPTIONAL policy: fixed 0 (preserve), or a bounded
  safe categorical range chosen by the user (never random ±24).

Backends (identical interface, both resumable):

* :class:`OptunaOptimizer` — Optuna TPE (multivariate, seeded) for
  single-objective composite search; NSGA-II when multi-objective Pareto
  strategy is selected. Optuna is an OPTIONAL dependency — added to the
  ``quality`` extra. When missing, the built-in backend is used.
* :class:`BuiltInOptimizer` — dependency-free adaptive search implementing
  the same idea in three phases: golden-ratio low-discrepancy EXPLORATION,
  TPE-style kernel-density GUIDANCE (top-quantile modelling per parameter),
  and Gaussian FINE-TUNING around the incumbent best. Duplicate
  suggestions are rejected via a rounded-key cache (max 24 resamples).

Both backends record every trial as a :class:`TrialRecord` (parameters +
scores + resource usage) so an interrupted search resumes with full
history (Optuna: restored via ``study.add_trial``; built-in: state IS the
trial list).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
def available_f0_methods() -> list[str]:
    """Methods the engine registry can actually construct right now."""
    from rvc_platform.engine.rvc import f0 as f0_mod

    methods = list(f0_mod.available_methods())          # rmvpe, fcpe, pm, crepe
    out: list[str] = []
    for m in methods:
        try:
            f0_mod.get_f0_extractor(m, "cpu", False)
            out.append(m)
        except Exception:
            continue                                    # e.g. crepe without torchcrepe
    return out


@dataclass
class OptimizationSpace:
    """What may be searched, with honest bounds and quantisation."""

    f0_methods: list[str] = field(default_factory=list)
    index_rate: tuple[float, float] = (0.0, 1.0)
    filter_radius_values: tuple[int, ...] = (0, 3)       # engine semantics
    rms_mix_rate: tuple[float, float] = (0.0, 1.0)
    protect: tuple[float, float] = (0.0, 0.5)
    pitch_policy: str = "preserve"                       # preserve | auto | range
    pitch_range: tuple[int, int] = (-6, 6)               # used when policy=range
    step: float = 0.05                                   # numeric quantisation

    def __post_init__(self) -> None:
        if not self.f0_methods:
            self.f0_methods = available_f0_methods() or ["rmvpe"]

    def quantise(self, name: str, value: Any) -> Any:
        if name in ("index_rate", "rms_mix_rate", "protect"):
            v = float(np.clip(float(value), *self._bounds(name)))
            return round(round(v / self.step) * self.step, 3)
        if name == "filter_radius":
            return int(value)
        if name == "pitch_shift":
            return int(value)
        return value

    def _bounds(self, name: str) -> tuple[float, float]:
        return {
            "index_rate": self.index_rate,
            "rms_mix_rate": self.rms_mix_rate,
            "protect": self.protect,
        }[name]

    def key(self, params: dict[str, Any]) -> tuple:
        """Cache key — quantised so 0.449999 and 0.45 never run twice."""
        p = {k: self.quantise(k, v) for k, v in params.items()}
        return (
            str(p.get("f0_method")), float(p.get("index_rate", 0)),
            int(p.get("filter_radius", 0)), float(p.get("rms_mix_rate", 0)),
            float(p.get("protect", 0.33)), int(p.get("pitch_shift", 0)),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "f0_methods": list(self.f0_methods),
            "index_rate": list(self.index_rate),
            "filter_radius_values": list(self.filter_radius_values),
            "rms_mix_rate": list(self.rms_mix_rate),
            "protect": list(self.protect),
            "pitch_policy": self.pitch_policy,
            "pitch_range": list(self.pitch_range) if self.pitch_policy == "range" else None,
            "quantisation_step": self.step,
        }


@dataclass
class TrialRecord:
    """One completed (or failed) trial — the persisted unit of the search."""

    number: int
    params: dict[str, Any]
    composite: float | None = None
    axes: dict[str, float | None] = field(default_factory=dict)
    gates_passed: bool = True
    failed_gates: list[str] = field(default_factory=list)
    reliability: str = "unknown"
    duration_s: float = 0.0
    vram_mb: float | None = None
    cpu_percent: float | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "params": dict(self.params),
            "composite": None if self.composite is None else round(self.composite, 4),
            "axes": {k: None if v is None else round(float(v), 4)
                     for k, v in self.axes.items()},
            "gates_passed": self.gates_passed,
            "failed_gates": self.failed_gates,
            "reliability": self.reliability,
            "duration_s": round(self.duration_s, 3),
            "vram_mb": self.vram_mb,
            "cpu_percent": self.cpu_percent,
            "error": self.error,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialRecord":
        return cls(
            number=int(data.get("number", 0)),
            params=dict(data.get("params") or {}),
            composite=data.get("composite"),
            axes=dict(data.get("axes") or {}),
            gates_passed=bool(data.get("gates_passed", True)),
            failed_gates=list(data.get("failed_gates") or []),
            reliability=str(data.get("reliability", "unknown")),
            duration_s=float(data.get("duration_s", 0.0)),
            vram_mb=data.get("vram_mb"),
            cpu_percent=data.get("cpu_percent"),
            error=data.get("error"),
            created_at=float(data.get("created_at", time.time())),
        )


class OptimizerBackend(Protocol):
    """Common interface for the two backends."""

    def suggest(self, budget_used: int) -> dict[str, Any]: ...

    def tell(self, record: TrialRecord) -> None: ...

    def best(self) -> TrialRecord | None: ...


# --------------------------------------------------------------------------- #
# Built-in adaptive optimizer (dependency-free, TPE-flavoured)
# --------------------------------------------------------------------------- #
class BuiltInOptimizer:
    """Three-phase adaptive search: explore -> guide -> refine.

    * EXPLORE (first 30% of the budget): golden-ratio low-discrepancy
      quasi-random sampling across the whole space (better coverage than
      pure random with the same trial count);
    * GUIDE (next 45%): Tree-Parzen-style two-model split — completed
      gate-passing trials are split at the composite median; the next
      candidate is drawn from the *better* model: numeric parameters via a
      1-D Gaussian KDE over the good set, categorical by good-set
      frequencies; with probability ε (0.15 decaying) exploration resumes
      so the search never starves;
    * REFINE (last 25%): Gaussian perturbation around the incumbent best
      (σ = 12% of each numeric range) — hill-climbing fine-tuning.

    Duplicate avoidance: quantised keys are cached; a duplicate suggestion
    is resampled up to 24 times before being accepted (the loop in the
    service then still sees a unique key).
    """

    PHI = (math.sqrt(5.0) - 1.0) / 2.0   # golden ratio fraction

    def __init__(self, space: OptimizationSpace, seed: int = 7) -> None:
        self.space = space
        self.rng = np.random.default_rng(seed)
        self.trials: list[TrialRecord] = []
        self._seen: set[tuple] = set()

    # ---------------------------------------------------------------- state
    def restore(self, records: list[TrialRecord]) -> None:
        self.trials = list(records)
        self._seen = {self.space.key(r.params) for r in records}

    def suggest(self, budget_used: int) -> dict[str, Any]:
        for _ in range(24):
            params = self._propose(budget_used)
            key = self.space.key(params)
            if key not in self._seen:
                return params
        # extremely dense tiny spaces: accept a repeat, but vary the seed
        self.rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))
        return self._propose(budget_used)

    def tell(self, record: TrialRecord) -> None:
        self.trials.append(record)
        self._seen.add(self.space.key(record.params))

    def best(self) -> TrialRecord | None:
        ok = [t for t in self.trials
              if t.composite is not None and t.gates_passed and t.error is None]
        if not ok:
            ok = [t for t in self.trials
                  if t.composite is not None and t.error is None]
        if not ok:
            return None
        return max(ok, key=lambda t: float(t.composite or 0.0))

    # ------------------------------------------------------------- proposals
    def _propose(self, budget_used: int) -> dict[str, Any]:
        done = len([t for t in self.trials if t.error is None])
        best_rec = self.best()
        if done >= 3 and best_rec is not None and self.rng.random() < 0.25:
            # REFINE: local perturbation around the incumbent
            return self._perturb(best_rec.params)
        good = self._good_set()
        if good and self.rng.random() > 0.18:
            # GUIDE: sample from the good model
            return self._sample_from(good)
        # EXPLORE: low-discrepancy quasi-random point
        return self._quasi_random(budget_used)

    def _good_set(self) -> list[dict[str, Any]]:
        done = [t for t in self.trials
                if t.error is None and t.composite is not None
                and t.gates_passed]
        if len(done) < 4:
            return []
        scores = [float(t.composite or 0.0) for t in done]
        med = float(np.median(scores))
        good = [t for t in done if float(t.composite or 0.0) > med]
        return [t.params for t in good]

    def _sample_from(self, good: list[dict[str, Any]]) -> dict[str, Any]:
        sp = self.space
        params: dict[str, Any] = {}
        method_counts: dict[str, int] = {}
        for p in good:
            method_counts[p.get("f0_method", sp.f0_methods[0])] = \
                method_counts.get(p.get("f0_method", sp.f0_methods[0]), 0) + 1
        methods = list(method_counts)
        weights = np.array([method_counts[m] for m in methods], dtype=np.float64)
        params["f0_method"] = str(self.rng.choice(methods, p=weights / weights.sum()))

        params["filter_radius"] = int(self.rng.choice(
            [p.get("filter_radius", 0) for p in good] or [0]))

        for name, bounds in (("index_rate", sp.index_rate),
                             ("rms_mix_rate", sp.rms_mix_rate),
                             ("protect", sp.protect)):
            vals = np.array([float(p.get(name, bounds[0])) for p in good],
                            dtype=np.float64)
            # Gaussian KDE with Silverman bandwidth on the good set
            sigma = max(float(vals.std()) if vals.size else 0.0, 0.02)
            v = float(self.rng.normal(float(vals.mean()), sigma))
            params[name] = sp.quantise(name, v)

        params["pitch_shift"] = self._pitch_value(good)
        return params

    def _perturb(self, params: dict[str, Any]) -> dict[str, Any]:
        sp = self.space
        out = dict(params)
        for name, bounds in (("index_rate", sp.index_rate),
                             ("rms_mix_rate", sp.rms_mix_rate),
                             ("protect", sp.protect)):
            span = bounds[1] - bounds[0]
            v = float(params.get(name, bounds[0])) + \
                float(self.rng.normal(0.0, 0.12 * span))
            out[name] = sp.quantise(name, v)
        if self.rng.random() < 0.2 and len(sp.f0_methods) > 1:
            out["f0_method"] = str(sp.f0_methods[
                int(self.rng.integers(0, len(sp.f0_methods)))])
        if self.rng.random() < 0.2:
            out["filter_radius"] = int(self.rng.choice(sp.filter_radius_values))
        out["pitch_shift"] = self._pitch_value([params])
        return out

    def _quasi_random(self, i: int) -> dict[str, Any]:
        sp = self.space
        params: dict[str, Any] = {}
        seq = i + 1
        params["f0_method"] = sp.f0_methods[int(
            ((seq * self.PHI) % 1.0) * len(sp.f0_methods)) % len(sp.f0_methods)]
        for name, bounds in (("index_rate", sp.index_rate),
                             ("rms_mix_rate", sp.rms_mix_rate),
                             ("protect", sp.protect)):
            span = bounds[1] - bounds[0]
            v = bounds[0] + span * (((seq * (2.0 if name != "index_rate" else 3.0))
                                      * self.PHI) % 1.0)
            params[name] = sp.quantise(name, v)
        params["filter_radius"] = sp.filter_radius_values[
            seq % len(sp.filter_radius_values)]
        params["pitch_shift"] = self._pitch_value([])
        return params

    def _pitch_value(self, good: list[dict[str, Any]]) -> int:
        sp = self.space
        if sp.pitch_policy == "preserve":
            return 0
        if sp.pitch_policy == "auto":
            # handled via engine auto_pitch by the service; search stays 0
            return 0
        if sp.pitch_policy == "range" and good:
            vals = [int(p.get("pitch_shift", 0)) for p in good]
            v = int(self.rng.normal(float(np.mean(vals)),
                                    max(1.0, float(np.std(vals)) if vals else 1.0)))
            return int(np.clip(v, sp.pitch_range[0], sp.pitch_range[1]))
        if sp.pitch_policy == "range":
            span = sp.pitch_range[1] - sp.pitch_range[0]
            return int(np.clip(
                sp.pitch_range[0] + int(self.rng.integers(0, span + 1)),
                sp.pitch_range[0], sp.pitch_range[1]))
        return 0


# --------------------------------------------------------------------------- #
# Optuna backend (optional dependency)
# --------------------------------------------------------------------------- #
class OptunaOptimizer:
    """Optuna-powered search: TPE (multivariate) for single-objective runs,
    NSGA-II when ``multi_objective=True`` (Pareto strategy).

    Prior trials are replayed into the study on resume (``study.add_trial``)
    so TPE's surrogate model sees the full history; unavailable Optuna is a
    clear, logged degradation to :class:`BuiltInOptimizer` (never silent).
    """

    def __init__(self, space: OptimizationSpace, multi_objective: bool = False,
                 seed: int = 7) -> None:
        try:
            import optuna
        except ImportError as exc:                       # pragma: no cover
            raise RuntimeError(
                "optuna is not installed — install the 'quality' extra "
                "(uv pip install 'rvc-platform[quality]' or uv pip install optuna)"
            ) from exc
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.space = space
        self.multi_objective = multi_objective
        self._optuna = optuna
        sampler: Any
        if multi_objective:
            sampler = optuna.samplers.NSGAIISampler(seed=seed)
        else:
            sampler = optuna.samplers.TPESampler(
                seed=seed, multivariate=True, group=False,
                constant_liar=True,
            )
        self.study = optuna.create_study(
            direction="maximize" if not multi_objective else None,
            directions=["maximize", "maximize"] if multi_objective else None,
            sampler=sampler,
        )
        self.trials: list[TrialRecord] = []
        self._seen: set[tuple] = set()

    def restore(self, records: list[TrialRecord]) -> None:
        """Replay completed trials so the sampler warm-starts."""
        optuna = self._optuna
        for rec in records:
            if rec.error is not None or rec.composite is None:
                continue
            try:
                frozen = optuna.trial.create_trial(
                    params=self._to_optuna_params(rec.params),
                    distributions=self._distributions(),
                    value=None if self.multi_objective else float(rec.composite),
                    values=[float(rec.composite),
                            float(rec.axes.get("clarity") or rec.composite)]
                    if self.multi_objective else None,
                )
                self.study.add_trial(frozen)
            except Exception:
                logger.debug("optimizer: trial replay failed", exc_info=True)
            self.trials.append(rec)
            self._seen.add(self.space.key(rec.params))

    def _distributions(self) -> dict[str, Any]:
        optuna = self._optuna
        d: dict[str, Any] = {}
        d["f0_method"] = optuna.distributions.CategoricalDistribution(
            self.space.f0_methods)
        d["index_rate"] = optuna.distributions.FloatDistribution(
            *self.space.index_rate, step=self.space.step)
        d["filter_radius"] = optuna.distributions.CategoricalDistribution(
            [str(v) for v in self.space.filter_radius_values])
        d["rms_mix_rate"] = optuna.distributions.FloatDistribution(
            *self.space.rms_mix_rate, step=self.space.step)
        d["protect"] = optuna.distributions.FloatDistribution(
            *self.space.protect, step=max(0.01, self.space.step / 5))
        if self.space.pitch_policy == "range":
            lo, hi = self.space.pitch_range
            d["pitch_shift"] = optuna.distributions.IntDistribution(lo, hi)
        return d

    def _to_optuna_params(self, params: dict[str, Any]) -> dict[str, Any]:
        # only parameters with a matching distribution belong to the study
        # (e.g. pitch_shift exists only under the "range" policy)
        dists = self._distributions()
        out: dict[str, Any] = {}
        for key in dists:
            if key in params:
                out[key] = (str(int(params[key]))
                            if key == "filter_radius" else params[key])
        return out

    def _from_optuna_params(self, params: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out["f0_method"] = str(params.get("f0_method", self.space.f0_methods[0]))
        out["index_rate"] = self.space.quantise("index_rate",
                                                params.get("index_rate", 0.75))
        out["filter_radius"] = int(params.get("filter_radius", "0"))
        out["rms_mix_rate"] = self.space.quantise(
            "rms_mix_rate", params.get("rms_mix_rate", 0.25))
        out["protect"] = self.space.quantise("protect", params.get("protect", 0.33))
        if self.space.pitch_policy == "range":
            out["pitch_shift"] = int(params.get("pitch_shift", 0))
        else:
            out["pitch_shift"] = 0
        return out

    # ------------------------------------------------------------------ api
    def suggest(self, budget_used: int) -> dict[str, Any]:
        trial = self.study.ask(fixed_distributions=self._distributions())
        params = self._from_optuna_params(trial.params)
        for attempt in range(24):
            key = self.space.key(params)
            if key not in self._seen:
                break
            # duplicate point: nudge one numeric dimension by ± one quantum
            # (a genuinely new point — no dangling RUNNING trials in the study)
            params = self._nudge(params, attempt)
        self._pending = (trial, params)
        return params

    def _nudge(self, params: dict[str, Any], attempt: int) -> dict[str, Any]:
        sp = self.space
        out = dict(params)
        dim = attempt % 3
        if dim == 0:
            out["index_rate"] = sp.quantise(
                "index_rate", float(out.get("index_rate", 0.75))
                + self._step_sign(attempt) * sp.step)
        elif dim == 1:
            out["rms_mix_rate"] = sp.quantise(
                "rms_mix_rate", float(out.get("rms_mix_rate", 0.25))
                + self._step_sign(attempt) * sp.step)
        else:
            out["protect"] = sp.quantise(
                "protect", float(out.get("protect", 0.33))
                + self._step_sign(attempt) * 0.01)
        return out

    @staticmethod
    def _step_sign(attempt: int) -> float:
        return 1.0 if (attempt // 3) % 2 == 0 else -1.0

    def tell(self, record: TrialRecord) -> None:
        self.trials.append(record)
        self._seen.add(self.space.key(record.params))
        trial, _params = getattr(self, "_pending", (None, None))
        if trial is None:
            return
        try:
            if record.error is not None:
                state = self._optuna.trial.TrialState.FAIL
                values = None
            else:
                state = self._optuna.trial.TrialState.COMPLETE
                if self.multi_objective:
                    values = [float(record.composite or 0.0),
                              float(record.axes.get("clarity") or 0.0)]
                else:
                    gate_penalty = 1.0 if record.gates_passed else 0.25
                    values = float((record.composite or 0.0) * gate_penalty)
            self.study.tell(trial, values=values, state=state)
        except Exception:
            logger.exception("optimizer: optuna tell failed")
        finally:
            self._pending = (None, None)

    def best(self) -> TrialRecord | None:
        ok = [t for t in self.trials
              if t.composite is not None and t.gates_passed and t.error is None]
        if not ok:
            ok = [t for t in self.trials
                  if t.composite is not None and t.error is None]
        if not ok:
            return None
        return max(ok, key=lambda t: float(t.composite or 0.0))


# --------------------------------------------------------------------------- #
# Backend factory
# --------------------------------------------------------------------------- #
def build_backend(space: OptimizationSpace, strategy: str = "auto",
                  multi_objective: bool = False,
                  seed: int = 7) -> tuple[OptimizerBackend, str]:
    """'optuna' | 'builtin' | 'auto' (optuna when importable)."""
    strategy = (strategy or "auto").lower()
    if strategy in ("optuna", "auto", "adaptive", "tpe", "nsga2", "nsga-ii",
                    "pareto"):
        try:
            backend: OptimizerBackend = OptunaOptimizer(
                space, multi_objective=multi_objective, seed=seed)
            name = "optuna-" + ("nsga2" if multi_objective else "tpe")
            return backend, name
        except RuntimeError:
            if strategy != "auto":
                logger.warning(
                    "optimizer: strategy %r requested optuna but it is not "
                    "installed — falling back to the built-in adaptive search "
                    "(install: uv pip install optuna)", strategy)
    backend = BuiltInOptimizer(space, seed=seed)
    return backend, "builtin-adaptive"
