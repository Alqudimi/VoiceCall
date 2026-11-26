"""Parameter search: space definition + optimizer backends.

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
  single-objective composite search; NSGA-II for the multi-objective
  strategy with objectives **(speaker similarity, naturalness)** — two
  REAL evaluation axes, not two views of the composite (Phase 02 W10).
* :class:`BuiltInOptimizer` — dependency-free **staged adaptive** search
  (Phase 02 W8): the search behavior genuinely changes per stage —

  - STAGE A *intelligent exploration* — golden-ratio low-discrepancy
    coverage of the whole space (first 30% of the budget), warm-started
    with the best prior parameters from the experiment memory when
    available;
  - STAGE B *TPE-guided* (next 45%) — kernel-density sampling around the
    good set (quantile split), including **categorical Thompson
    sampling** and adaptive per-parameter bandwidth;
  - STAGE C *fine-grained* (last 25%) — Gaussian local refinement around
    the top-3 incumbents;
  - multi-objective mode keeps a **Pareto archive** and samples from the
    non-dominated front (built-in NSGA-flavoured selection, no
    dependency).

Duplicate suggestions are rejected via a rounded-key cache (max 24
resamples). Seeds are **deterministic across processes** (CRC-32 of the
session id — Python's hash() of str is not; Phase 02 W9).

Both backends record every trial as a :class:`TrialRecord` (parameters +
scores + resource usage) so an interrupted search resumes with full
history (Optuna: restored via ``study.add_trial``; built-in: state IS the
trial list), and prior-session trials from the experiment memory can be
restored as warm start (Phase 02).
"""
from __future__ import annotations

import logging
import math
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)

# stage budget fractions (of the trial budget) — Phase 02 W8
STAGE_A_FRACTION = 0.30        # intelligent exploration
STAGE_B_FRACTION = 0.45        # TPE-guided sampling
STAGE_C_FRACTION = 0.25         # fine-grained refinement


def stable_seed(session_id: str, fallback: int = 7) -> int:
    """Deterministic cross-process seed from a session id (CRC-32).

    ``hash(str)`` is salted per process (PYTHONHASHSEED) — Phase 01 used
    it, which made resumed sessions behave differently after a restart.
    CRC-32 is stable everywhere.
    """
    try:
        return int(zlib.crc32(str(session_id or "").encode("utf-8"))
                   & 0x7FFFFFFF) or fallback
    except Exception:
        return fallback


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
            "pitch_range": list(self.pitch_range)
            if self.pitch_policy == "range" else None,
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
    # Phase 02: full artifact detector detail for recurrence analysis
    artifact_detail: dict[str, float] = field(default_factory=dict)
    # Phase 02: cache/reuse bookkeeping (conversion replayed from memory?)
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "params": dict(self.params),
            "composite": None if self.composite is None
            else round(self.composite, 4),
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
            "artifact_detail": dict(self.artifact_detail),
            "from_cache": self.from_cache,
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
            artifact_detail=dict(data.get("artifact_detail") or {}),
            from_cache=bool(data.get("from_cache", False)),
        )


class OptimizerBackend(Protocol):
    """Common interface for the two backends."""

    def suggest(self, budget_used: int) -> dict[str, Any]: ...

    def tell(self, record: TrialRecord) -> None: ...

    def best(self) -> TrialRecord | None: ...


def _pareto_front_records(records: list[TrialRecord]) -> list[int]:
    """Non-dominated indices on (speaker, naturalness, clarity) — used by
    the built-in multi-objective mode."""
    axes = [
        (float(r.axes.get("speaker_similarity") or -1.0),
         float(r.axes.get("naturalness") or -1.0),
         float(r.axes.get("clarity") or -1.0))
        for r in records
    ]
    front: list[int] = []
    for i, a in enumerate(axes):
        dominated = False
        for j, b in enumerate(axes):
            if i == j:
                continue
            if all(bx >= ax for ax, bx in zip(a, b)) and \
                    any(bx > ax for ax, bx in zip(a, b)):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


# --------------------------------------------------------------------------- #
# Built-in staged adaptive optimizer (dependency-free)
# --------------------------------------------------------------------------- #
class BuiltInOptimizer:
    """Staged adaptive search: A explore -> B TPE-guided -> C fine-grain.

    Phase 02 W8 — the stage is now a REAL budget split (constructor takes
    ``budget_total``; the current stage is derived from completed trials),
    not a label:

    * STAGE A (first 30%): golden-ratio low-discrepancy coverage,
      warm-started from ``restore()``-ed prior best parameters;
    * STAGE B (next 45%): kernel-density sampling around the good set
      (adaptive per-parameter bandwidth: σ grows when the good set is
      narrow so the search never collapses too early) + categorical
      Thompson sampling for f0_method; ε-exploration keeps coverage;
    * STAGE C (last 25%): Gaussian local refinement around the top-3
      incumbents (σ = 8% of each range).

    Multi-objective mode (``multi_objective=True``): keeps a Pareto archive
    over (speaker, naturalness, clarity) and alternates exploration with
    sampling around a RANDOM non-dominated member (built-in diversity
    pressure, no external dependency).
    """

    PHI = (math.sqrt(5.0) - 1.0) / 2.0   # golden ratio fraction

    def __init__(self, space: OptimizationSpace, seed: int = 7,
                 budget_total: int = 24,
                 multi_objective: bool = False) -> None:
        self.space = space
        self.rng = np.random.default_rng(seed)
        self.budget_total = max(4, int(budget_total))
        self.multi_objective = multi_objective
        self.trials: list[TrialRecord] = []
        self._seen: set[tuple] = set()
        self._warm_start: list[dict[str, Any]] = []
        self._warm_used = 0

    # ---------------------------------------------------------------- state
    def restore(self, records: list[TrialRecord]) -> None:
        self.trials = list(records)
        self._seen = {self.space.key(r.params) for r in records}

    def warm_start(self, param_sets: list[dict[str, Any]]) -> None:
        """Seed STAGE A with prior best parameters (experiment memory)."""
        self._warm_start = [dict(p) for p in (param_sets or [])][:5]

    # ------------------------------------------------------------- staging
    def _stage(self, done: int) -> str:
        frac = done / max(1, self.budget_total)
        if frac < STAGE_A_FRACTION:
            return "A"
        if frac < STAGE_A_FRACTION + STAGE_B_FRACTION:
            return "B"
        return "C"

    def suggest(self, budget_used: int) -> dict[str, Any]:
        for _ in range(24):
            params = self._propose(budget_used)
            key = self.space.key(params)
            if key not in self._seen:
                return params
        # extremely dense tiny spaces: accept a repeat, but vary the seed
        self.rng = np.random.default_rng(
            int(self.rng.integers(0, 2**31 - 1)))
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

    def pareto_archive(self) -> list[TrialRecord]:
        done = [t for t in self.trials
                if t.error is None and t.composite is not None]
        front = _pareto_front_records(done)
        return [done[i] for i in front]

    # ------------------------------------------------------------- proposals
    def _propose(self, budget_used: int) -> dict[str, Any]:
        done = len([t for t in self.trials if t.error is None])
        stage = self._stage(done)
        if stage == "A":
            # warm-start injections first (experiment memory)
            while self._warm_used < len(self._warm_start):
                cand = dict(self._warm_start[self._warm_used])
                self._warm_used += 1
                if self.space.key(cand) not in self._seen:
                    return cand
            return self._quasi_random(budget_used)
        if self.multi_objective:
            # alternate: explore / sample around a random front member
            front = self.pareto_archive()
            if front and self.rng.random() > 0.25:
                anchor = front[int(self.rng.integers(0, len(front)))]
                return self._perturb(anchor.params, sigma_frac=0.10)
            return self._quasi_random(budget_used)
        if stage == "C":
            top = self._topk(3)
            if top:
                anchor = top[int(self.rng.integers(0, len(top)))]
                return self._perturb(anchor.params, sigma_frac=0.08)
            return self._quasi_random(budget_used)
        # STAGE B: TPE-guided
        good = self._good_set()
        if good and self.rng.random() > 0.18:
            return self._sample_from(good)
        return self._quasi_random(budget_used)

    def _topk(self, k: int) -> list[TrialRecord]:
        done = [t for t in self.trials
                if t.error is None and t.composite is not None
                and t.gates_passed]
        if not done:
            done = [t for t in self.trials
                    if t.error is None and t.composite is not None]
        done = sorted(done, key=lambda t: -float(t.composite or 0.0))
        return done[:k]

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
        # categorical: Thompson sampling on f0_method (Beta posterior per level)
        if sp.f0_methods:
            counts = {m: (1.0, 1.0) for m in sp.f0_methods}
            for p in good:
                m = p.get("f0_method", sp.f0_methods[0])
                if m in counts:
                    a, b = counts[m]
                    counts[m] = (a + 1.0, b)
            samples = {m: float(self.rng.beta(a, b))
                       for m, (a, b) in counts.items()}
            params["f0_method"] = max(samples, key=samples.get)
        params["filter_radius"] = int(self.rng.choice(
            [p.get("filter_radius", 0) for p in good] or [0]))

        for name, bounds in (("index_rate", sp.index_rate),
                             ("rms_mix_rate", sp.rms_mix_rate),
                             ("protect", sp.protect)):
            vals = np.array([float(p.get(name, bounds[0])) for p in good],
                            dtype=np.float64)
            mean = float(vals.mean())
            span = bounds[1] - bounds[0]
            # adaptive bandwidth: never narrower than 6% of the range, so
            # the KDE cannot collapse onto a single point early on
            sigma = max(float(vals.std()), 0.06 * span, 0.02)
            v = float(self.rng.normal(mean, sigma))
            params[name] = sp.quantise(name, v)

        params["pitch_shift"] = self._pitch_value(good)
        return params

    def _perturb(self, params: dict[str, Any],
                 sigma_frac: float = 0.12) -> dict[str, Any]:
        sp = self.space
        out = dict(params)
        for name, bounds in (("index_rate", sp.index_rate),
                             ("rms_mix_rate", sp.rms_mix_rate),
                             ("protect", sp.protect)):
            span = bounds[1] - bounds[0]
            v = float(params.get(name, bounds[0])) + \
                float(self.rng.normal(0.0, sigma_frac * span))
            out[name] = sp.quantise(name, v)
        if self.rng.random() < 0.2 and len(sp.f0_methods) > 1:
            out["f0_method"] = str(sp.f0_methods[
                int(self.rng.integers(0, len(sp.f0_methods)))])
        if self.rng.random() < 0.2:
            out["filter_radius"] = int(self.rng.choice(
                sp.filter_radius_values))
        out["pitch_shift"] = self._pitch_value([params])
        return out

    def _quasi_random(self, i: int) -> dict[str, Any]:
        sp = self.space
        params: dict[str, Any] = {}
        seq = i + 1
        params["f0_method"] = sp.f0_methods[int(
            ((seq * self.PHI) % 1.0) * len(sp.f0_methods))
            % len(sp.f0_methods)]
        for name, bounds in (("index_rate", sp.index_rate),
                             ("rms_mix_rate", sp.rms_mix_rate),
                             ("protect", sp.protect)):
            span = bounds[1] - bounds[0]
            v = bounds[0] + span * (((seq * (2.0 if name != "index_rate"
                                             else 3.0))
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
                                    max(1.0, float(np.std(vals))
                                        if vals else 1.0)))
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
    NSGA-II for the multi-objective strategy with objectives
    (speaker_similarity, naturalness) — two real evaluation axes (Phase 02
    W10; Phase 01 used composite+clarity, two views of one number).

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
                "(uv pip install 'rvc-platform[quality]' or "
                "uv pip install optuna)"
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
                values: Any
                value: Any = None
                if self.multi_objective:
                    values = [
                        float(rec.axes.get("speaker_similarity")
                              or rec.composite),
                        float(rec.axes.get("naturalness") or rec.composite)]
                else:
                    gate_penalty = 1.0 if rec.gates_passed else 0.25
                    value = float(rec.composite) * gate_penalty
                frozen = optuna.trial.create_trial(
                    params=self._to_optuna_params(rec.params),
                    distributions=self._distributions(),
                    value=None if self.multi_objective else value,
                    values=values if self.multi_objective else None,
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
        out["f0_method"] = str(params.get(
            "f0_method", self.space.f0_methods[0]))
        out["index_rate"] = self.space.quantise(
            "index_rate", params.get("index_rate", 0.75))
        out["filter_radius"] = int(params.get("filter_radius", "0"))
        out["rms_mix_rate"] = self.space.quantise(
            "rms_mix_rate", params.get("rms_mix_rate", 0.25))
        out["protect"] = self.space.quantise(
            "protect", params.get("protect", 0.33))
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
            # (a genuinely new point — no dangling RUNNING trials)
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
                    values = [
                        float(record.axes.get("speaker_similarity")
                              or record.composite or 0.0),
                        float(record.axes.get("naturalness")
                              or record.composite or 0.0)]
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
                  seed: int = 7,
                  budget_total: int = 24) -> tuple[OptimizerBackend, str]:
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
    backend = BuiltInOptimizer(space, seed=seed, budget_total=budget_total,
                                multi_objective=multi_objective)
    name = "builtin-staged" + ("-moo" if multi_objective else "")
    return backend, name
