"""Optimizer search backends — space, duplicates, budgets, restore."""
from __future__ import annotations

import pytest

from rvc_platform.optimizer.search import (
    BuiltInOptimizer,
    OptimizationSpace,
    OptunaOptimizer,
    TrialRecord,
    available_f0_methods,
    build_backend,
)


def test_available_f0_methods_are_constructable():
    methods = available_f0_methods()
    assert "pm" in methods  # parselmouth is a hard dependency
    for m in methods:
        assert isinstance(m, str) and m


def test_space_quantisation_and_keys():
    sp = OptimizationSpace(f0_methods=["pm", "fcpe"])
    assert sp.quantise("index_rate", 0.449999) == 0.45
    assert sp.quantise("index_rate", 5.0) == 1.0
    assert sp.quantise("protect", -0.5) == 0.0
    a = sp.key({"f0_method": "pm", "index_rate": 0.4499999, "filter_radius": 3,
                "rms_mix_rate": 0.25, "protect": 0.33, "pitch_shift": 0})
    b = sp.key({"f0_method": "pm", "index_rate": 0.45, "filter_radius": 3,
                "rms_mix_rate": 0.25, "protect": 0.33, "pitch_shift": 0})
    assert a == b, "quantised duplicates must share one cache key"
    assert sp.describe()["filter_radius_values"] == [0, 3]


def test_builtin_never_suggests_duplicates():
    sp = OptimizationSpace(f0_methods=["pm"], pitch_policy="preserve")
    be = BuiltInOptimizer(sp, seed=3)
    seen = set()
    for i in range(40):
        params = be.suggest(i)
        key = sp.key(params)
        assert key not in seen, f"duplicate at iteration {i}"
        seen.add(key)
        be.tell(TrialRecord(number=i + 1, params=params, composite=0.5))


def test_builtin_best_prefers_gate_passing():
    sp = OptimizationSpace(f0_methods=["pm"])
    be = BuiltInOptimizer(sp, seed=1)
    be.tell(TrialRecord(number=1, params=be.suggest(0), composite=0.9,
                        gates_passed=False))
    be.tell(TrialRecord(number=2, params=be.suggest(1), composite=0.7,
                        gates_passed=True))
    assert be.best().number == 2


def test_builtin_exploration_then_guidance_uses_good_set():
    sp = OptimizationSpace(f0_methods=["pm", "fcpe"], pitch_policy="range")
    be = BuiltInOptimizer(sp, seed=11)
    # feed a strong preference: high index_rate wins
    for i in range(20):
        params = be.suggest(i)
        score = 0.2 + 0.7 * float(params["index_rate"])
        be.tell(TrialRecord(number=i + 1, params=params, composite=score))
    trials = be.trials
    assert len(trials) == 20
    # with guidance, later trials should sample more high index_rate values
    late = [t for t in trials[12:]]
    early = [t for t in trials[:8]]
    late_mean = sum(t.params["index_rate"] for t in late) / len(late)
    early_mean = sum(t.params["index_rate"] for t in early) / len(early)
    assert late_mean >= early_mean


def test_builtin_restore_replays_history():
    sp = OptimizationSpace(f0_methods=["pm"])
    be = BuiltInOptimizer(sp, seed=2)
    records = []
    for i in range(5):
        p = be.suggest(i)
        rec = TrialRecord(number=i + 1, params=p, composite=0.4 + 0.05 * i)
        be.tell(rec)
        records.append(rec)
    be2 = BuiltInOptimizer(sp, seed=2)
    be2.restore(records)
    assert len(be2.trials) == 5
    assert be2.best().composite == pytest.approx(0.6)
    # restored seen-keys prevent duplicate suggestions
    nxt = be2.suggest(5)
    assert sp.key(nxt) not in {sp.key(r.params) for r in records}


@pytest.mark.parametrize("multi", [False, True])
def test_optuna_backend_roundtrip(multi):
    optuna = pytest.importorskip("optuna")
    sp = OptimizationSpace(f0_methods=["pm", "fcpe"],
                           pitch_policy="range" if multi else "preserve")
    be, name = build_backend(sp, strategy="optuna", multi_objective=multi)
    assert isinstance(be, OptunaOptimizer)
    assert name.startswith("optuna-")
    seen = set()
    for i in range(20):
        p = be.suggest(i)
        key = sp.key(p)
        assert key not in seen, f"optuna duplicate at {i}"
        seen.add(key)
        rec = TrialRecord(number=i + 1, params=p, composite=0.3 + 0.02 * i,
                          axes={"clarity": 0.6}, gates_passed=True)
        be.tell(rec)
    assert be.best().composite == pytest.approx(0.3 + 0.02 * 19)
    # study completed trials count matches told trials (no dangling RUNNING)
    completed = [t for t in be.study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) == 20


def test_optuna_restore_replays_trials():
    pytest.importorskip("optuna")
    sp = OptimizationSpace(f0_methods=["pm"])
    be = OptunaOptimizer(sp)
    records = []
    for i in range(4):
        p = be.suggest(i)
        rec = TrialRecord(number=i + 1, params=p, composite=0.4 + 0.1 * i)
        be.tell(rec)
        records.append(rec)
    be2 = OptunaOptimizer(sp)
    be2.restore(records)
    assert len(be2.trials) == 4
    assert len(be2.study.trials) >= 4


def test_build_backend_falls_back_to_builtin():
    sp = OptimizationSpace(f0_methods=["pm"])
    be, name = build_backend(sp, strategy="auto")
    assert name in ("optuna-tpe", "builtin-staged")
    be2, name2 = build_backend(sp, strategy="builtin")
    assert name2 == "builtin-staged"
    assert isinstance(be2, BuiltInOptimizer)


def test_trial_record_serialisation_roundtrip():
    rec = TrialRecord(number=7, params={"index_rate": 0.5}, composite=0.62,
                     axes={"clarity": 0.5}, gates_passed=False,
                     failed_gates=["clarity"], reliability="low",
                     duration_s=1.5, vram_mb=128, cpu_percent=44.0,
                     error=None)
    data = rec.to_dict()
    assert data["number"] == 7 and data["composite"] == 0.62
    rec2 = TrialRecord.from_dict(data)
    assert rec2.params == rec.params and rec2.failed_gates == ["clarity"]
    # failed trial keeps its error
    fail = TrialRecord(number=8, params={}, error="boom").to_dict()
    assert fail["error"] == "boom"
