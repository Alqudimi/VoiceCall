"""Parameter relationship discovery (Phase 02 W7)."""
from __future__ import annotations

import numpy as np

from rvc_platform.optimizer.interactions import (
    MIN_TRIALS_FOR_ANALYSIS,
    analyze_parameter_effects,
)


def _trials(n: int, effect: dict | None = None, seed: int = 5,
            noise: float = 0.02) -> list[dict]:
    """Synthetic trial history where index_rate drives the composite."""
    rng = np.random.default_rng(seed)
    effect = effect or {}
    out = []
    for i in range(n):
        ir = round(float(rng.uniform(0, 1)), 2)
        p = {"index_rate": ir,
             "rms_mix_rate": round(float(rng.uniform(0, 1)), 2),
             "protect": round(float(rng.uniform(0, 0.5)), 2),
             "pitch_shift": 0,
             "f0_method": str(rng.choice(["pm", "fcpe"])),
             "filter_radius": int(rng.choice([0, 3]))}
        comp = 0.4 + 0.4 * ir + noise * rng.standard_normal()
        for k, w in effect.items():
            comp += w * float(p.get(k, 0.0))
        out.append({"params": p, "composite": float(np.clip(comp, 0, 1)),
                    "axes": {"speaker_similarity": comp},
                    "gates_passed": True, "error": None})
    return out


def test_insufficient_data_degrades_honestly():
    rep = analyze_parameter_effects(_trials(5))
    assert rep["status"] == "insufficient-data"
    assert "only 5 usable trials" in rep["note"]
    assert "parameters" not in rep


def test_marginal_effect_detected_for_driving_parameter():
    rep = analyze_parameter_effects(_trials(24))
    assert rep["status"] == "ok"
    ir = rep["parameters"]["index_rate"]
    assert ir["type"] == "numeric"
    assert ir["n_distinct_values"] > 5
    assert len(ir["bins"]) >= 2
    assert ir["best_bin"]["mean_composite"] > 0.5
    # sensitivity anchored to the total composite spread
    assert ir["sensitivity"] > 0.5


def test_constant_parameter_reported_as_unmeasurable():
    trials = _trials(12)
    for t in trials:
        t["params"]["protect"] = 0.33
    rep = analyze_parameter_effects(trials)
    pr = rep["parameters"]["protect"]
    assert pr["n_distinct_values"] == 1
    assert "no effect measurable" in pr["note"]


def test_importance_ranking_orders_parameters():
    rep = analyze_parameter_effects(_trials(24))
    ranking = rep["importance_ranking"]
    assert ranking
    etas = [r["importance_eta2"] for r in ranking]
    assert etas == sorted(etas, reverse=True)
    # index_rate drives the synthetic signal: it must rank first
    assert ranking[0]["parameter"] == "index_rate"


def test_categorical_levels_reported():
    rep = analyze_parameter_effects(_trials(24))
    fm = rep["parameters"]["f0_method"]
    assert fm["type"] == "categorical"
    assert set(fm["levels"]) == {"pm", "fcpe"}


def test_numeric_pair_interaction_detected_when_present():
    # interaction: protect hurts only when index_rate is high
    trials = []
    rng = np.random.default_rng(7)
    for i in range(30):
        ir = round(float(rng.uniform(0, 1)), 2)
        pr = round(float(rng.uniform(0, 0.5)), 2)
        comp = 0.5 + 0.3 * ir - 0.5 * pr * ir + 0.01 * rng.standard_normal()
        trials.append({
            "params": {"index_rate": ir, "protect": pr,
                       "rms_mix_rate": 0.25, "pitch_shift": 0,
                       "f0_method": "pm", "filter_radius": 0},
            "composite": float(np.clip(comp, 0, 1)),
            "axes": {}, "gates_passed": True, "error": None})
    rep = analyze_parameter_effects(trials)
    pairs = [i for i in rep["interactions"] if "a" in i]
    assert any({i["a"], i["b"]} == {"index_rate", "protect"}
               for i in pairs)


def test_no_interaction_invented_from_noise():
    rep = analyze_parameter_effects(_trials(24, noise=0.01))
    # pure main-effect data: no numeric-pair interaction should be claimed
    # (median-split cells deviate additively)
    pairs = [i for i in rep["interactions"] if "a" in i and "b" in i]
    for p in pairs:
        eff = p["strongest_cell"]["interaction_effect"]
        assert abs(eff) < 0.12        # no strong interaction claimed


def test_recommendations_reference_real_numbers():
    rep = analyze_parameter_effects(_trials(24))
    assert rep["recommendations"]
    assert any("index_rate" in r for r in rep["recommendations"])
    # the recommendation quotes its evidence (range + mean + n)
    r = next(r for r in rep["recommendations"] if r.startswith("index_rate"))
    assert "mean" in r and "n=" in r


def test_unstable_region_detected_for_high_variance_bin():
    trials = _trials(20, seed=11)
    # inject wild swings in one region of index_rate
    rng = np.random.default_rng(12)
    for i, t in enumerate(trials):
        if t["params"]["index_rate"] < 0.3:
            t["composite"] = float(
                np.clip(0.05 + rng.uniform(0.0, 0.9), 0, 1))
    rep = analyze_parameter_effects(trials)
    ir = rep["parameters"]["index_rate"]
    assert "unstable_regions" in ir
    assert ir["unstable_regions"]
