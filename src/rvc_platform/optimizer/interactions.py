"""Parameter relationship discovery (Phase 02 W7 — the user's top request).

The optimizer must not treat parameters as independent knobs. This module
extracts, from the session's REAL trial history (no simulated data, no
assumed relationships), a structured view of:

1. **marginal effects** — for every numeric parameter: quantile bins with
   mean/std composite, gate-pass rate, best bin, linear trend;
2. **sensitivity** — how much the composite actually moves when the
   parameter moves (binned-mean range, anchored to the overall composite
   spread so "sensitivity 0.4" means "moves 40% of what the whole search
   moves");
3. **importance** — one-factor ANOVA-style eta-squared: the share of
   composite variance each parameter's bins explain (computed only with
   enough trials; reported with n);
4. **interactions** — pairwise: 2-D cell analysis (cell mean deviation from
   the additive main-effects model) for numeric pairs, and per-level
   sensitivity differences for categorical x numeric pairs. Only cells with
   enough observations are reported; sparse data degrades to an honest
   "insufficient evidence" note — never an invented interaction;
5. **unstable regions** — parameter ranges where the composite swings
   wildly between neighbouring trials (high bin variance) — the regions
   the search should avoid or refine carefully;
6. **recommendations** — derived statements ("index_rate explains the most
   variance; best observed range 0.65-0.85 with mean composite 0.71"),
   each backed by the numbers above.

Everything here is derived from ``TrialRecord``-shaped dicts (params +
composite + axes + gates) — the same data the session persisted, so the
analysis is reproducible from a saved session file.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["analyze_parameter_effects", "MIN_TRIALS_FOR_ANALYSIS"]

MIN_TRIALS_FOR_ANALYSIS = 8          # below this the report degrades honestly
_MIN_BIN_N = 3                       # trials per bin for a trustworthy bin mean
_MIN_CELL_N = 3                      # trials per 2-D cell for interaction claims
_NUMERIC_PARAMS = ("index_rate", "rms_mix_rate", "protect", "pitch_shift")
_CATEGORICAL_PARAMS = ("f0_method", "filter_radius")


def _safe_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in trials or []:
        if t.get("error") is not None:
            continue
        if t.get("composite") is None:
            continue
        if not isinstance(t.get("params"), dict):
            continue
        out.append(t)
    return out


def _composite(t: dict[str, Any]) -> float:
    return float(t.get("composite") or 0.0)


def _bin_numeric(values: np.ndarray, scores: np.ndarray, passed: np.ndarray,
                 n_bins: int = 4) -> list[dict[str, Any]]:
    """Quantile bins of a numeric parameter with composite statistics."""
    order = np.argsort(values)
    v, s, p = values[order], scores[order], passed[order]
    edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    # de-duplicate edges (discrete grids like protect 0/0.5 collapse bins)
    uniq = np.unique(edges)
    if uniq.size < 2:
        n_bins = 1
        edges = np.array([v.min(), v.max()])
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (v >= lo) & (v <= hi if i == n_bins - 1 else v < hi)
        if not mask.any():
            continue
        bs = s[mask]
        bins.append({
            "lo": round(float(lo), 3),
            "hi": round(float(hi), 3),
            "n": int(mask.sum()),
            "mean_composite": round(float(bs.mean()), 4),
            "std_composite": round(float(bs.std()), 4),
            "max_composite": round(float(bs.max()), 4),
            "gate_pass_rate": round(float(p[mask].mean()), 3),
        })
    return bins


def _eta_squared(groups: list[np.ndarray]) -> float | None:
    """One-way ANOVA eta-squared: share of variance explained by grouping.

    Returns None when any group has < 2 observations or total variance is
    zero — insufficient evidence, never a fabricated number.
    """
    valid = [g for g in groups if g.size >= 2]
    if len(valid) < 2:
        return None
    allv = np.concatenate(valid)
    grand = float(allv.mean())
    ss_between = sum(g.size * (float(g.mean()) - grand) ** 2 for g in valid)
    ss_total = float(np.sum((allv - grand) ** 2))
    if ss_total <= 1e-12:
        return None
    return float(ss_between / ss_total)


def _numeric_param_report(trials: list[dict[str, Any]], name: str,
                          all_scores: np.ndarray) -> dict[str, Any]:
    values = np.array([float(t["params"].get(name, 0.0)) for t in trials])
    scores = np.array([_composite(t) for t in trials])
    passed = np.array([bool(t.get("gates_passed", True)) for t in trials])
    n_values = len(np.unique(values))
    report: dict[str, Any] = {
        "type": "numeric",
        "range_tested": [round(float(values.min()), 3),
                         round(float(values.max()), 3)],
        "n_distinct_values": int(n_values),
    }
    # linear trend (Spearman — monotone, robust)
    if n_values >= 3 and scores.size >= 4:
        report["trend_spearman"] = _spearman(values.tolist(),
                                             scores.tolist())
    bins = _bin_numeric(values, scores, passed)
    report["bins"] = bins
    if len(bins) >= 2:
        means = [b["mean_composite"] for b in bins]
        report["best_bin"] = bins[int(np.argmax(means))]
        # sensitivity anchored to the overall composite spread
        spread = float(max(1e-9, all_scores.max() - all_scores.min()))
        delta = float(max(means) - min(means))
        report["sensitivity"] = round(delta / spread, 4)
        report["abs_composite_delta"] = round(delta, 4)
        # importance (eta-squared) — enough trials per bin only
        groups = [scores[(values >= b["lo"]) & (values <= b["hi"])]
                  for b in bins]
        eta = _eta_squared(groups)
        if eta is not None:
            report["importance_eta2"] = round(eta, 4)
        # unstable region detection: a bin whose composite swings far more
        # than the *other* bins (robust comparison — the unstable bin itself
        # inflates any overall-spread baseline), with a small absolute floor
        stable_stds = [b["std_composite"] for b in bins]
        bin_ns = [b["n"] for b in bins]
        if len(bins) >= 2 and max(bin_ns) >= _MIN_BIN_N:
            others_median = float(np.median(
                [s for i, s in enumerate(stable_stds)
                 if bin_ns[i] >= _MIN_BIN_N] or [0.0]))
            unstable = [b for b in bins
                        if b["n"] >= _MIN_BIN_N
                        and b["std_composite"] > max(0.10,
                                                     2.0 * others_median)]
            if unstable:
                report["unstable_regions"] = [
                    {"range": [b["lo"], b["hi"]],
                     "std_composite": b["std_composite"],
                     "n": b["n"]} for b in unstable]
    return report


def _categorical_param_report(trials: list[dict[str, Any]], name: str,
                              all_scores: np.ndarray) -> dict[str, Any]:
    scores = np.array([_composite(t) for t in trials])
    levels: dict[str, dict[str, Any]] = {}
    for t in trials:
        lvl = str(t["params"].get(name))
        slot = levels.setdefault(lvl, {"n": 0, "scores": [], "passes": []})
        slot["n"] += 1
        slot["scores"].append(_composite(t))
        slot["passes"].append(bool(t.get("gates_passed", True)))
    report: dict[str, Any] = {
        "type": "categorical",
        "levels": {},
    }
    level_means = {}
    for lvl, slot in levels.items():
        arr = np.asarray(slot["scores"])
        report["levels"][lvl] = {
            "n": slot["n"],
            "mean_composite": round(float(arr.mean()), 4),
            "std_composite": round(float(arr.std()), 4),
            "max_composite": round(float(arr.max()), 4),
            "gate_pass_rate": round(
                float(np.mean(slot["passes"])), 3),
        }
        level_means[lvl] = float(arr.mean())
    if len(level_means) >= 2:
        best = max(level_means, key=level_means.get)
        report["best_level"] = best
        spread = float(max(1e-9, all_scores.max() - all_scores.min()))
        delta = float(max(level_means.values()) - min(level_means.values()))
        report["sensitivity"] = round(delta / spread, 4)
        report["abs_composite_delta"] = round(delta, 4)
        groups = [np.asarray(slot["scores"]) for slot in levels.values()]
        eta = _eta_squared(groups)
        if eta is not None:
            report["importance_eta2"] = round(eta, 4)
    return report


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 4:
        return 0.0
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ra = _rank(a)
    rb = _rank(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return round(float(np.corrcoef(ra, rb)[0, 1]), 3)


def _rank(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, arr.size + 1)
    return ranks


def _numeric_pairs_interactions(trials: list[dict[str, Any]],
                                scores: np.ndarray) -> list[dict[str, Any]]:
    """2-D cell analysis: deviation from the additive main-effects model.

    For each numeric pair and each 2-D cell with >= _MIN_CELL_N trials,
    effect = cell_mean - (mean_a + mean_b - grand). Only cells with enough
    observations AND a notable effect size are reported; everything else
    degrades to nothing (no invented interactions).
    """
    out: list[dict[str, Any]] = []
    n = len(trials)
    grand = float(scores.mean())
    for i, a in enumerate(_NUMERIC_PARAMS):
        for b in _NUMERIC_PARAMS[i + 1:]:
            va = np.array([float(t["params"].get(a, 0.0)) for t in trials])
            vb = np.array([float(t["params"].get(b, 0.0)) for t in trials])
            # median split per dimension -> 4 cells
            ma, mb = float(np.median(va)), float(np.median(vb))
            if va.std() < 1e-9 or vb.std() < 1e-9:
                continue
            cells: dict[tuple[int, int], list[float]] = {}
            for k in range(n):
                key = (0 if va[k] <= ma else 1, 0 if vb[k] <= mb else 1)
                cells.setdefault(key, []).append(scores[k])
            if any(len(v) < _MIN_CELL_N for v in cells.values()):
                continue
            # main effects from the cell margins (flatten the score lists)
            def _margin(dim: int, level: int) -> float:
                vals = [x for key, cell in cells.items()
                        if key[dim] == level for x in cell]
                return float(np.mean(vals)) if vals else 0.0

            mean_a0 = _margin(0, 0)
            mean_a1 = _margin(0, 1)
            mean_b0 = _margin(1, 0)
            mean_b1 = _margin(1, 1)
            best_cell = None
            for key, vals in cells.items():
                expected = (mean_a0 if key[0] == 0 else mean_a1) + \
                    (mean_b0 if key[1] == 0 else mean_b1) - grand
                effect = float(np.mean(vals)) - expected
                entry = {
                    "cell": [f"{a}{'>med' if key[0] else '<=med'}",
                             f"{b}{'>med' if key[1] else '<=med'}"],
                    "n": len(vals),
                    "cell_mean": round(float(np.mean(vals)), 4),
                    "additive_expectation": round(expected, 4),
                    "interaction_effect": round(effect, 4),
                }
                if abs(effect) >= 0.02 and (best_cell is None
                                           or abs(effect) >
                                           abs(best_cell["interaction_effect"])):
                    best_cell = entry
            if best_cell is not None:
                out.append({
                    "a": a, "b": b,
                    "strongest_cell": best_cell,
                    "cells": {f"{k[0]}{k[1]}": {"n": len(v),
                                                "mean": round(float(
                                                    np.mean(v)), 4)}
                              for k, v in cells.items()},
                    "note": "deviation from the additive main-effects model "
                            "over this session's trials (median split)",
                })
    return out


def _cat_num_interactions(trials: list[dict[str, Any]],
                          ) -> list[dict[str, Any]]:
    """Categorical x numeric: does the numeric parameter's effect differ
    between categorical levels (e.g. index_rate matters more with rmvpe)?"""
    out: list[dict[str, Any]] = []
    for cat in _CATEGORICAL_PARAMS:
        levels = {str(t["params"].get(cat)) for t in trials}
        if len(levels) < 2:
            continue
        for num in _NUMERIC_PARAMS:
            per_level: dict[str, dict[str, float]] = {}
            enough = True
            for lvl in levels:
                sub = [t for t in trials
                       if str(t["params"].get(cat)) == lvl]
                vals = np.array([float(t["params"].get(num, 0.0))
                                 for t in sub])
                sc = np.array([_composite(t) for t in sub])
                if vals.std() < 1e-9 or len(sub) < 4:
                    enough = False
                    break
                rho = _spearman(vals.tolist(), sc.tolist())
                per_level[lvl] = {
                    "n": len(sub),
                    "trend_spearman": rho,
                }
            if not enough or len(per_level) < 2:
                continue
            rhos = [v["trend_spearman"] for v in per_level.values()]
            if max(rhos) - min(rhos) >= 0.35:
                out.append({
                    "categorical": cat,
                    "numeric": num,
                    "per_level": per_level,
                    "trend_spread": round(float(max(rhos) - min(rhos)), 3),
                    "note": f"the effect of {num} differs between {cat} "
                            "levels (Spearman trend spread >= 0.35)",
                })
    return out


def analyze_parameter_effects(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Full parameter-intelligence report from the session's real trials."""
    ok = _safe_trials(trials)
    n = len(ok)
    base = {
        "n_trials": n,
        "generated_by": "rvc-platform optimizer interactions v2",
        "note": "all statistics derive from this session's real trials only",
    }
    if n < MIN_TRIALS_FOR_ANALYSIS:
        base["status"] = "insufficient-data"
        base["note"] += (f" — only {n} usable trials; "
                         f">= {MIN_TRIALS_FOR_ANALYSIS} recommended for "
                         "parameter-effect analysis")
        return base

    scores = np.array([_composite(t) for t in ok])
    params: dict[str, Any] = {}
    for name in _NUMERIC_PARAMS:
        values = np.array(
            [float(t["params"].get(name, 0.0)) for t in ok])
        if values.size and values.std() < 1e-9:
            params[name] = {
                "type": "numeric",
                "range_tested": [round(float(values[0]), 3)] * 2,
                "n_distinct_values": 1,
                "note": "parameter held constant this session — no "
                        "effect measurable",
            }
            continue
        params[name] = _numeric_param_report(ok, name, scores)
    for name in _CATEGORICAL_PARAMS:
        levels = {str(t["params"].get(name)) for t in ok}
        if len(levels) < 2:
            params[name] = {
                "type": "categorical",
                "levels": {next(iter(levels)): {
                    "n": n, "mean_composite": round(float(scores.mean()), 4)}},
                "note": "single level tested — no effect measurable",
            }
            continue
        params[name] = _categorical_param_report(ok, name, scores)

    interactions = _numeric_pairs_interactions(ok, scores) + \
        _cat_num_interactions(ok)

    # importance ranking (eta-squared where computable)
    ranking = sorted(
        ({"parameter": k,
          "importance_eta2": v.get("importance_eta2"),
          "sensitivity": v.get("sensitivity")}
         for k, v in params.items()
         if v.get("importance_eta2") is not None),
        key=lambda r: -(r["importance_eta2"] or 0.0))

    recommendations = _recommendations(params, interactions)
    return {
        **base,
        "status": "ok",
        "parameters": params,
        "importance_ranking": ranking,
        "interactions": interactions,
        "recommendations": recommendations,
    }


def _recommendations(params: dict[str, Any],
                     interactions: list[dict[str, Any]]) -> list[str]:
    """Derived, evidence-backed statements for the UI + export."""
    out: list[str] = []
    ranked = sorted(
        ((k, v) for k, v in params.items()
         if v.get("sensitivity") is not None),
        key=lambda kv: -(kv[1].get("sensitivity") or 0.0))
    for name, rep in ranked[:3]:
        sens = rep.get("sensitivity")
        if sens is None or sens < 0.05:
            continue
        if rep["type"] == "numeric" and rep.get("best_bin"):
            b = rep["best_bin"]
            out.append(
                f"{name}: highest composite observed in "
                f"{b['lo']}-{b['hi']} (mean {b['mean_composite']}, n={b['n']}); "
                f"sensitivity {sens:.0%} of the search's total composite spread")
        elif rep["type"] == "categorical" and rep.get("best_level"):
            out.append(
                f"{name}: best level '{rep['best_level']}' "
                f"(sensitivity {sens:.0%})")
    inert = [k for k, v in params.items()
             if v.get("sensitivity") is not None and v["sensitivity"] < 0.03
             and v.get("n_distinct_values", 2) > 1]
    if inert:
        out.append("negligible observed effect this session: "
                   + ", ".join(inert)
                   + " — spending more trials on it is unlikely to pay off")
    for inter in interactions[:3]:
        if "a" in inter and "b" in inter:
            out.append(
                f"interaction detected between {inter['a']} and "
                f"{inter['b']} — treat them as a pair, not independent knobs")
        elif "categorical" in inter and "numeric" in inter:
            out.append(
                f"interaction detected between {inter['categorical']} and "
                f"{inter['numeric']} — the best {inter['numeric']} depends "
                f"on the {inter['categorical']} choice")
    unstable = [(k, v.get("unstable_regions")) for k, v in params.items()]
    for name, regions in unstable:
        if regions:
            out.append(
                f"{name}: unstable region(s) detected "
                + ", ".join(f"{r['range'][0]}-{r['range'][1]}"
                            for r in regions[:2])
                + " — composite swings widely there; fine-search carefully")
    return out
