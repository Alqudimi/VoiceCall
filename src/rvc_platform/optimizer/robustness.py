"""Generalisation & anti-overfitting protocol (Phase 02 W12).

Phase 01 validated on interleaved segments of the SAME utterance the
search optimised on — adjacent-segment leakage that overestimates
generalisation. Phase 02 introduces an honest data-role protocol:

1. **File-level assignment** (preferred): with MULTIPLE input files,
   whole files are assigned to SEARCH / VALIDATION / TEST roles —
   different takes / sessions / phrases never share a role. With >= 3
   files the split is 1+ search, 1 validation, 1 test (extra files feed
   search).
2. **Two files**: one file searches, the other is split into validation +
   test segments (segment-level, flagged).
3. **One file**: last-resort segment-interleaved split with an explicit
   reliability note — the user is TOLD the validation shares the take.

Robustness evaluation: final candidates are re-converted on PROBES —
distinct segment TYPES (short / long / quiet / loud / voiced-heavy /
unvoiced-heavy) — so the reported winner is the one that performs
*stably* across input variety, not the one that lucked out on a single
segment (stability = mean - 0.5 x std across probes, reported per type).
"""
from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["assign_input_roles", "robustness_probes", "robustness_scores",
           "SR_ANALYSIS"]

SR_ANALYSIS = 16_000
_MIN_SEG_S = 1.0            # probes shorter than this are not scored alone
_MAX_PROBE_S = 8.0
_MAX_PROBES_PER_TYPE = 2


def assign_input_roles(input_paths: list[str], seed: int = 7
                       ) -> dict[str, Any]:
    """Assign input FILES to search / validation / test roles.

    Returns {strategy, search: [...], validation: [...], test: [...],
    notes: [...]} — strategy is one of:

    * ``file-level``     — different files per role (the honest split);
    * ``file+segment``   — validation/test share one other file;
    * ``segment-level``  — single take split (reliability-limited).
    """
    paths = [str(p) for p in (input_paths or []) if str(p).strip()]
    notes: list[str] = []
    if not paths:
        return {"strategy": "none", "search": [], "validation": [],
                "test": [], "notes": ["no input audio provided"]}
    if len(paths) == 1:
        notes.append(
            "single input file: search/validation/test split happens at "
            "SEGMENT level within the same take — results generalise less "
            "than a multi-file split (add another take to strengthen them)")
        return {"strategy": "segment-level", "search": paths,
                "validation": [],   # filled by the service's segment split
                "test": [], "notes": notes}
    if len(paths) == 2:
        rng = random.Random(seed)
        order = list(paths)
        rng.shuffle(order)
        notes.append(
            "two input files: the search runs on one file; validation and "
            "test share the second (segment-level within that file)")
        return {"strategy": "file+segment", "search": [order[0]],
                "validation": [order[1]], "test": [order[1]],
                "notes": notes}
    # >= 3 files: file-level split — one search file (prefer the largest
    # count for search), one validation, one test
    rng = random.Random(seed)
    order = list(paths)
    rng.shuffle(order)
    validation = [order[1]]
    test = [order[2]]
    search = [order[0]] + order[3:]
    notes.append(
        f"file-level split: {len(search)} search file(s), 1 validation, "
        "1 test — roles never share a take")
    return {"strategy": "file-level", "search": search,
            "validation": validation, "test": test, "notes": notes}


def robustness_probes(analyzer: Any, wav: np.ndarray,
                      ) -> list[dict[str, Any]]:
    """Build distinct probe segments from an utterance.

    Probe types (evidence-driven, from the analyzer's own VAD + envelope):

    * ``short``    — speech segment < 2 s;
    * ``long``     — speech segment >= 3 s;
    * ``quiet``    — lowest-tertile RMS speech segment;
    * ``loud``     — highest-tertile RMS speech segment;
    * ``voiced_heavy`` / ``unvoiced_heavy`` — extreme voiced-ratio segments;
    * ``normal``   — a representative mid segment (always included).

    Each probe is a REAL slice of the utterance (no synthesis, no resample
    trickery). Max 2 probes per type keeps evaluation cost bounded.
    """
    wav = np.asarray(wav, dtype=np.float32).ravel()
    if wav.size < int(2.0 * SR_ANALYSIS):
        return [{"type": "normal", "audio": wav, "note": "single short probe"}]
    try:
        segs = analyzer.segment_speech(wav)
    except Exception:
        segs = []
    if not segs:
        mid = wav.size // 2
        return [{"type": "normal", "audio": wav[mid // 2: mid + wav.size // 4],
                 "note": "VAD found no speech — centre probe used"}]
    rms_db = analyzer._frame_rms_db(wav)
    hop_s = 256 / SR_ANALYSIS
    seg_info = []
    for a, b in segs:
        i0, i1 = int(a / hop_s), int(b / hop_s)
        i1 = min(i1, rms_db.size)
        if i1 <= i0:
            continue
        seg_info.append({
            "a": a, "b": b, "dur": b - a,
            "rms": float(np.mean(rms_db[i0:i1])),
            "voiced": float(np.mean(rms_db[i0:i1] > -40.0)),
        })
    if not seg_info:
        return [{"type": "normal", "audio": wav,
                 "note": "degenerate VAD segments"}]

    probes: list[dict[str, Any]] = []
    taken: set[int] = set()

    def _add(ptype: str, idx: int, note: str = "") -> None:
        if len([p for p in probes if p["type"] == ptype]) >= \
                _MAX_PROBES_PER_TYPE:
            return
        if idx in taken:
            return
        info = seg_info[idx]
        piece = wav[int(info["a"] * SR_ANALYSIS):
                    int(info["b"] * SR_ANALYSIS)]
        probes.append({"type": ptype, "audio": piece,
                       "duration_s": round(info["dur"], 2), "note": note})
        taken.add(idx)

    # duration types
    for i, s in enumerate(seg_info):
        if s["dur"] < 2.0:
            _add("short", i, f"{s['dur']:.1f}s segment")
            break
    for i, s in enumerate(seg_info):
        if s["dur"] >= 3.0:
            _add("long", i, f"{s['dur']:.1f}s segment")
            break
    # loudness types
    if len(seg_info) >= 2:
        by_rms = sorted(range(len(seg_info)),
                        key=lambda i: seg_info[i]["rms"])
        _add("quiet", by_rms[0],
             f"lowest-RMS segment ({seg_info[by_rms[0]]['rms']:.0f} dB)")
        _add("loud", by_rms[-1],
             f"highest-RMS segment ({seg_info[by_rms[-1]]['rms']:.0f} dB)")
        # voiced-ratio types
        by_voiced = sorted(range(len(seg_info)),
                           key=lambda i: seg_info[i]["voiced"])
        _add("unvoiced_heavy", by_voiced[0],
             f"low voiced share ({seg_info[by_voiced[0]]['voiced']:.0%})")
        _add("voiced_heavy", by_voiced[-1],
             f"high voiced share ({seg_info[by_voiced[-1]]['voiced']:.0%})")
    # representative normal probe (median-duration, median-RMS)
    if seg_info:
        med = sorted(range(len(seg_info)),
                     key=lambda i: (abs(seg_info[i]["dur"] - 2.5),
                                    seg_info[i]["rms"]))[0]
        _add("normal", med, "representative segment")
    # drop degenerate probes (< _MIN_SEG_S handled by the caller's scorer
    # guard, but keep at least one)
    if not probes and seg_info:
        i = int(np.argmax([s["dur"] for s in seg_info]))
        _add("normal", i, "fallback longest segment")
    return probes


def robustness_scores(probe_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-probe-type scores + overall stability.

    ``probe_results`` — [{type, composite, gates_passed?, error?}].
    Stability = mean - 0.5 x std across probes (a candidate that swings
    between 0.8 and 0.4 ranks below one steady at 0.65 — anti-overfitting
    by design). Types with a single probe report std=0 with a note.
    """
    ok = [p for p in (probe_results or [])
          if p.get("error") is None and p.get("composite") is not None]
    if not ok:
        return {"n_probes": 0, "stability": None,
                "note": "no successful probe conversions — robustness "
                        "unmeasurable"}
    by_type: dict[str, list[float]] = {}
    for p in ok:
        by_type.setdefault(str(p.get("type", "normal")), []).append(
            float(p["composite"]))
    per_type: dict[str, dict[str, Any]] = {}
    for t, vals in sorted(by_type.items()):
        arr = np.asarray(vals, dtype=np.float64)
        per_type[t] = {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4),
        }
        if arr.size < 2:
            per_type[t]["note"] = "single probe — no spread measurable"
    all_vals = np.asarray([float(p["composite"]) for p in ok])
    stability = float(all_vals.mean() - 0.5 * all_vals.std())
    worst_type = min(per_type, key=lambda t: per_type[t]["mean"])
    errors = sum(1 for p in (probe_results or []) if p.get("error"))
    return {
        "n_probes": len(ok),
        "n_errors": errors,
        "per_type": per_type,
        "stability": round(stability, 4),
        "mean": round(float(all_vals.mean()), 4),
        "std": round(float(all_vals.std()), 4),
        "worst_type": worst_type,
        "note": "stability = mean - 0.5 x std across probe types "
                "(anti-overfitting measure)",
    }
