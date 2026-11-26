"""Human-in-the-loop intelligence: blind A/B preferences (Phase 02 W16).

Computational metrics are not a full substitute for listening. This module
implements the blind-comparison protocol:

1. the system generates pairs of candidate renders (typically best-by-
   strategy winners and/or the default baseline);
2. labels are RANDOMISED (the UI shows "A"/"B" only — parameters hidden
   until the choice is locked in);
3. the user picks the one closest to the reference voice;
4. the choice is stored and REVEALED (parameters + scores of both sides),
   permanently linked to the session, model and profile.

Preference learning (strictly separated from objective scoring):

* preference records feed a *transparent* affinity score: how often
  parameter neighborhoods like this candidate's were the preferred side;
* the combined recommendation may add a small bounded preference boost
  (default <= 5%) to RANKING only — objective axis scores are NEVER
  modified, and the UI shows objective, preference and combined as three
  separate views;
* too few preferences -> no boost at all (the system says so explicitly
  instead of inventing a signal).
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["PreferenceStore", "PREFERENCE_BOOST_MAX"]

PREFERENCE_BOOST_MAX = 0.05        # ranking-only, never applied to scores
_MIN_PREFERENCES_FOR_BOOST = 3
_NEIGHBORHOOD_TOL = {"index_rate": 0.10, "rms_mix_rate": 0.10,
                     "protect": 0.08, "pitch_shift": 1,
                     "filter_radius": 0, "f0_method": 0}


class PreferenceStore:
    """Filesystem preference store: blind pairs + choices, per session."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            from rvc_platform.infrastructure.paths import get_paths

            root = get_paths().data / "optimizer" / "preferences"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- paths
    def _session_pairs_path(self, session_id: str) -> Path:
        safe = "".join(c for c in session_id
                       if c.isalnum() or c in "-_") or "s"
        d = self.root / safe
        d.mkdir(parents=True, exist_ok=True)
        return d / "pairs.json"

    def _choices_path(self) -> Path:
        return self.root / "preferences.jsonl"

    # ----------------------------------------------------------------- pairs
    def create_pair(
        self,
        session_id: str,
        candidate_a: dict[str, Any],
        candidate_b: dict[str, Any],
        rng: random.Random | None = None,
    ) -> dict[str, Any]:
        """Create one blind pair; labels are shuffled (a/b).

        Each candidate dict: {number, params, audio, axes?, label?} — the
        ``label`` is an internal descriptor shown only AFTER the reveal.
        """
        rng = rng or random.Random()
        swap = rng.random() < 0.5
        left, right = ((candidate_b, candidate_a) if swap
                       else (candidate_a, candidate_b))
        pair = {
            "pair_id": f"pair_{time.strftime('%Y%m%d%H%M%S')}_"
                       f"{rng.randrange(16**6):06x}",
            "session_id": session_id,
            "created_at": time.time(),
            "candidates": {
                "a": left,
                "b": right,
            },
            "chosen": None,
            "chosen_at": None,
            "revealed": False,
        }
        path = self._session_pairs_path(session_id)
        with self._lock:
            pairs = self._read_pairs(session_id)
            pairs.append(pair)
            path.write_text(json.dumps(pairs, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        return pair

    def _read_pairs(self, session_id: str) -> list[dict[str, Any]]:
        path = self._session_pairs_path(session_id)
        if not path.is_file():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8")) or []
        except (json.JSONDecodeError, OSError):
            logger.exception("optimizer: preference pairs unreadable")
            return []

    def get_pair(self, session_id: str,
                 pair_id: str) -> dict[str, Any] | None:
        for p in self._read_pairs(session_id):
            if p.get("pair_id") == pair_id:
                return p
        return None

    def pairs(self, session_id: str) -> list[dict[str, Any]]:
        """All pairs for a session (blind view: chosen/revealed respected)."""
        return self._read_pairs(session_id)

    # --------------------------------------------------------------- choice
    def record_choice(self, session_id: str, pair_id: str,
                      choice: str, note: str = "") -> dict[str, Any]:
        """Lock in the user's blind choice and REVEAL both sides.

        Returns the revealed pair (with both candidates' parameters) —
        nothing about the parameters was available before this call.
        """
        if choice not in ("a", "b"):
            raise ValueError("choice must be 'a' or 'b'")
        path = self._session_pairs_path(session_id)
        with self._lock:
            pairs = self._read_pairs(session_id)
            pair = next((p for p in pairs
                         if p.get("pair_id") == pair_id), None)
            if pair is None:
                raise ValueError(f"pair {pair_id!r} not found in session "
                                 f"{session_id!r}")
            if pair.get("chosen") is not None:
                raise ValueError("pair already answered — one blind choice "
                                 "per pair (honesty of the protocol)")
            pair["chosen"] = choice
            pair["chosen_at"] = time.time()
            pair["revealed"] = True
            pair["note"] = (note or "")[:500]
            path.write_text(json.dumps(pairs, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        # persist the choice record (append-only, cross-session queries)
        chosen = pair["candidates"][choice]
        other_key = "b" if choice == "a" else "a"
        other = pair["candidates"][other_key]
        record = {
            "recorded_at": time.time(),
            "session_id": session_id,
            "pair_id": pair_id,
            "chosen": choice,
            "chosen_params": chosen.get("params") or {},
            "chosen_label": chosen.get("label"),
            "chosen_axes": chosen.get("axes") or {},
            "other_params": other.get("params") or {},
            "other_label": other.get("label"),
            "other_axes": other.get("axes") or {},
            "note": pair.get("note") or "",
        }
        try:
            with open(self._choices_path(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("optimizer: preference record append failed")
        return pair

    # -------------------------------------------------------------- queries
    def _choices(self) -> list[dict[str, Any]]:
        path = self._choices_path()
        if not path.is_file():
            return []
        out = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            logger.exception("optimizer: preference records unreadable")
        return out

    def preferences_for_model(self, model_name: str) -> list[dict[str, Any]]:
        """Choice records from sessions that ran this model (model link is
        carried in the record via the pair candidates' model field)."""
        out = []
        for r in self._choices():
            if str(r.get("model") or model_name) == model_name:
                out.append(r)
        return out

    def preference_summary(self, model_name: str) -> dict[str, Any]:
        """What the user tends to prefer (parameter views), per model."""
        records = self.preferences_for_model(model_name)
        n = len(records)
        if not n:
            return {"n_preferences": 0,
                    "note": "no blind A/B choices recorded for this model "
                            "yet — preference learning inactive"}
        wins: dict[str, dict[str, float]] = {}
        for r in records:
            for k, v in (r.get("chosen_params") or {}).items():
                slot = wins.setdefault(k, {"sum": 0.0, "n": 0})
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    slot["sum"] += float(v)
                    slot["n"] += 1
                else:
                    slot[f"level::{v}"] = slot.get(f"level::{v}", 0) + 1
        view: dict[str, Any] = {}
        for k, slot in wins.items():
            if slot.get("n"):
                view[k] = {"preferred_mean": round(slot["sum"] / slot["n"], 3),
                           "n": slot["n"]}
            for lk, cnt in slot.items():
                if lk.startswith("level::"):
                    view.setdefault(k, {})[lk] = cnt
        return {"n_preferences": n, "parameter_views": view}

    # ------------------------------------------------------- preference boost
    def preference_affinity(self, model_name: str,
                            params: dict[str, Any]) -> float:
        """Affinity of a parameter set with past preferred sets (0..1).

        Affinity = mean over records of the closeness between this
        candidate's params and the preferred side's params (1 when
        identical, decaying by parameter-specific tolerance). Returns 0
        when there is not enough evidence — never a fabricated signal.
        """
        records = self.preferences_for_model(model_name)
        if len(records) < _MIN_PREFERENCES_FOR_BOOST:
            return 0.0
        closeness = []
        for r in records:
            closeness.append(self._params_closeness(
                params, r.get("chosen_params") or {}))
        return float(sum(closeness) / len(closeness))

    @staticmethod
    def _params_closeness(a: dict[str, Any], b: dict[str, Any]) -> float:
        if not a or not b:
            return 0.0
        total = 0.0
        n = 0
        for key, tol in _NEIGHBORHOOD_TOL.items():
            if key not in a or key not in b:
                continue
            n += 1
            va, vb = a[key], b[key]
            if isinstance(va, (int, float)) and isinstance(
                    vb, (int, float)):
                if tol <= 0:
                    total += 1.0 if float(va) == float(vb) else 0.0
                else:
                    total += float(max(0.0, 1.0 - abs(
                        float(va) - float(vb)) / tol))
            else:
                total += 1.0 if str(va) == str(vb) else 0.0
        return total / n if n else 0.0

    def preference_boost(self, model_name: str,
                         params: dict[str, Any]) -> float:
        """Bounded ranking-only boost (<= PREFERENCE_BOOST_MAX).

        Zero with insufficient preferences; NEVER applied to objective
        scores — see best_with_preferences().
        """
        affinity = self.preference_affinity(model_name, params)
        return float(PREFERENCE_BOOST_MAX * affinity)
