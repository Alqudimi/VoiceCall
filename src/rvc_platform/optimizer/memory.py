"""Experiment memory & knowledge system (Phase 02 W14).

Phase 01 trials lived inside one session and died with it. Phase 02 adds a
persistent, append-only experiment log under
``data/optimizer/memory/experiments.jsonl``. Every trial records:

* **hashes** — model (PTH), index (when present), input audio content,
  profile identity;
* **versions** — optimizer engine version, evaluation embedder set,
  software version, so stale results are DETECTED and EXCLUDED rather than
  silently mixed;
* **the trial itself** — parameters, F0 method, all axis scores, gates,
  artifact detail, processing time, device, session id, timestamp.

What the memory is used for (all implemented, all tested):

* **duplicate prevention** — a (model, input, params) triple already
  measured under the current versions is never re-run inside a session
  (the search space already dedups) NOR across sessions (this module);
* **warm start** — the best parameters observed in prior sessions for the
  same model (+ same input, then any input) seed the new search so it
  starts from knowledge, not from zero;
* **conversion cache** — a cache hit replays the stored analysis +
  similarity report through the CURRENT evaluation (weights, profile
  cohorts) without re-converting: cheap re-scoring, honest about the
  audio file being gone;
* **knowledge queries** — best-known settings per model, trials-count
  history, session comparisons (used by the UI's "best known parameters").

Invalidation rules (strict):

* a record is only reusable when its ``optimizer_version`` equals the
  current one (scoring semantics changed -> old scores are not comparable);
* a cache entry is only reusable when its model hash, index hash, input
  hash and embedder fingerprint ALL match the current run;
* nothing is ever deleted automatically — incompatible records stay on
  disk for audit, they are just excluded from reuse.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["ExperimentMemory", "OPTIMIZER_VERSION", "file_sha256",
           "audio_sha256", "params_key"]

#: bumped whenever scoring semantics change — invalidates prior records
OPTIMIZER_VERSION = "2.0.0"
_SCHEMA = 2


def file_sha256(path: Path | str) -> str | None:
    """Stable content hash of a file (None for missing/unreadable)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def audio_sha256(wav: Any) -> str | None:
    """Content hash of an in-memory waveform (raw-bytes of float32)."""
    try:
        import numpy as np

        arr = np.ascontiguousarray(wav, dtype="<f4")
        return hashlib.sha256(arr.tobytes()).hexdigest()
    except Exception:
        return None


def params_key(params: dict[str, Any]) -> str:
    """Canonical key of a parameter set (sorted, rounded)."""
    try:
        canon = {
            str(k): (round(float(v), 3)
                     if isinstance(v, (int, float)) and not isinstance(v, bool)
                     else str(v))
            for k, v in (params or {}).items()
        }
        return json.dumps(canon, sort_keys=True)
    except (TypeError, ValueError):
        return json.dumps({k: str(v) for k, v in (params or {}).items()},
                          sort_keys=True)


class ExperimentMemory:
    """Append-only experiment log + conversion cache + lookup APIs."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            from rvc_platform.infrastructure.paths import get_paths

            root = get_paths().data / "optimizer" / "memory"
        self.root = Path(root)
        self.log_path = self.root / "experiments.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index: list[dict[str, Any]] | None = None
        self._by_key: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------ loading
    def _load(self) -> list[dict[str, Any]]:
        if self._index is not None:
            return self._index
        records: list[dict[str, Any]] = []
        if self.log_path.is_file():
            try:
                for line in self.log_path.read_text(
                        encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue                # torn tail write — skip
            except OSError:
                logger.exception("optimizer: experiment log unreadable")
        self._index = records
        self._rebuild_keys(records)
        return records

    def _rebuild_keys(self, records: list[dict[str, Any]]) -> None:
        self._by_key = {}
        for r in records:
            key = r.get("cache_key")
            if key:
                self._by_key[key] = r

    # ------------------------------------------------------------- record
    def record_trial(
        self,
        *,
        model_hash: str | None,
        index_hash: str | None,
        profile_id: str,
        input_hash: str | None,
        params: dict[str, Any],
        session_id: str,
        axes: dict[str, Any] | None = None,
        composite: float | None = None,
        gates_passed: bool = True,
        failed_gates: list[str] | None = None,
        reliability: str = "unknown",
        artifact_detail: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
        processing_time_s: float | None = None,
        device: str = "cpu",
        embedder_fingerprint: dict[str, Any] | None = None,
        software_version: str = "",
        output_analysis: dict[str, Any] | None = None,
        similarity_report: dict[str, Any] | None = None,
        audio_available: bool = False,
    ) -> dict[str, Any]:
        """Append one trial to the experiment log (conversion-cache
        payload optional)."""
        if model_hash is None:
            # no model hash -> the record is kept for audit but can never
            # be reused as a cache entry
            cache_key = None
        else:
            cache_key = "|".join([
                OPTIMIZER_VERSION, str(model_hash),
                str(index_hash or "-"), str(input_hash or "-"),
                params_key(params),
            ])
        record = {
            "schema": _SCHEMA,
            "optimizer_version": OPTIMIZER_VERSION,
            "recorded_at": time.time(),
            "session_id": session_id,
            "model_hash": model_hash,
            "index_hash": index_hash,
            "profile_id": profile_id,
            "input_hash": input_hash,
            "params": dict(params or {}),
            "f0_method": str((params or {}).get("f0_method", "")),
            "axes": dict(axes or {}),
            "composite": composite,
            "gates_passed": bool(gates_passed),
            "failed_gates": list(failed_gates or []),
            "reliability": reliability,
            "artifact_detail": dict(artifact_detail or {}),
            "validation": validation,
            "processing_time_s": processing_time_s,
            "device": device,
            "embedder_fingerprint": dict(embedder_fingerprint or {}),
            "software_version": software_version,
            "cache_key": cache_key,
            "audio_available": bool(audio_available),
        }
        if output_analysis is not None:
            record["output_analysis"] = output_analysis
        if similarity_report is not None:
            record["similarity_report"] = similarity_report
        with self._lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                logger.exception("optimizer: experiment log append failed")
                return record
            if self._index is not None:
                self._index.append(record)
            if cache_key:
                self._by_key[cache_key] = record
        return record

    # ------------------------------------------------------------- queries
    def _reusable(self, record: dict[str, Any]) -> bool:
        return (record.get("optimizer_version") == OPTIMIZER_VERSION
                and record.get("schema") == _SCHEMA
                and record.get("model_hash") is not None)

    def trials_for(self, model_hash: str,
                   input_hash: str | None = None) -> list[dict[str, Any]]:
        """Reusable prior trials for a model (optionally same input)."""
        out = []
        for r in self._load():
            if not self._reusable(r):
                continue
            if r.get("model_hash") != model_hash:
                continue
            if input_hash is not None and \
                    r.get("input_hash") not in (None, input_hash):
                # different-input trials stay useful for warm start, so
                # they are returned when input_hash is not given
                continue
            out.append(r)
        return out

    def best_known(self, model_hash: str,
                  input_hash: str | None = None) -> dict[str, Any] | None:
        """Best gate-passing parameters ever recorded for this model."""
        best: dict[str, Any] | None = None
        for r in self.trials_for(model_hash, input_hash):
            if r.get("composite") is None or not r.get("gates_passed"):
                continue
            if best is None or float(r["composite"]) > \
                    float(best["composite"]):
                best = r
        if best is None:
            return None
        return {
            "params": best.get("params"),
            "composite": best.get("composite"),
            "axes": best.get("axes"),
            "session_id": best.get("session_id"),
            "recorded_at": best.get("recorded_at"),
            "source": ("same-input" if best.get("input_hash") == input_hash
                       else "prior-session"),
        }

    def warm_start_params(self, model_hash: str, k: int = 3,
                          exclude: set[str] | None = None) -> list[dict[str, Any]]:
        """Top-k distinct parameter sets to seed a new search.

        Prior same-model trials, best-first, deduplicated by params key;
        trials that already failed the gates are excluded (a warm start
        must not begin from known-bad settings).
        """
        exclude = exclude or set()
        ranked = sorted(
            (r for r in self.trials_for(model_hash)
             if r.get("composite") is not None and r.get("gates_passed")),
            key=lambda r: -float(r["composite"]))
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for r in ranked:
            key = params_key(r.get("params") or {})
            if key in seen or key in exclude:
                continue
            seen.add(key)
            out.append(dict(r["params"]))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------ conversion cache
    def lookup_conversion(
        self,
        model_hash: str | None,
        index_hash: str | None,
        input_hash: str | None,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Cached (analysis + similarity report) for this exact conversion.

        Returned ONLY when every hash and the optimizer version match —
        otherwise None (and the caller re-converts honestly).
        """
        if not model_hash or not input_hash:
            return None
        key = "|".join([
            OPTIMIZER_VERSION, str(model_hash), str(index_hash or "-"),
            str(input_hash), params_key(params),
        ])
        rec = self._by_key.get(key)
        if rec is None:
            return None
        if not self._reusable(rec):
            return None
        if "output_analysis" not in rec or "similarity_report" not in rec:
            return None
        return {
            "params": dict(rec.get("params") or {}),
            "output_analysis": rec.get("output_analysis"),
            "similarity_report": rec.get("similarity_report"),
            "recorded_at": rec.get("recorded_at"),
            "audio_available": bool(rec.get("audio_available")),
            "cached": True,
        }

    # ------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        records = self._load()
        reusable = [r for r in records if self._reusable(r)]
        models: dict[str, int] = {}
        for r in reusable:
            mh = str(r.get("model_hash"))
            models[mh] = models.get(mh, 0) + 1
        return {
            "n_records": len(records),
            "n_reusable": len(reusable),
            "optimizer_version": OPTIMIZER_VERSION,
            "n_models": len(models),
            "records_per_model": models,
            "oldest_record": min(
                (r.get("recorded_at") for r in records), default=None),
            "newest_record": max(
                (r.get("recorded_at") for r in records), default=None),
        }

    def sessions_summary(self, limit: int = 20) -> list[dict[str, Any]]:
        """Per-session aggregates (for cross-session comparison views)."""
        by_session: dict[str, dict[str, Any]] = {}
        for r in self._load():
            sid = str(r.get("session_id") or "unknown")
            slot = by_session.setdefault(sid, {
                "session_id": sid, "n_trials": 0, "best_composite": None,
                "optimizer_version": r.get("optimizer_version"),
                "model_hash": r.get("model_hash"),
                "first_at": r.get("recorded_at"),
                "last_at": r.get("recorded_at"),
                "reusable": self._reusable(r),
            })
            slot["n_trials"] += 1
            slot["last_at"] = r.get("recorded_at")
            if r.get("composite") is not None and r.get("gates_passed"):
                c = float(r["composite"])
                if slot["best_composite"] is None or \
                        c > slot["best_composite"]:
                    slot["best_composite"] = c
        out = sorted(by_session.values(),
                     key=lambda s: -(s.get("last_at") or 0))
        return out[:limit]
