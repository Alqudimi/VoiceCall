"""Per-backend speaker-similarity calibration (Phase 02 — W1).

Cosine similarities from *different* speaker-verification models are NOT
directly comparable: WavLM x-vector same-speaker cosines typically live in a
different range than ECAPA-TDNN same-speaker cosines. Averaging raw scores
across backends therefore biases the ensemble toward whichever model happens
to produce higher raw numbers.

This module builds, **per backend**, a same-speaker cohort distribution from
the reference profile's own samples (intra-speaker pairwise cosines), and
converts every raw cosine into:

* ``margin``      — a z-score against the speaker's own cohort
                    (``(raw - cohort_mean) / cohort_std``): 0 means "exactly
                    as similar as the speaker's own samples are to each
                    other";
* ``calibrated``  — a bounded 0..1 score via a logistic squash of the
                    margin, so different backends become comparable before
                    any ensemble; 0.5 == cohort-typical, >0.5 == above the
                    speaker's own intra-speaker spread, <0.5 == below.

Honesty rules:

* a backend with fewer than TWO reference samples has no cohort (std is
  undefined) — its raw cosine is reported unchanged, the ensemble is
  flagged ``"uncalibrated-backends"`` and reliability is degraded instead of
  silently averaging incomparable scales;
* raw per-backend cosines are ALWAYS reported next to the calibrated values
  (nothing is hidden behind the calibration);
* the ensemble is the mean of *calibrated* scores only across backends that
  have a cohort; a single available calibrated backend passes through
  unchanged (no averaging happens at all).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["BackendCohort", "SpeakerScoreCalibrator"]


@dataclass
class BackendCohort:
    """Same-speaker cosine distribution for ONE embedding backend."""

    backend: str
    n_pairs: int = 0
    mean: float = 0.0
    std: float = 0.0
    # robust fallbacks: median / MAD for small-sample honesty
    median: float = 0.0
    min: float = 0.0
    max: float = 0.0

    @property
    def usable(self) -> bool:
        """A cohort needs >= 2 samples (>= 1 pair) and a positive spread."""
        return self.n_pairs >= 1 and self.std > 1e-6

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "n_pairs": self.n_pairs,
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "median": round(self.median, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
        }


class SpeakerScoreCalibrator:
    """Calibrates raw per-backend cosines against per-backend cohorts."""

    def __init__(self) -> None:
        self._cohorts: dict[str, BackendCohort] = {}

    # ------------------------------------------------------------------ fit
    def fit_from_embeddings(
        self, embeddings: dict[str, list[np.ndarray]],
    ) -> dict[str, BackendCohort]:
        """Build cohorts from a profile's per-backend embedding lists.

        The intra-speaker cohort of a backend is the distribution of all
        pairwise cosines between that speaker's own reference embeddings —
        i.e. how similar the speaker is to *herself* under that model.
        """
        for backend, vecs in (embeddings or {}).items():
            mats = [np.asarray(v, dtype=np.float64).ravel() for v in vecs]
            mats = [m for m in mats if m.size and np.isfinite(m).all()]
            # all embeddings of one backend must share its dimensionality
            dims = {m.size for m in mats}
            if len(dims) > 1:
                continue
            if len(mats) < 2:
                self._cohorts[backend] = BackendCohort(backend, n_pairs=0)
                continue
            mat = np.vstack(mats)
            mat = mat / np.maximum(
                np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
            sims = (mat @ mat.T)[np.triu_indices(len(mats), k=1)]
            self._cohorts[backend] = BackendCohort(
                backend=backend,
                n_pairs=int(sims.size),
                mean=float(sims.mean()) if sims.size else 0.0,
                std=float(sims.std()) if sims.size else 0.0,
                median=float(np.median(sims)) if sims.size else 0.0,
                min=float(sims.min()) if sims.size else 0.0,
                max=float(sims.max()) if sims.size else 0.0,
            )
        return dict(self._cohorts)

    @property
    def cohorts(self) -> dict[str, BackendCohort]:
        return dict(self._cohorts)

    def has_cohort(self, backend: str) -> bool:
        c = self._cohorts.get(backend)
        return c is not None and c.usable

    # ------------------------------------------------------------- calibrate
    @staticmethod
    def calibrate_value(raw: float, cohort: BackendCohort) -> tuple[float, float]:
        """(margin, calibrated) for one raw cosine against one cohort."""
        margin = (float(raw) - cohort.mean) / cohort.std
        # logistic squash: margin 0 -> 0.5, +2 -> ~0.88, -2 -> ~0.12
        calibrated = 1.0 / (1.0 + math.exp(-margin))
        return margin, calibrated

    def calibrate_report(
        self, raw_similarities: dict[str, float],
    ) -> dict[str, Any]:
        """Full calibration view of one similarity report.

        Returns raw values (unchanged, always reported), per-backend
        margin/calibrated values, the calibrated ensemble, and honest notes
        about backends that could not be calibrated.
        """
        raw = {k: float(v) for k, v in (raw_similarities or {}).items()}
        per_backend: dict[str, dict[str, float]] = {}
        calibrated_values: list[float] = []
        notes: list[str] = []
        for backend, value in raw.items():
            entry: dict[str, float] = {"raw": round(value, 4)}
            cohort = self._cohorts.get(backend)
            if cohort is not None and cohort.usable:
                margin, calibrated = self.calibrate_value(value, cohort)
                entry["margin"] = round(margin, 3)
                entry["calibrated"] = round(calibrated, 4)
                calibrated_values.append(calibrated)
            else:
                entry["calibrated"] = None
                if cohort is None:
                    notes.append(
                        f"{backend}: no cohort (profile has < 2 samples for "
                        "this backend) — raw cosine reported uncalibrated")
                else:
                    notes.append(
                        f"{backend}: cohort spread is zero — raw cosine "
                        "reported uncalibrated")
            per_backend[backend] = entry
        ensemble: float | None = None
        if calibrated_values:
            ensemble = float(np.mean(calibrated_values))
        elif raw:
            # nothing calibrated — the ensemble stays undefined rather than
            # silently averaging incomparable raw scales
            notes.append("ensemble withheld: no backend could be calibrated")
        return {
            "raw": {k: round(v, 4) for k, v in raw.items()},
            "per_backend": per_backend,
            "calibrated_ensemble": (
                round(ensemble, 4) if ensemble is not None else None),
            "n_calibrated": len(calibrated_values),
            "notes": notes,
        }

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "cohorts": {
                k: {
                    "n_pairs": c.n_pairs, "mean": c.mean, "std": c.std,
                    "median": c.median, "min": c.min, "max": c.max,
                } for k, c in self._cohorts.items()
            },
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        for backend, c in ((data or {}).get("cohorts") or {}).items():
            self._cohorts[backend] = BackendCohort(
                backend=backend,
                n_pairs=int(c.get("n_pairs", 0)),
                mean=float(c.get("mean", 0.0)),
                std=float(c.get("std", 0.0)),
                median=float(c.get("median", 0.0)),
                min=float(c.get("min", 0.0)),
                max=float(c.get("max", 0.0)),
            )

    def fit_from_profile(self, profile_dict: dict[str, Any]) -> None:
        """Convenience: fit directly from ``ReferenceProfile.to_dict()``."""
        emb = (profile_dict or {}).get("embeddings") or {}
        # prefer cohort stats persisted by the profile when present
        persisted = (
            (profile_dict or {}).get("stats", {}).get("cohorts") or {})
        if persisted:
            self.load_dict({"cohorts": persisted})
        else:
            self.fit_from_embeddings(emb)
