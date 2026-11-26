"""Reference speaker profiles — per-model, persisted, multi-profile.

A *reference profile* is the optimizer's memory of the target speaker:
neural speaker-verification embeddings of clean samples, F0 statistics,
timbre statistics (MFCC / band profile / spectral stats / formants) and
voice-quality anchors (HNR, jitter, shimmer, loudness profile).

Storage layout (under the platform data root)::

    data/optimizer/reference/<model>/<profile>/samples/<uuid>_<name>.<ext>
    data/optimizer/reference/<model>/<profile>/profile.json
    data/optimizer/applied/<model>.json        (applied best settings)

* one model may own SEVERAL profiles (different sample sets / sessions);
* a profile never leaks to another model: the directory tree is keyed by
  model name and the optimizer refuses to score against a mismatched
  model's profile;
* ``profile.json`` keeps the aggregated statistics + embedding lists so a
  new server process rebuilds the profile without re-analysing audio.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from rvc_platform.infrastructure.paths import get_paths

logger = logging.getLogger(__name__)

_SAFE = re.compile(r"[^A-Za-z0-9_.\-]")
PROFILE_NAME_MAX = 60


def _safe(name: str, fallback: str = "profile") -> str:
    out = _SAFE.sub("_", (name or "").strip())[:PROFILE_NAME_MAX]
    return out or fallback


def _f(x: Any, nd: int = 4) -> float | None:
    try:
        v = float(x)
        return round(v, nd) if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


class ReferenceProfile:
    """In-memory reference profile (build -> use -> persist)."""

    def __init__(self, model_name: str, name: str = "default") -> None:
        self.model_name = _safe(model_name, "model")
        self.name = _safe(name, "default")
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.sample_files: list[dict[str, Any]] = []   # [{id,file,duration_s}]
        self.embeddings: dict[str, list[list[float]]] = {}  # backend -> list
        self.stats: dict[str, Any] = {}                # aggregated statistics
        self.history: list[dict[str, Any]] = []        # optimization history

    # ---------------------------------------------------------------- build
    def add_sample_analysis(self, analysis: dict[str, Any],
                            sample_meta: dict[str, Any]) -> None:
        """Fold one analysed reference sample into the profile."""
        self.sample_files.append(sample_meta)
        for backend, vec in (analysis.get("embeddings") or {}).items():
            self.embeddings.setdefault(backend, []).append(
                [float(v) for v in vec])
        self._fold(analysis)
        self.updated_at = time.time()

    def _fold(self, analysis: dict[str, Any]) -> None:
        """Aggregate reference statistics (means + spreads of every stat)."""
        acc = self.stats.setdefault("samples", [])
        acc.append(analysis)
        # keep the aggregated view recomputed lazily via aggregate()
        self.stats["n_samples"] = len(acc)

    # -------------------------------------------------------------- aggregate
    def aggregate(self) -> dict[str, Any]:
        """Recompute aggregate statistics over the folded sample analyses."""
        samples = self.stats.get("samples") or []
        n = len(samples)
        if not n:
            return {}

        def collect_num(path: list[str]) -> list[float]:
            vals = []
            for s in samples:
                node: Any = s
                for key in path:
                    node = (node or {}).get(key) if isinstance(node, dict) else None
                if node is None:
                    continue
                if isinstance(node, (int, float)):
                    vals.append(float(node))
            return vals

        def stat(vals: list[float]) -> dict[str, float | None]:
            if not vals:
                return {"median": None, "mean": None, "std": None}
            arr = np.asarray(vals, dtype=np.float64)
            return {
                "median": _f(np.median(arr)),
                "mean": _f(arr.mean()),
                "std": _f(arr.std()),
            }

        agg: dict[str, Any] = {
            "n_samples": n,
            "f0_median_hz": stat(collect_num(["f0", "median_hz"])),
            "f0_range_semitones": stat(collect_num(["f0", "range_semitones"])),
            "f0_jitter_semitones": stat(collect_num(["f0", "jitter_semitones"])),
            "f0_octave_jump_rate": stat(collect_num(["f0", "octave_jump_rate"])),
            "f0_voiced_ratio": stat(collect_num(["f0", "voiced_ratio"])),
            "hnr_db": stat(collect_num(["hnr_db"])),
            "shimmer_db": stat(collect_num(["shimmer_db"])),
            "spectral_flatness": stat(collect_num(["spectral", "flatness"])),
            "spectral_centroid_hz": stat(collect_num(["spectral", "centroid_hz"])),
            "rms_db": stat(collect_num(["loudness", "rms_db"])),
            "noise_floor_db": stat(collect_num(["loudness", "noise_floor_db"])),
            "duration_s": stat(collect_num(["duration_s"])),
        }
        # pitch histogram (mean over samples)
        hists = [s.get("f0", {}).get("histogram") or [] for s in samples]
        hists = [h for h in hists if len(h) == 24]
        if hists:
            agg["f0_histogram"] = np.mean(np.asarray(hists), axis=0).tolist()
        # MFCC mean + band energies mean
        mf = [s.get("mfcc_mean") or [] for s in samples]
        mf = [m for m in mf if len(m) == 20]
        if mf:
            agg["mfcc_mean"] = np.mean(np.asarray(mf), axis=0).tolist()
            agg["mfcc_std_across_samples"] = np.std(np.asarray(mf), axis=0).tolist()
        be = [s.get("band_energies") or [] for s in samples]
        be = [b for b in be if len(b) == 8]
        if be:
            agg["band_energies"] = np.mean(np.asarray(be), axis=0).tolist()
        # formants (median of medians; formants_hz is a flat list of 4
        # (values or None) per sample)
        f_agg = []
        for j in range(4):
            vals = []
            for s in samples:
                fh = s.get("formants_hz") or []
                if len(fh) > j and fh[j] is not None:
                    vals.append(float(fh[j]))
            f_agg.append(_f(np.median(vals), 1) if vals else None)
        agg["formants_hz"] = f_agg
        self.stats["aggregate"] = agg
        return agg

    # ------------------------------------------------------------- reliability
    def reliability_report(self) -> dict[str, Any]:
        """Honest profile reliability: sample count, duration, consistency."""
        samples = self.stats.get("samples") or []
        n = len(samples)
        total_dur = float(np.sum([
            float(s.get("duration_s") or 0.0) for s in samples
        ])) if n else 0.0
        backends = {k: len(v) for k, v in self.embeddings.items()}
        notes: list[str] = []
        level = "ok"
        if n == 0:
            return {"level": "empty", "notes": ["no reference samples analysed"]}
        if n < 2:
            level = "low"
            notes.append("single reference sample — add 2-5 varied samples "
                         "for a more robust identity anchor")
        if total_dur < 6.0:
            level = "low" if level == "ok" else level
            notes.append(f"only {total_dur:.1f}s of reference speech — "
                         "prefer >= 10s across samples")
        # embedding self-consistency: mean pairwise cosine within backends
        consistency: dict[str, float | None] = {}
        for backend, vecs in self.embeddings.items():
            if len(vecs) >= 2:
                mat = np.asarray(vecs, dtype=np.float64)
                mat = mat / np.maximum(
                    np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
                sims = (mat @ mat.T)[np.triu_indices(len(vecs), k=1)]
                consistency[backend] = _f(float(np.mean(sims)))
        if consistency:
            worst = min(v for v in consistency.values() if v is not None)
            if worst < 0.6:
                level = "warn"
                notes.append(
                    f"reference samples are not very consistent (worst "
                    f"self-similarity {worst:.2f}) — check for mixed speakers "
                    "or very different recording conditions")
        return {
            "level": level,
            "n_samples": n,
            "total_duration_s": _f(total_dur, 2),
            "embedding_counts": backends,
            "self_consistency": consistency,
            "notes": notes,
        }

    # --------------------------------------------------------------- persist
    def to_dict(self) -> dict[str, Any]:
        # ensure the aggregate view exists whenever there are samples
        # (a freshly built profile must serialise with its statistics)
        if (self.stats.get("samples")) and not self.stats.get("aggregate"):
            self.aggregate()
        return {
            "model": self.model_name,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sample_files": self.sample_files,
            "embeddings": self.embeddings,
            "stats": self.stats,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceProfile":
        p = cls(data.get("model", "model"), data.get("name", "default"))
        p.created_at = float(data.get("created_at") or time.time())
        p.updated_at = float(data.get("updated_at") or time.time())
        p.sample_files = list(data.get("sample_files") or [])
        p.embeddings = {
            k: [list(map(float, v)) for v in vecs]
            for k, vecs in (data.get("embeddings") or {}).items()
        }
        p.stats = dict(data.get("stats") or {})
        p.history = list(data.get("history") or [])
        return p


class ReferenceProfileStore:
    """Filesystem store: samples in, profiles out, per-model isolation."""

    def __init__(self) -> None:
        p = get_paths()
        self.root = p.data / "optimizer" / "reference"
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- paths
    def _model_dir(self, model_name: str) -> Path:
        d = self.root / _safe(model_name, "model")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _profile_dir(self, model_name: str, profile_name: str) -> Path:
        d = self._model_dir(model_name) / _safe(profile_name, "default")
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --------------------------------------------------------------- samples
    def store_sample(self, model_name: str, profile_name: str,
                     filename: str, payload: bytes) -> dict[str, Any]:

        from rvc_platform.infrastructure.audio.io import (
            ALLOWED_AUDIO_SUFFIXES, validate_audio_file,
        )
        if not payload:
            raise ValueError("empty upload")
        safe = _SAFE.sub("_", filename or "sample.wav")
        suffix = Path(safe).suffix.lower()
        if suffix not in ALLOWED_AUDIO_SUFFIXES:
            raise ValueError(f"unsupported audio format: {suffix}")
        dest = self._profile_dir(model_name, profile_name) / "samples"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"{uuid.uuid4().hex[:8]}_{safe}"
        path.write_bytes(payload)
        try:
            validate_audio_file(path, max_mb=200)   # magic-byte + size check
        except Exception as exc:
            try:
                path.unlink(missing_ok=True)      # never keep rejected files
            except OSError:
                pass
            raise ValueError(str(exc)) from exc
        return {"id": path.stem, "file": path.name,
                "size_bytes": len(payload)}

    def samples_dir(self, model_name: str, profile_name: str) -> Path:
        d = self._profile_dir(model_name, profile_name) / "samples"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def sample_files(self, model_name: str, profile_name: str) -> list[Path]:
        d = self.samples_dir(model_name, profile_name)
        exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm"}
        return sorted(p for p in d.iterdir() if p.suffix.lower() in exts)

    def delete_sample(self, model_name: str, profile_name: str,
                      sample_id: str) -> bool:
        safe = _SAFE.sub("_", sample_id)
        for p in self.sample_files(model_name, profile_name):
            if p.stem == safe:
                p.unlink(missing_ok=True)
                return True
        return False

    # --------------------------------------------------------------- profile
    def save(self, profile: ReferenceProfile) -> Path:
        profile.aggregate()
        path = self._profile_dir(profile.model_name, profile.name) / "profile.json"
        path.write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False), encoding="utf-8")
        return path

    def load(self, model_name: str, profile_name: str) -> ReferenceProfile | None:
        path = (self._profile_dir(model_name, profile_name) / "profile.json")
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ReferenceProfile.from_dict(data)
        except (json.JSONDecodeError, OSError, ValueError):
            logger.exception("optimizer: profile %s/%s unreadable", model_name,
                             profile_name)
            return None

    def list_profiles(self, model_name: str) -> list[dict[str, Any]]:
        out = []
        model_dir = self.root / _safe(model_name, "model")
        if not model_dir.is_dir():
            return out
        for d in sorted(model_dir.iterdir()):
            if not d.is_dir():
                continue
            pj = d / "profile.json"
            if not pj.is_file():
                continue
            try:
                data = json.loads(pj.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            n_samples = len(data.get("sample_files") or [])
            out.append({
                "name": d.name,
                "model": model_name,
                "n_samples": n_samples,
                "updated_at": data.get("updated_at"),
                "reliability": ReferenceProfile.from_dict(data).reliability_report()
                if n_samples else {"level": "empty", "notes": []},
            })
        return out

    def delete_profile(self, model_name: str, profile_name: str) -> bool:
        import shutil

        d = self._profile_dir(model_name, profile_name)
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            return True
        return False

    # -------------------------------------------------------- applied results
    def applied_path(self, model_name: str) -> Path:
        d = self.root.parent / "applied"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{_safe(model_name, 'model')}.json"

    def save_applied(self, model_name: str, payload: dict[str, Any]) -> Path:
        path = self.applied_path(model_name)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    def load_applied(self, model_name: str) -> dict[str, Any] | None:
        path = self.applied_path(model_name)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
