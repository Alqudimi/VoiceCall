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

Phase 02 upgrades (audited W4 / W1 / W14):

* **per-sample quality assessment** — every folded sample is graded
  (duration, SNR proxy, clipping, silence share, voiced share); low-quality
  samples are flagged, never silently mixed in;
* **quality- and consistency-weighted centroid** per backend — a unified
  speaker representation, while individual sample embeddings are kept for
  outlier analysis and per-sample scoring;
* **outlier flagging** — a sample whose embedding sits far from the
  profile's other samples (different speaker / corrupted recording) is
  reported and down-weighted instead of polluting the anchor;
* **intra-speaker cohort stats per backend** — persisted so the
  calibrator rebuilds without touching audio (score calibration W1);
* **embedder fingerprint** — which backends + dimensions built this
  profile, so cross-process mismatches are detectable, never silent.
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

# quality-model thresholds (documented proxies, tuned on synthetic +
# clean-speech statistics; all values are reported, never hidden)
_MIN_GOOD_DURATION_S = 2.0
_MIN_GOOD_SNR_DB = 18.0
_MAX_OK_SNR_DB = 12.0
_MAX_OK_CLIPPING = 0.001
_MAX_OK_SILENCE = 0.65


def _safe(name: str, fallback: str = "profile") -> str:
    out = _SAFE.sub("_", (name or "").strip())[:PROFILE_NAME_MAX]
    return out or fallback


def _f(x: Any, nd: int = 4) -> float | None:
    try:
        v = float(x)
        return round(v, nd) if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def grade_sample_quality(analysis: dict[str, Any]) -> dict[str, Any]:
    """Grade ONE reference-sample analysis (0..1 + reasons).

    Honest, transparent scoring: every contributing factor and its value
    are returned so the UI and diagnostics can explain WHY a sample is
    graded low (never a black-box number).
    """
    dur = float(analysis.get("duration_s") or 0.0)
    loud = analysis.get("loudness") or {}
    snr = loud.get("snr_proxy_db")
    clip = loud.get("clipping_ratio")
    vad = analysis.get("vad") or {}
    voiced = vad.get("voiced_ratio")
    f0 = analysis.get("f0") or {}
    reasons: list[str] = []

    q_dur = float(np.clip(dur / _MIN_GOOD_DURATION_S, 0.0, 1.0))
    if dur < 1.0:
        reasons.append(f"very short sample ({dur:.1f}s) — embeddings unstable")
    elif dur < _MIN_GOOD_DURATION_S:
        reasons.append(f"short sample ({dur:.1f}s) — prefer >= 2s")

    if snr is None:
        q_snr = 0.5
        reasons.append("SNR unavailable")
    else:
        if snr >= _MIN_GOOD_SNR_DB:
            q_snr = 1.0
        elif snr <= _MAX_OK_SNR_DB:
            q_snr = 0.0
            reasons.append(f"noisy sample (SNR proxy {snr:.1f} dB "
                           "< 12 dB) — check the recording")
        else:
            q_snr = (snr - _MAX_OK_SNR_DB) / (
                _MIN_GOOD_SNR_DB - _MAX_OK_SNR_DB)
            reasons.append(f"moderate noise (SNR proxy {snr:.1f} dB)")

    if clip is None:
        q_clip = 0.5
    else:
        q_clip = 1.0 if clip <= _MAX_OK_CLIPPING else max(
            0.0, 1.0 - clip / (10 * _MAX_OK_CLIPPING + 1e-9))
        if clip > _MAX_OK_CLIPPING:
            reasons.append(f"clipping detected ({clip:.3%} of samples)")

    silence = 1.0 - (voiced if voiced is not None else 0.5)
    if silence > _MAX_OK_SILENCE:
        q_sil = max(0.0, 1.0 - (silence - _MAX_OK_SILENCE) / 0.35)
        reasons.append(f"mostly silence ({silence:.0%} non-speech)")
    else:
        q_sil = 1.0

    if (f0.get("voiced_ratio") or 0.0) < 0.15:
        reasons.append("barely any voiced/pitch content — check the content")

    score = float(0.35 * q_dur + 0.30 * q_snr + 0.15 * q_clip + 0.20 * q_sil)
    return {
        "score": round(score, 3),
        "grade": ("good" if score >= 0.75 else
                  "acceptable" if score >= 0.45 else "poor"),
        "factors": {
            "duration_s": _f(dur, 2),
            "snr_proxy_db": _f(snr, 1),
            "clipping_ratio": _f(clip, 5),
            "silence_share": _f(silence, 3),
            "voiced_ratio": _f(voiced, 3),
        },
        "reasons": reasons,
    }


class ReferenceProfile:
    """In-memory reference profile (build -> use -> persist)."""

    def __init__(self, model_name: str, name: str = "default") -> None:
        self.model_name = _safe(model_name, "model")
        self.name = _safe(name, "default")
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.sample_files: list[dict[str, Any]] = []   # [{id,file,duration_s}]
        self.embeddings: dict[str, list[list[float]]] = {}  # backend -> list
        # Phase 02: per-sample quality + weighted centroids + cohort stats
        self.sample_quality: list[dict[str, Any]] = []
        self.centroids: dict[str, list[float]] = {}
        self.sample_weights: list[float] = []
        self.embedder_fingerprint: dict[str, int] = {}   # backend -> dim
        self.stats: dict[str, Any] = {}                # aggregated statistics
        self.history: list[dict[str, Any]] = []        # optimization history

    # ---------------------------------------------------------------- build
    def add_sample_analysis(self, analysis: dict[str, Any],
                            sample_meta: dict[str, Any]) -> None:
        """Fold one analysed reference sample into the profile.

        Phase 02: the sample's quality is graded; poor samples are kept
        for transparency but their embeddings are down-weighted in the
        unified centroid (and flagged in the reliability report).
        """
        self.sample_files.append(sample_meta)
        quality = grade_sample_quality(analysis)
        self.sample_quality.append(quality)
        for backend, vec in (analysis.get("embeddings") or {}).items():
            arr = [float(v) for v in vec]
            self.embeddings.setdefault(backend, []).append(arr)
            dim = len(arr)
            prev = self.embedder_fingerprint.get(backend)
            self.embedder_fingerprint[backend] = dim if prev in (None, dim) \
                else -1                     # inconsistent dims -> mismatch
        self._fold(analysis)
        self.updated_at = time.time()

    def _fold(self, analysis: dict[str, Any]) -> None:
        """Aggregate reference statistics (means + spreads of every stat)."""
        acc = self.stats.setdefault("samples", [])
        acc.append(analysis)
        self.stats["n_samples"] = len(acc)

    # ---------------------------------------------------------- weighting
    def compute_sample_weights(self) -> list[float]:
        """Per-sample weights = quality score × embedding consistency.

        A sample that is both low-quality (noise/clipping/silence) AND whose
        embedding disagrees with the other samples gets a small weight;
        embeddings remain visible so the user can delete them instead.
        Weights are reported, not hidden.
        """
        n = len(self.sample_quality)
        if n == 0:
            return []
        qualities = np.array(
            [max(0.05, float(q.get("score") or 0.0))
             for q in self.sample_quality], dtype=np.float64)
        # embedding consistency per sample (mean sim to the other samples,
        # per backend, min across backends)
        consist = np.ones(n, dtype=np.float64)
        for backend, vecs in self.embeddings.items():
            if len(vecs) != n or n < 2:
                continue
            mat = np.asarray(vecs, dtype=np.float64)
            if mat.ndim != 2:
                continue
            mat = mat / np.maximum(
                np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
            sims = mat @ mat.T
            np.fill_diagonal(sims, np.nan)
            with np.errstate(invalid="ignore"):
                mean_sim = np.nanmean(sims, axis=1)
            consist = np.minimum(consist, np.nan_to_num(mean_sim, nan=0.5))
        # map consistency (say 0.3..0.9) to a multiplier 0.2..1.0
        mult = np.clip((consist - 0.2) / 0.7, 0.2, 1.0)
        w = qualities * mult
        self.sample_weights = (w / max(1e-9, w.sum()) * n).tolist()
        return self.sample_weights

    def weighted_centroids(self) -> dict[str, np.ndarray]:
        """Quality-weighted, L2-normalised centroid per backend."""
        out: dict[str, np.ndarray] = {}
        weights = self.compute_sample_weights()
        for backend, vecs in self.embeddings.items():
            if not vecs:
                continue
            mat = np.asarray(vecs, dtype=np.float64)
            if mat.ndim != 2:
                continue
            w = np.asarray(weights[: mat.shape[0]], dtype=np.float64) \
                if len(weights) >= mat.shape[0] else np.ones(mat.shape[0])
            if w.sum() <= 0:
                w = np.ones_like(w)
            w = w / w.sum()
            c = w @ mat
            norm = float(np.linalg.norm(c))
            out[backend] = c / norm if norm > 0 else c
        self.centroids = {k: v.tolist() for k, v in out.items()}
        return out

    def outlier_samples(self, threshold: float = 0.45) -> list[dict[str, Any]]:
        """Samples whose embeddings sit far from the rest (per backend).

        Returns [{index, file, similarity_to_rest, backend}] — evidence for
        "mixed speaker / corrupted recording" diagnostics.
        """
        out: list[dict[str, Any]] = []
        for backend, vecs in self.embeddings.items():
            if len(vecs) < 3:
                continue
            mat = np.asarray(vecs, dtype=np.float64)
            if mat.ndim != 2:
                continue
            mat = mat / np.maximum(
                np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
            sims = mat @ mat.T
            np.fill_diagonal(sims, np.nan)
            with np.errstate(invalid="ignore"):
                mean_sim = np.nanmean(sims, axis=1)
            for i, s in enumerate(mean_sim):
                if np.isfinite(s) and s < threshold:
                    meta = self.sample_files[i] if i < len(self.sample_files) \
                        else {"file": f"sample_{i}"}
                    out.append({
                        "index": i,
                        "file": meta.get("file"),
                        "backend": backend,
                        "similarity_to_rest": round(float(s), 4),
                    })
        return out

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
                    node = (node or {}).get(key) if isinstance(node, dict) \
                        else None
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
            "f0_jitter_semitones": stat(collect_num(
                ["f0", "jitter_semitones"])),
            "f0_octave_jump_rate": stat(collect_num(
                ["f0", "octave_jump_rate"])),
            "f0_voiced_ratio": stat(collect_num(["f0", "voiced_ratio"])),
            "hnr_db": stat(collect_num(["hnr_db"])),
            "shimmer_db": stat(collect_num(["shimmer_db"])),
            "spectral_flatness": stat(collect_num(["spectral", "flatness"])),
            "spectral_centroid_hz": stat(collect_num(
                ["spectral", "centroid_hz"])),
            "rms_db": stat(collect_num(["loudness", "rms_db"])),
            "noise_floor_db": stat(collect_num(["loudness", "noise_floor_db"])),
            "duration_s": stat(collect_num(["duration_s"])),
        }
        # pitch histogram (mean over samples)
        hists = [s.get("f0", {}).get("histogram") or [] for s in samples]
        hists = [h for h in hists if len(h) == 24]
        if hists:
            agg["f0_histogram"] = np.mean(
                np.asarray(hists), axis=0).tolist()
        # MFCC mean + band energies mean
        mf = [s.get("mfcc_mean") or [] for s in samples]
        mf = [m for m in mf if len(m) == 20]
        if mf:
            agg["mfcc_mean"] = np.mean(np.asarray(mf), axis=0).tolist()
            agg["mfcc_std_across_samples"] = np.std(
                np.asarray(mf), axis=0).tolist()
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

        # Phase 02: cohorts (intra-speaker similarity stats per backend) —
        # persisted for the calibrator, rebuilt from embeddings
        try:
            from rvc_platform.optimizer.calibration import \
                SpeakerScoreCalibrator

            cal = SpeakerScoreCalibrator()
            cohorts = cal.fit_from_embeddings(self.embeddings)
            self.stats["cohorts"] = {
                k: v.to_dict() for k, v in cohorts.items()}
        except Exception:
            logger.debug("optimizer: cohort stats failed", exc_info=True)
        self.weighted_centroids()
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
            return {"level": "empty",
                    "notes": ["no reference samples analysed"]}
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
        # Phase 02: quality + outlier reporting
        poor = [i for i, q in enumerate(self.sample_quality)
                if (q.get("grade") == "poor")]
        if poor:
            names = [self.sample_files[i].get("file") if i < len(
                self.sample_files) else str(i) for i in poor]
            notes.append(f"low-quality reference sample(s): "
                         f"{', '.join(str(x) for x in names)} — graded poor "
                         "(noise/clipping/silence); they are down-weighted, "
                         "consider replacing them")
        outliers = self.outlier_samples()
        if outliers:
            files = sorted({str(o.get("file")) for o in outliers})
            notes.append(
                f"possible outlier sample(s) — embeddings far from the rest: "
                f"{', '.join(files)}; check for a different speaker or a "
                "corrupted recording")
        return {
            "level": level,
            "n_samples": n,
            "total_duration_s": _f(total_dur, 2),
            "embedding_counts": backends,
            "self_consistency": consistency,
            "sample_quality": [
                {"file": (self.sample_files[i].get("file")
                          if i < len(self.sample_files) else str(i)),
                 "score": q.get("score"), "grade": q.get("grade")}
                for i, q in enumerate(self.sample_quality)],
            "outliers": outliers,
            "notes": notes,
        }

    # --------------------------------------------------------------- persist
    def to_dict(self) -> dict[str, Any]:
        if (self.stats.get("samples")) and not self.stats.get("aggregate"):
            self.aggregate()
        return {
            "model": self.model_name,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sample_files": self.sample_files,
            "embeddings": self.embeddings,
            "sample_quality": self.sample_quality,
            "centroids": self.centroids,
            "sample_weights": self.sample_weights,
            "embedder_fingerprint": self.embedder_fingerprint,
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
        p.sample_quality = list(data.get("sample_quality") or [])
        p.centroids = {
            k: list(map(float, v))
            for k, v in (data.get("centroids") or {}).items()
        }
        p.sample_weights = [float(w) for w in (data.get("sample_weights")
                                               or [])]
        p.embedder_fingerprint = dict(
            data.get("embedder_fingerprint") or {})
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

    def sample_files(self, model_name: str,
                     profile_name: str) -> list[Path]:
        d = self.samples_dir(model_name, profile_name)
        exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
                ".opus", ".webm"}
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
        path = self._profile_dir(profile.model_name, profile.name) \
            / "profile.json"
        path.write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False),
            encoding="utf-8")
        return path

    def load(self, model_name: str,
             profile_name: str) -> ReferenceProfile | None:
        path = (self._profile_dir(model_name, profile_name)
                / "profile.json")
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ReferenceProfile.from_dict(data)
        except (json.JSONDecodeError, OSError, ValueError):
            logger.exception("optimizer: profile %s/%s unreadable",
                            model_name, profile_name)
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
                "reliability":
                    ReferenceProfile.from_dict(data).reliability_report()
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

    def save_applied(self, model_name: str,
                     payload: dict[str, Any]) -> Path:
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
