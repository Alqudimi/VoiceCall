"""Calibration — per-backend speaker-score calibration (Phase 02 W1)."""
from __future__ import annotations

import numpy as np

from rvc_platform.optimizer.calibration import (
    SpeakerScoreCalibrator,
)


def _emb(seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float64)
    return v / np.linalg.norm(v)


def _profile_embeddings(n: int, jitter: float = 0.15, base=None,
                        dim: int = 16) -> list[np.ndarray]:
    base = base if base is not None else _emb(1, dim)
    out = []
    for i in range(n):
        v = base + jitter * _emb(10 + i, dim)
        out.append(v / np.linalg.norm(v))
    return out


def test_cohort_built_from_two_or_more_samples():
    cal = SpeakerScoreCalibrator()
    cal.fit_from_embeddings({
        "wavlm": _profile_embeddings(3),
        "ecapa": _profile_embeddings(1),       # no cohort possible
    })
    assert cal.has_cohort("wavlm")
    assert not cal.has_cohort("ecapa")


def test_cohort_stats_are_real_distribution_values():
    cal = SpeakerScoreCalibrator()
    embs = _profile_embeddings(4)
    cal.fit_from_embeddings({"wavlm": embs})
    c = cal.cohorts["wavlm"]
    mat = np.vstack(embs)
    sims = (mat @ mat.T)[np.triu_indices(4, k=1)]
    assert c.n_pairs == 6
    assert c.mean == float(np.mean(sims))
    assert c.std > 0


def test_raw_score_at_cohort_mean_calibrates_to_half():
    cal = SpeakerScoreCalibrator()
    embs = _profile_embeddings(5)
    cal.fit_from_embeddings({"wavlm": embs})
    c = cal.cohorts["wavlm"]
    report = cal.calibrate_report({"wavlm": c.mean})
    entry = report["per_backend"]["wavlm"]
    assert entry["margin"] == 0.0
    assert abs(entry["calibrated"] - 0.5) < 1e-6
    assert report["calibrated_ensemble"] == 0.5


def test_above_cohort_scores_above_half_and_below_below_half():
    cal = SpeakerScoreCalibrator()
    embs = _profile_embeddings(5)
    cal.fit_from_embeddings({"wavlm": embs})
    c = cal.cohorts["wavlm"]
    hi = cal.calibrate_report({"wavlm": c.mean + 2 * c.std})
    lo = cal.calibrate_report({"wavlm": c.mean - 2 * c.std})
    assert hi["per_backend"]["wavlm"]["calibrated"] > 0.85
    assert lo["per_backend"]["wavlm"]["calibrated"] < 0.15


def test_raw_values_always_reported_next_to_calibrated():
    cal = SpeakerScoreCalibrator()
    cal.fit_from_embeddings({"wavlm": _profile_embeddings(3)})
    report = cal.calibrate_report({"wavlm": 0.81})
    assert report["raw"]["wavlm"] == 0.81
    assert "calibrated" in report["per_backend"]["wavlm"]


def test_ensemble_withheld_without_any_calibratable_backend():
    cal = SpeakerScoreCalibrator()
    report = cal.calibrate_report({"ecapa": 0.4})     # no cohort fitted
    assert report["calibrated_ensemble"] is None
    assert report["raw"]["ecapa"] == 0.4
    assert any("uncalibrated" in n or "no cohort" in n
               for n in report["notes"])


def test_models_on_different_scales_end_up_comparable():
    """Two backends with very different raw ranges: the calibrated ensemble
    must not simply average the raw numbers."""
    cal = SpeakerScoreCalibrator()
    # wavlm same-speaker raws ~0.9x; ecapa same-speaker raws ~0.3x
    embs_w = _profile_embeddings(4, jitter=0.1)
    rng = np.random.default_rng(2)
    base_e = _emb(3, 16)
    embs_e = []
    for i in range(4):
        v = base_e + 0.5 * rng.standard_normal(16)
        embs_e.append(v / np.linalg.norm(v))
    cal.fit_from_embeddings({"wavlm": embs_w, "ecapa": embs_e})
    c_w = cal.cohorts["wavlm"]
    c_e = cal.cohorts["ecapa"]
    # candidate at each backend's own cohort mean -> ensemble 0.5
    report = cal.calibrate_report({
        "wavlm": c_w.mean, "ecapa": c_e.mean})
    assert abs(report["calibrated_ensemble"] - 0.5) < 1e-6
    # raw ensemble would have been (c_w.mean + c_e.mean)/2 != 0.5
    raw_mean = 0.5 * (c_w.mean + c_e.mean)
    assert abs(raw_mean - 0.5) > 0.05


def test_persistence_roundtrip():
    cal = SpeakerScoreCalibrator()
    cal.fit_from_embeddings({"wavlm": _profile_embeddings(3)})
    data = cal.to_dict()
    cal2 = SpeakerScoreCalibrator()
    cal2.load_dict(data)
    assert cal2.has_cohort("wavlm")
    c1 = cal.cohorts["wavlm"]
    c2 = cal2.cohorts["wavlm"]
    assert c1.mean == c2.mean and c1.n_pairs == c2.n_pairs


def test_fit_from_profile_uses_persisted_cohorts():
    from rvc_platform.optimizer.profile import ReferenceProfile

    prof = ReferenceProfile("m", "p")
    base = _emb(1)
    for i in range(3):
        # each sample embeds the same speaker with small variation -> a
        # real intra-speaker cohort with non-zero spread
        v = base + 0.15 * _emb(10 + i)
        v = v / np.linalg.norm(v)
        analysis = {"embeddings": {"wavlm": v.tolist()},
                    "duration_s": 3.0, "loudness": {}, "vad": {}}
        prof.add_sample_analysis(analysis, {"id": f"s{i}",
                                            "file": f"s{i}.wav"})
    prof.aggregate()
    cal = SpeakerScoreCalibrator()
    cal.fit_from_profile(prof.to_dict())
    assert cal.has_cohort("wavlm")
