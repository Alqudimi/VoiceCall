"""Root-cause diagnostics (Phase 02 W15)."""
from __future__ import annotations

from rvc_platform.optimizer.diagnostics import (
    model_quality_report,
    root_cause_analysis,
)


def _trial(comp: float = 0.6, err: str | None = None):
    return {"params": {"index_rate": 0.75}, "composite": comp,
            "axes": {"speaker_similarity": comp},
            "gates_passed": err is None, "error": err}


def _best(comp=0.72):
    return {"params": {"index_rate": 0.8}, "composite": comp,
            "axes": {"speaker_similarity": comp}}


class TestModelQualityReport:
    def test_missing_index_is_reported(self):
        rep = model_quality_report(
            {"path": "/nonexistent.pth", "index_path": None}, None, None)
        statuses = " ".join(c["status"] for c in rep["checks"])
        assert "no index file" in statuses

    def test_roundtrip_ceiling_is_the_strongest_evidence(self):
        rep = model_quality_report(
            {"path": "/nonexistent.pth"}, None, None,
            roundtrip={"speaker_similarity": 0.3})
        limit = [c for c in rep["checks"]
                 if c["check"] == "roundtrip_speaker_similarity"]
        assert limit
        assert "LIMIT" in limit[0]["status"]

    def test_unavailable_measurements_are_listed_honestly(self):
        rep = model_quality_report({"path": "/nonexistent.pth"}, None, None)
        assert any("NOT observable" in u for u in rep["unavailable"])
        assert "proxy-based" in rep["disclaimer"]


class TestRootCauseAnalysis:
    def test_no_improvement_points_away_from_more_trials(self):
        rep = root_cause_analysis(
            trials=[_trial(0.5), _trial(0.51), _trial(0.5)],
            baseline={"evaluation": {"composite": 0.50}},
            best={"composite": 0.51, "params": {}, "axes": {}},
            validation=[], profile_reliability={"level": "ok",
                                                "n_samples": 3,
                                                "total_duration_s": 12,
                                                "notes": []},
            input_analyses=[{"source": "input.wav", "duration_s": 8,
                             "loudness": {"snr_proxy_db": 25,
                                          "clipping_ratio": 0.0},
                             "vad": {"voiced_ratio": 0.6}}],
            model_info={"path": "/m.pth", "index_path": None},
        )
        inf_params = rep["problem_classes"]["inference_parameters"]
        no_gain = [d for d in inf_params
                   if "no improvement" in (d.get("problem") or "")]
        assert no_gain
        assert "more search will not fix" in \
            no_gain[0]["recommendation"]

    def test_improvement_entry_carries_the_engine_action(self):
        rep = root_cause_analysis(
            trials=[_trial(0.5), _trial(0.7)],
            baseline={"evaluation": {"composite": 0.50}},
            best=_best(0.72),
            validation=[], profile_reliability={"level": "ok"},
            input_analyses=[],
            model_info={"path": "/m.pth", "index_path": "/i.index"},
            insights={"status": "ok",
                      "recommendations": [
                          "index_rate: highest composite observed in 0.7-0.9 "
                          "(mean 0.71, n=6); sensitivity 80%"]},
        )
        inf = [d for d in rep["problem_classes"]["inference_parameters"]
               if d.get("summary")]
        assert inf and "improved composite by" in inf[0]["summary"]
        assert inf[0]["engine_action"] and "index_rate" in inf[0]["engine_action"]

    def test_weak_reference_profile_diagnosed_with_evidence(self):
        rep = root_cause_analysis(
            trials=[_trial(0.5)], baseline=None, best=None, validation=[],
            profile_reliability={
                "level": "low", "n_samples": 1, "total_duration_s": 3.0,
                "notes": ["single reference sample"],
                "embedding_counts": {"wavlm": 1},
                "sample_quality": [{"file": "a.wav", "score": 0.3,
                                    "grade": "poor"}],
                "outliers": []},
            input_analyses=[], model_info=None)
        ref = rep["problem_classes"]["reference_data"]
        assert ref
        assert any("calibration" in (d.get("problem") or "")
                   for d in ref)       # embedding counts < 2

    def test_noisy_input_diagnosed(self):
        rep = root_cause_analysis(
            trials=[_trial(0.5)], baseline=None, best=None, validation=[],
            profile_reliability={"level": "ok"},
            input_analyses=[{
                "source": "input.wav", "duration_s": 8.0,
                "loudness": {"snr_proxy_db": 8.0,
                             "clipping_ratio": 0.01},
                "vad": {"voiced_ratio": 0.6}}],
            model_info=None)
        inp = rep["problem_classes"]["input_audio"]
        assert inp
        assert "noisy input" in inp[0]["problem"]

    def test_missing_embedder_diagnosed(self):
        rep = root_cause_analysis(
            trials=[], baseline=None, best=None, validation=[],
            profile_reliability={"level": "ok"}, input_analyses=[],
            model_info=None,
            embedder_status={"wavlm": False, "ecapa": False,
                             "reliability": "low", "messages": {}})
        ev = rep["problem_classes"]["evaluation_models"]
        assert ev
        assert "DSP voiceprint fallback" in ev[0]["problem"]

    def test_overfit_validation_drop_diagnosed(self):
        rep = root_cause_analysis(
            trials=[_trial(0.7)], baseline=None, best=_best(0.7),
            validation=[{"drop": 0.3}],
            profile_reliability={"level": "ok"}, input_analyses=[],
            model_info=None)
        inf = rep["problem_classes"]["inference_parameters"]
        assert any("overfitted" in (d.get("problem") or "")
                   for d in inf)

    def test_robustness_swings_diagnosed(self):
        rep = root_cause_analysis(
            trials=[_trial(0.7)], baseline=None, best=_best(0.7),
            validation=[], profile_reliability={"level": "ok"},
            input_analyses=[], model_info=None,
            robustness={"mean": 0.6, "std": 0.2, "worst_type": "quiet",
                        "per_type": {"quiet": {"mean": 0.3}}})
        inf = rep["problem_classes"]["inference_parameters"]
        assert any("unevenly" in (d.get("problem") or "") for d in inf)

    def test_diagnoses_carry_confidence_and_disclaimer(self):
        rep = root_cause_analysis(
            trials=[_trial(0.5)], baseline={"evaluation":
                                            {"composite": 0.5}},
            best=_best(0.51), validation=[],
            profile_reliability={"level": "low", "n_samples": 1,
                                 "total_duration_s": 2.0, "notes": [],
                                 "embedding_counts": {}},
            input_analyses=[{"source": "i.wav", "duration_s": 8.0,
                             "loudness": {"snr_proxy_db": 9.0,
                                          "clipping_ratio": 0.0},
                             "vad": {"voiced_ratio": 0.6}}],
            model_info={"path": "/m.pth", "index_path": None},
            embedder_status={"wavlm": False, "ecapa": False,
                             "reliability": "low"})
        for d in rep["diagnoses"]:
            assert 0.0 <= d["confidence"] <= 1.0
            assert d["evidence"]
        assert "never asserted as proven" in rep["disclaimer"]
        assert set(rep["problem_classes"]) == {
            "inference_parameters", "reference_data", "input_audio",
            "model_quality", "evaluation_models"}
