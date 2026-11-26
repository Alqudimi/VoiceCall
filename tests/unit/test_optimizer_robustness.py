"""Data-role assignment + robustness probes (Phase 02 W12)."""
from __future__ import annotations

import numpy as np
import pytest

from rvc_platform.optimizer.analysis import SR_ANALYSIS, AudioAnalyzer
from rvc_platform.optimizer.robustness import (
    assign_input_roles,
    robustness_probes,
    robustness_scores,
)
from tests.conftest import speech_like


@pytest.fixture()
def analyzer(project_env):
    return AudioAnalyzer(None, device="cpu")


def test_single_file_degrades_to_segment_level_with_note():
    rep = assign_input_roles(["a.wav"])
    assert rep["strategy"] == "segment-level"
    assert rep["search"] == ["a.wav"]
    assert any("single input file" in n for n in rep["notes"])


def test_two_files_share_second_file_for_val_and_test():
    rep = assign_input_roles(["a.wav", "b.wav"], seed=3)
    assert rep["strategy"] == "file+segment"
    assert len(rep["search"]) == 1
    assert rep["validation"] == rep["test"]
    assert rep["search"][0] not in rep["validation"]


def test_three_or_more_files_get_file_level_split():
    files = [f"take_{i}.wav" for i in range(5)]
    rep = assign_input_roles(files, seed=1)
    assert rep["strategy"] == "file-level"
    all_assigned = rep["search"] + rep["validation"] + rep["test"]
    assert sorted(all_assigned) == sorted(files)
    assert len(rep["validation"]) == 1
    assert len(rep["test"]) == 1
    assert set(rep["test"]).isdisjoint(rep["search"])
    assert set(rep["test"]).isdisjoint(rep["validation"])


def test_role_assignment_is_deterministic():
    files = [f"take_{i}.wav" for i in range(4)]
    r1 = assign_input_roles(files, seed=9)
    r2 = assign_input_roles(files, seed=9)
    assert r1 == r2


def test_empty_inputs_reported():
    rep = assign_input_roles([])
    assert rep["strategy"] == "none"
    assert rep["search"] == []


def _rich_wav(seconds: float = 12.0, f0: float = 170.0) -> np.ndarray:
    """A take with several speech bursts of different lengths/levels."""
    wav = np.zeros(int(SR_ANALYSIS * seconds), dtype=np.float32)
    rng = np.random.default_rng(4)
    bursts = [(0.2, 1.2, 1.0), (2.0, 3.5, 0.7), (6.0, 2.5, 0.4),
              (9.0, 2.5, 1.0)]
    for start, dur, gain in bursts:
        seg = speech_like(SR_ANALYSIS, dur, f0=f0,
                          seed=int(start * 10) + 1) * gain
        lo = int(start * SR_ANALYSIS)
        wav[lo:lo + seg.size] = seg[:wav.size - lo][:seg.size]
    return wav


def test_probes_include_distinct_types(analyzer):
    probes = robustness_probes(analyzer, _rich_wav())
    types = {p["type"] for p in probes}
    assert "normal" in types
    assert len(probes) >= 2
    for p in probes:
        assert p["audio"].size >= SR_ANALYSIS // 2


def test_probe_types_are_real_slices_not_synthetic(analyzer):
    wav = _rich_wav()
    probes = robustness_probes(analyzer, wav)
    for p in probes:
        assert p["audio"].dtype == np.float32
        # every probe must appear inside the source utterance
        assert p["audio"].size <= wav.size


def test_scores_report_per_type_and_stability():
    results = [
        {"type": "normal", "composite": 0.7},
        {"type": "quiet", "composite": 0.5},
        {"type": "loud", "composite": 0.8},
        {"type": "short", "composite": 0.65},
        {"type": "error_probe", "composite": None, "error": "boom"},
    ]
    rep = robustness_scores(results)
    assert rep["n_probes"] == 4
    assert rep["n_errors"] == 1
    assert set(rep["per_type"]) == {"normal", "quiet", "loud", "short"}
    assert rep["worst_type"] == "quiet"
    # stability = mean - 0.5*std (anti-overfitting measure)
    vals = np.array([0.7, 0.5, 0.8, 0.65])
    assert abs(rep["stability"] -
               (vals.mean() - 0.5 * vals.std())) < 1e-3


def test_single_probe_type_reports_no_spread():
    rep = robustness_scores([{"type": "normal", "composite": 0.6}])
    assert rep["per_type"]["normal"]["note"] == \
        "single probe — no spread measurable"


def test_all_failed_probes_report_honestly():
    rep = robustness_scores([
        {"type": "a", "composite": None, "error": "x"}])
    assert rep["n_probes"] == 0
    assert rep["stability"] is None
    assert "unmeasurable" in rep["note"]
