"""Experiment memory & conversion cache (Phase 02 W14)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from rvc_platform.optimizer.memory import (
    OPTIMIZER_VERSION,
    ExperimentMemory,
    audio_sha256,
    params_key,
)


@pytest.fixture()
def mem(tmp_path):
    return ExperimentMemory(tmp_path / "mem")


def _record(mem, session="s1", ir=0.75, comp=0.7, gates=True,
            model_hash="mh", input_hash="ih", index_hash="x",
            with_payload=False, **over):
    return mem.record_trial(
        model_hash=model_hash, index_hash=index_hash,
        profile_id="m/default", input_hash=input_hash,
        params={"f0_method": "pm", "index_rate": ir, "filter_radius": 0,
                "rms_mix_rate": 0.25, "protect": 0.33, "pitch_shift": 0},
        session_id=session,
        axes={"speaker_similarity": comp, "clarity": comp},
        composite=comp, gates_passed=gates,
        reliability="high", device="cpu",
        output_analysis={"duration_s": 5.0} if with_payload else None,
        similarity_report={"similarities": {"wavlm": 0.5}}
        if with_payload else None,
        **over)


def test_record_and_query_roundtrip(mem):
    _record(mem, ir=0.8, comp=0.8)
    _record(mem, ir=0.5, comp=0.5, session="s2")
    trials = mem.trials_for("mh")
    assert len(trials) == 2
    same_input = mem.trials_for("mh", input_hash="ih")
    assert len(same_input) == 2
    other = mem.trials_for("mh", input_hash="zz")
    assert other == []


def test_best_known_prefers_gate_passing_highest(mem):
    _record(mem, ir=0.9, comp=0.9, gates=False)     # high but failed gates
    _record(mem, ir=0.8, comp=0.8, gates=True)
    best = mem.best_known("mh")
    assert best["params"]["index_rate"] == 0.8
    assert best["composite"] == 0.8


def test_warm_start_dedups_and_orders(mem):
    _record(mem, ir=0.9, comp=0.9, session="s1")
    _record(mem, ir=0.9, comp=0.9, session="s2")     # same params again
    _record(mem, ir=0.7, comp=0.7, session="s3")
    warm = mem.warm_start_params("mh", k=3)
    assert len(warm) == 2
    assert warm[0]["index_rate"] == 0.9


def test_warm_start_excludes_known_bad(mem):
    _record(mem, ir=0.95, comp=0.95, gates=False)
    warm = mem.warm_start_params("mh")
    assert warm == []


def test_conversion_cache_hit_requires_all_hashes(mem):
    _record(mem, ir=0.8, comp=0.8, with_payload=True)
    params = {"f0_method": "pm", "index_rate": 0.8, "filter_radius": 0,
              "rms_mix_rate": 0.25, "protect": 0.33, "pitch_shift": 0}
    hit = mem.lookup_conversion("mh", "x", "ih", params)
    assert hit is not None and hit["cached"]
    assert hit["output_analysis"]["duration_s"] == 5.0
    assert hit["similarity_report"]["similarities"]["wavlm"] == 0.5
    # wrong index hash -> miss
    miss = mem.lookup_conversion("mh", "OTHER", "ih", params)
    assert miss is None
    # wrong params -> miss
    p2 = dict(params, index_rate=0.4)
    assert mem.lookup_conversion("mh", "x", "ih", p2) is None
    # missing payload -> miss
    _record(mem, session="s9", ir=0.3, comp=0.3, with_payload=False)
    p3 = dict(params, index_rate=0.3)
    assert mem.lookup_conversion("mh", "x", "ih", p3) is None


def test_records_without_model_hash_are_never_cache_entries(mem):
    rec = mem.record_trial(
        model_hash=None, index_hash=None, profile_id="p",
        input_hash=None, params={"index_rate": 0.5}, session_id="s0",
        composite=0.5)
    assert rec["cache_key"] is None


def test_stats_and_sessions_summary(mem):
    _record(mem, session="s1", comp=0.7)
    _record(mem, session="s1", comp=0.75)
    _record(mem, session="s2", comp=0.6)
    stats = mem.stats()
    assert stats["n_records"] == 3
    assert stats["n_reusable"] == 3
    assert stats["optimizer_version"] == OPTIMIZER_VERSION
    sessions = mem.sessions_summary()
    assert {s["session_id"] for s in sessions} == {"s1", "s2"}
    s1 = next(s for s in sessions if s["session_id"] == "s1")
    assert s1["n_trials"] == 2 and s1["best_composite"] == 0.75


def test_torn_tail_write_is_skipped(mem, tmp_path):
    _record(mem, comp=0.7)
    with open(mem.log_path, "a", encoding="utf-8") as fh:
        fh.write('{"schema": 2, "broken": ')      # torn write
    stats = mem.stats()
    assert stats["n_records"] == 1                 # bad line ignored


def test_params_key_is_canonical():
    a = {"index_rate": 0.449999, "f0_method": "pm", "protect": 0.33}
    b = {"f0_method": "pm", "protect": 0.33, "index_rate": 0.45}
    assert params_key(a) == params_key(b)


def test_audio_sha256_is_content_based():
    x = np.linspace(-0.5, 0.5, 8000, dtype=np.float32)
    assert audio_sha256(x) == audio_sha256(x.copy())
    assert audio_sha256(x) != audio_sha256(x * 0.5)


def test_persistence_across_instances(mem):
    _record(mem, comp=0.77)
    mem2 = ExperimentMemory(mem.root)
    assert len(mem2.trials_for("mh")) == 1
    assert mem2.best_known("mh")["composite"] == 0.77
