"""Blind A/B preferences + preference learning (Phase 02 W16)."""
from __future__ import annotations

import random

import pytest

from rvc_platform.optimizer.preferences import (
    PREFERENCE_BOOST_MAX,
    PreferenceStore,
)


@pytest.fixture()
def store(tmp_path):
    return PreferenceStore(tmp_path / "prefs")


def _cand(number, ir):
    return {"number": number, "audio": f"cand_{number}.wav",
            "params": {"index_rate": ir, "protect": 0.33,
                       "rms_mix_rate": 0.25, "pitch_shift": 0,
                       "f0_method": "pm", "filter_radius": 0},
            "axes": {"speaker_similarity": 0.5 + ir / 10},
            "label": f"trial #{number}"}


def test_pair_labels_are_randomised_blind(store):
    """Across many pairs both label orders must occur (blind protocol)."""
    seen = set()
    for i in range(40):
        pair = store.create_pair("s1", _cand(1, 0.5), _cand(2, 0.9),
                                 rng=random.Random(i))
        assert pair["chosen"] is None
        # candidate with number=1 ends up on side a or b
        side = ("a" if pair["candidates"]["a"]["number"] == 1 else "b")
        seen.add(side)
    assert seen == {"a", "b"}


def test_choice_reveals_parameters(store):
    pair = store.create_pair("s1", _cand(1, 0.5), _cand(2, 0.9))
    revealed = store.record_choice("s1", pair["pair_id"], "a")
    chosen_params = revealed["candidates"]["a"]["params"]
    assert chosen_params["index_rate"] in (0.5, 0.9)
    assert revealed["chosen"] == "a"


def test_one_choice_per_pair_enforced(store):
    pair = store.create_pair("s1", _cand(1, 0.5), _cand(2, 0.9))
    store.record_choice("s1", pair["pair_id"], "a")
    with pytest.raises(ValueError, match="already answered"):
        store.record_choice("s1", pair["pair_id"], "b")


def test_invalid_choice_rejected(store):
    pair = store.create_pair("s1", _cand(1, 0.5), _cand(2, 0.9))
    with pytest.raises(ValueError):
        store.record_choice("s1", pair["pair_id"], "c")


def test_unknown_pair_rejected(store):
    store.create_pair("s1", _cand(1, 0.5), _cand(2, 0.9))
    with pytest.raises(ValueError):
        store.record_choice("s1", "pair_unknown", "a")


def test_preference_summary_aggregates_winning_parameters(store):
    for i, chosen_ir in enumerate([0.8, 0.85, 0.9]):
        pair = store.create_pair("s1", _cand(1, chosen_ir),
                                 _cand(2, 0.2), rng=random.Random(i))
        side = ("a" if pair["candidates"]["a"]["number"] == 1 else "b")
        store.record_choice("s1", pair["pair_id"], side)
    summary = store.preference_summary("mymodel")
    assert summary["n_preferences"] == 3
    view = summary["parameter_views"]["index_rate"]
    assert view["preferred_mean"] == 0.85


def test_boost_requires_minimum_evidence(store):
    # zero preferences -> no boost
    assert store.preference_boost("m", {"index_rate": 0.8}) == 0.0
    # 1-2 preferences -> still no boost (documented minimum)
    for i, ir in enumerate([0.8, 0.2]):
        pair = store.create_pair("s1", _cand(1, ir), _cand(2, 0.5),
                                 rng=random.Random(i))
        side = ("a" if pair["candidates"]["a"]["number"] == 1 else "b")
        store.record_choice("s1", pair["pair_id"], side)
    assert store.preference_boost("m", {"index_rate": 0.8}) == 0.0


def test_boost_is_bounded_and_prefers_chosen_neighborhoods(store):
    for i in range(5):
        pair = store.create_pair("s1", _cand(1, 0.85), _cand(2, 0.2),
                                 rng=random.Random(i))
        side = ("a" if pair["candidates"]["a"]["number"] == 1 else "b")
        store.record_choice("s1", pair["pair_id"], side)
    near = store.preference_boost("m", {"index_rate": 0.85,
                                        "protect": 0.33,
                                        "rms_mix_rate": 0.25,
                                        "pitch_shift": 0,
                                        "f0_method": "pm",
                                        "filter_radius": 0})
    far = store.preference_boost("m", {"index_rate": 0.1,
                                       "protect": 0.33,
                                       "rms_mix_rate": 0.25,
                                       "pitch_shift": 0,
                                       "f0_method": "pm",
                                       "filter_radius": 0})
    assert near > far >= 0.0
    assert near <= PREFERENCE_BOOST_MAX


def test_affinity_is_zero_without_preferences(store):
    assert store.preference_affinity("m", {"index_rate": 0.5}) == 0.0
