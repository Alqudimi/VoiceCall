"""Regression tests: training templates must match official RVC configs.

Round-7 audit found our per-rate templates mixed v1 and v2 generator shapes
(official configs/v1/* and configs/v2/* DIFFER for the same rate:
v1_32k upsample_rates=[10,4,2,2,2] vs v2_32k [10,8,2,2]; v1_48k additionally
segment_size=11520 vs v2 17280). A checkpoint trained with the wrong shape
cannot load in official tooling. The fixes:
* per-(version, rate) templates v1_32k/40k/48k + v2_32k/48k (official files
  verbatim);
* plain <sr>.json kept as the v2 copy for backwards compatibility;
* 40k+v2 rejected (no official v2 40k exists).
The expected values below are pinned from the official repository files.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rvc_platform.core.errors import ConfigurationError
from rvc_platform.training.run_config import load_template

TPL = Path(__file__).resolve().parents[2] / "src" / "rvc_platform" / "training" / "templates"

# pinned from official configs/v1/*.json and configs/v2/*.json (main branch)
OFFICIAL_SHAPES = {
    ("v1", "32k"): {"upsample_rates": [10, 4, 2, 2, 2],
                    "upsample_kernel_sizes": [16, 16, 4, 4, 4]},
    ("v1", "40k"): {"upsample_rates": [10, 10, 2, 2],
                    "upsample_kernel_sizes": [16, 16, 4, 4]},
    ("v1", "48k"): {"upsample_rates": [10, 6, 2, 2, 2],
                    "upsample_kernel_sizes": [16, 16, 4, 4, 4]},
    ("v2", "32k"): {"upsample_rates": [10, 8, 2, 2],
                    "upsample_kernel_sizes": [20, 16, 4, 4]},
    ("v2", "48k"): {"upsample_rates": [12, 10, 2, 2],
                    "upsample_kernel_sizes": [24, 20, 4, 4]},
}
OFFICIAL_SEGMENT = {("v1", "48k"): 11520, ("v2", "48k"): 17280,
                    ("v1", "40k"): 12800, ("v2", "32k"): 12800,
                    ("v1", "32k"): 12800}
OFFICIAL_TRAIN = {  # shared constants across all five official configs
    "learning_rate": 1e-4, "betas": [0.8, 0.99], "eps": 1e-9,
    "lr_decay": 0.999875, "c_kl": 1.0, "c_mel": 45, "seed": 1234,
}


class TestTemplateShapes:
    @pytest.mark.parametrize("ver,sr", sorted(OFFICIAL_SHAPES))
    def test_generator_shape_matches_official(self, ver, sr):
        t = load_template(sr, ver)
        m = t["model"]
        assert m["upsample_rates"] == OFFICIAL_SHAPES[(ver, sr)]["upsample_rates"]
        assert m["upsample_kernel_sizes"] == OFFICIAL_SHAPES[(ver, sr)]["upsample_kernel_sizes"]

    @pytest.mark.parametrize("ver,sr", sorted(OFFICIAL_SHAPES))
    def test_train_constants_match_official(self, ver, sr):
        t = load_template(sr, ver)["train"]
        for k, v in OFFICIAL_TRAIN.items():
            assert t[k] == v, (ver, sr, k)

    @pytest.mark.parametrize("ver,sr", sorted(OFFICIAL_SHAPES))
    def test_segment_size_matches_official(self, ver, sr):
        t = load_template(sr, ver)["train"]
        assert t["segment_size"] == OFFICIAL_SEGMENT[(ver, sr)]

    def test_v1_and_v2_differ_for_same_rate(self):
        """The round-7 bug: one shared template cannot serve both versions."""
        a = load_template("32k", "v1")["model"]
        b = load_template("32k", "v2")["model"]
        assert a["upsample_rates"] != b["upsample_rates"], (
            "v1/v2 official generator shapes differ — shared template = bug"
        )

    def test_plain_templates_are_v2(self):
        """Backwards-compatible plain names must resolve to official v2."""
        for sr in ("32k", "48k"):
            plain = load_template(sr)
            v2 = load_template(sr, "v2")
            assert plain == v2


class TestV2HasNo40k:
    def test_v2_40k_rejected(self):
        with pytest.raises(ConfigurationError, match="40k"):
            load_template("40k", "v2")

    def test_v1_40k_allowed(self):
        t = load_template("40k", "v1")
        assert t["model"]["upsample_rates"] == [10, 10, 2, 2]


class TestTemplateFiles:
    def test_no_extra_non_official_v2_40k_template(self):
        assert not (TPL / "v2_40k.json").exists(), (
            "v2 40k is not official — an invented template invites "
            "incompatible checkpoints"
        )

    def test_official_template_files_exist_verbatim(self):
        for name in ("v1_32k.json", "v1_40k.json", "v1_48k.json",
                     "v2_32k.json", "v2_48k.json"):
            assert (TPL / name).is_file(), name

    def test_hop_win_consistent_with_rate(self):
        WIN = {"32k": 1024, "40k": 2048, "48k": 2048}  # official win_length
        for ver, sr in sorted(OFFICIAL_SHAPES):
            d = load_template(sr, ver)["data"]
            assert d["hop_length"] == SR2HOP[sr]
            assert d["win_length"] == WIN[sr]


SR2HOP = {"32k": 320, "40k": 400, "48k": 480}
