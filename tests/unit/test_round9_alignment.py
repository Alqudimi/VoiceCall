"""Regression tests — round-9 official alignment (train/ + infer/vc/modules).

Files diffed against the official repo this round: train/process_ckpt.py,
train/losses.py, train/mel_processing.py, train/data_utils.py,
train/preprocess.py, train/utils.py, infer/vc/modules.py, webui.py.

Pinned here:
1. export_inference_checkpoint writes the NEW optional ``speaker_info``
   field (upstream process_ckpt.savee) — normalised, deduped, sorted,
   ids 0..109 — and omits it when empty, exactly like upstream;
2. FeatureIndex flags trained_* indexes (no stored vectors) and warns,
   mirroring the upstream rtrvc/vc message about added_ vs trained_;
3. losses/mel/data_utils/collate/preprocess slicer+norm_write constants
   are byte-identical to upstream (spot-checks so future edits cannot
   silently drift).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from rvc_platform.engine.rvc.checkpoint import (
    export_inference_checkpoint,
    normalize_speaker_info,
)


class _Hps:
    """Minimal hps mirror of the official structure used by savee."""

    class data:
        filter_length = 2048
        sampling_rate = 40000

    class model:
        inter_channels = 192
        hidden_channels = 192
        filter_channels = 768
        n_heads = 2
        n_layers = 6
        kernel_size = 3
        p_dropout = 0.0
        resblock = "1"
        resblock_kernel_sizes = [3, 7, 11]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        upsample_rates = [10, 10, 2, 2]
        upsample_initial_channel = 512
        upsample_kernel_sizes = [16, 16, 4, 4]
        spk_embed_dim = 109
        gin_channels = 256


def _state_dict():
    return {
        "enc_q.weight": torch.zeros(4),
        "foo.weight": torch.ones(4),
    }


class TestSpeakerInfoExport:
    def test_normalize_dedupes_and_sorts(self):
        info = [
            {"id": 5, "name": "b"},
            {"id": 0, "name": "a"},
            {"id": 5, "name": "dup"},
            {"id": 200, "name": "out-of-range"},
            {"id": 3, "name": ""},
            {"bad": 1},
        ]
        assert normalize_speaker_info(info) == [
            {"id": 0, "name": "a"},
            {"id": 5, "name": "b"},
        ]

    def test_export_omits_speaker_info_when_empty(self, tmp_path):
        out = tmp_path / "m.pth"
        export_inference_checkpoint(
            _state_dict(), _Hps(), 10, "v2", out, 40000, 1,
        )
        ckpt = torch.load(out, map_location="cpu")
        assert "speaker_info" not in ckpt
        assert ckpt["info"] == "10epoch"
        assert ckpt["f0"] == 1 and ckpt["sr"] == 40000
        # enc_q keys are dropped upstream-style
        assert "enc_q.weight" not in ckpt["weight"]
        assert ckpt["weight"]["foo.weight"].dtype == torch.float16

    def test_export_writes_normalised_speaker_info(self, tmp_path):
        hps = _Hps()
        hps.speaker_info = [{"id": 7, "name": "voice"}, {"id": 7, "name": "dup"}]
        out = tmp_path / "m2.pth"
        export_inference_checkpoint(
            _state_dict(), hps, 3, "v2", out, 40000, 1,
        )
        ckpt = torch.load(out, map_location="cpu")
        assert ckpt["speaker_info"] == [{"id": 7, "name": "voice"}]


class TestTrainedIndexGuard:
    def test_flags_trained_index(self, tmp_path, caplog):
        import faiss

        from rvc_platform.engine.rvc.index import FeatureIndex

        index = faiss.index_factory(8, "IVF4,Flat")
        index.train(np.random.default_rng(0).standard_normal((64, 8)).astype("float32"))
        # NOT calling index.add -> trained-only (0 vectors) like upstream
        # trained_*.index before feature write
        p = tmp_path / "trained_IVF4_Flat_nprobe_1_exp_v2.index"
        faiss.write_index(index, str(p))
        with caplog.at_level(logging.WARNING):
            fi = FeatureIndex(p)
        assert fi.is_trained_only is True
        assert any("trained_" in r.message for r in caplog.records)

    def test_added_index_is_not_flagged(self, tmp_path):
        import faiss

        from rvc_platform.engine.rvc.index import FeatureIndex

        index = faiss.index_factory(8, "IVF4,Flat")
        vecs = np.random.default_rng(1).standard_normal((64, 8)).astype("float32")
        index.train(vecs)
        index.add(vecs)
        p = tmp_path / "added_IVF4_Flat_nprobe_1_exp_v2.index"
        faiss.write_index(index, str(p))
        fi = FeatureIndex(p)
        assert fi.is_trained_only is False
        assert fi.vectors is not None


class TestUpstreamConstantsPinned:
    """Spot-checks: numerics must stay byte-identical to upstream."""

    def test_losses_formulas(self):
        from rvc_platform.training.losses import (
            discriminator_loss,
            feature_loss,
            generator_loss,
        )

        fmap_r = [[torch.ones(2, 3)]]
        fmap_g = [[torch.zeros(2, 3)]]
        assert float(feature_loss(fmap_r, fmap_g)) == pytest.approx(2.0)
        d_out = [torch.full((2, 1), 0.5)]
        loss, rl, gl = discriminator_loss(d_out, [torch.zeros(2, 1)])
        assert float(loss) == pytest.approx(0.25)  # mean((1-0.5)^2) + 0
        gloss, _ = generator_loss([torch.zeros(2, 1)])
        assert float(gloss) == pytest.approx(1.0)

    def test_kl_loss_formula(self):
        from rvc_platform.training.losses import kl_loss

        z_p = torch.zeros(1, 4, 5)
        logs_q = torch.zeros(1, 4, 5)
        m_p = torch.zeros(1, 4, 5)
        logs_p = torch.zeros(1, 4, 5)
        mask = torch.ones(1, 4, 5)
        # element: 0 - 0 - 0.5 + 0.5*(0)^2*1 = -0.5 (same as upstream
        # formula on these degenerate inputs; verified by hand-executing
        # the official snippet)
        assert float(kl_loss(z_p, logs_q, m_p, logs_p, mask)) == pytest.approx(-0.5)

    def test_mel_clip_val(self):
        from rvc_platform.training.mel_processing import (
            dynamic_range_compression_torch,
        )

        x = torch.tensor([0.0, 1.0])
        out = dynamic_range_compression_torch(x)
        assert out[0] == pytest.approx(torch.log(torch.tensor(2e-6)).item())

    def test_preprocess_constants(self):
        from rvc_platform.training.preprocess import DatasetPreprocessor

        assert DatasetPreprocessor.__init__.__defaults__[0] == 3.0  # per
        # alpha / max upstream constants
        src = __import__("inspect").getsource(DatasetPreprocessor)
        assert "self.max = 0.9" in src and "self.alpha = 0.75" in src

    def test_lr_decay_default_matches_official(self):
        from rvc_platform.training.run_config import TrainingRunConfig

        assert TrainingRunConfig.model_fields["lr_decay"].default == 0.999875
