"""Unit tests — batch 10: VITS building blocks (commons / modules /
attentions), best_model tracking + validation splits, and the missing
preprocess helpers (silence/noise-floor heuristics).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch


# --------------------------------------------------------------------------- #
# commons — tensor utilities
# --------------------------------------------------------------------------- #
class TestCommons:
    def test_init_weights(self):
        from rvc_platform.engine.rvc.models.commons import init_weights

        conv = torch.nn.Conv1d(8, 8, 3)
        init_weights(conv)                        # Conv weights -> std 0.01
        assert float(conv.weight.std()) < 0.05
        lin = torch.nn.Linear(8, 8)
        before = lin.weight.clone()
        init_weights(lin)                         # non-Conv -> untouched
        assert torch.equal(lin.weight, before)

    def test_get_padding(self):
        from rvc_platform.engine.rvc.models.commons import get_padding

        assert get_padding(3) == 1
        assert get_padding(3, dilation=2) == 2
        assert get_padding(5, dilation=1) == 2

    def test_kl_divergence_zero_when_identical(self):
        from rvc_platform.engine.rvc.models.commons import kl_divergence

        m = torch.zeros(2, 8, 10)
        logs = torch.zeros(2, 8, 10)
        kl = kl_divergence(m, logs, m, logs)
        assert float(kl.abs().max()) < 1e-6

    def test_kl_divergence_positive_when_different(self):
        from rvc_platform.engine.rvc.models.commons import kl_divergence

        m_p = torch.zeros(1, 8, 10)
        logs_p = torch.zeros(1, 8, 10)
        m_q = torch.ones(1, 8, 10)
        logs_q = torch.zeros(1, 8, 10)
        kl = kl_divergence(m_p, logs_p, m_q, logs_q)
        assert float(kl.mean()) > 0

    def test_rand_gumbel_like(self):
        from rvc_platform.engine.rvc.models.commons import rand_gumbel_like

        x = torch.zeros(3, 4)
        g = rand_gumbel_like(x)
        assert g.shape == x.shape
        assert float(g.std()) > 0.5             # gumbel is wide (std ~1.28)

    def test_slice_segments(self):
        from rvc_platform.engine.rvc.models.commons import slice_segments

        x = torch.arange(20).float().view(1, 1, 20)
        out = slice_segments(x, ids_str=torch.tensor([3]), segment_size=4)
        assert out.shape == (1, 1, 4)
        assert out.flatten().tolist() == [3.0, 4.0, 5.0, 6.0]

    def test_slice_segments2(self):
        from rvc_platform.engine.rvc.models.commons import slice_segments2

        x = torch.arange(20).float().view(1, 20)
        out = slice_segments2(x, ids_str=torch.tensor([5]), segment_size=3)
        assert out.flatten().tolist() == [5.0, 6.0, 7.0]

    def test_rand_slice_segments(self):
        from rvc_platform.engine.rvc.models.commons import rand_slice_segments

        x = torch.randn(2, 8, 100)
        x_lengths = torch.tensor([100, 100])
        out, ids = rand_slice_segments(x, x_lengths, segment_size=16)
        assert out.shape == (2, 8, 16)
        assert ids.shape == (2,)

    def test_timing_signal(self):
        from rvc_platform.engine.rvc.models.commons import (
            add_timing_signal_1d,
            cat_timing_signal_1d,
            get_timing_signal_1d,
        )

        ts = get_timing_signal_1d(50, 64)
        assert ts.shape == (1, 64, 50)
        x = torch.zeros(1, 64, 50)
        added = add_timing_signal_1d(x)
        assert added.shape == x.shape
        assert not torch.equal(added, x)
        cat = cat_timing_signal_1d(x)
        assert cat.shape == (1, 128, 50)        # doubled channels

    def test_subsequent_mask(self):
        from rvc_platform.engine.rvc.models.commons import subsequent_mask

        m = subsequent_mask(4)
        assert m.shape == (1, 1, 4, 4)            # (b, 1, t, t)
        # tril: self + past visible (1), future masked (0)
        assert m[0, 0, 3, 0].item() == 1
        assert m[0, 0, 3, 3].item() == 1
        assert m[0, 0, 0, 3].item() == 0

    def test_fused_add_tanh_sigmoid_multiply(self):
        from rvc_platform.engine.rvc.models.commons import (
            fused_add_tanh_sigmoid_multiply,
        )

        a = torch.randn(1, 8, 10)
        b = torch.randn(1, 8, 10)
        out = fused_add_tanh_sigmoid_multiply(a, b, n_channels=[4])
        assert out.shape == (1, 4, 10)          # half the channels


# --------------------------------------------------------------------------- #
# modules — WaveNet pieces
# --------------------------------------------------------------------------- #
class TestModules:
    def test_layer_norm(self):
        from rvc_platform.engine.rvc.models.modules import LayerNorm

        ln = LayerNorm(channels=16)
        x = torch.randn(2, 16, 30)
        out = ln(x)
        assert out.shape == x.shape
        assert float(out.mean()) == pytest.approx(0.0, abs=1e-4)

    def test_conv_relu_norm(self):
        from rvc_platform.engine.rvc.models.modules import ConvReluNorm

        crn = ConvReluNorm(16, 16, 16, kernel_size=3, n_layers=2, p_dropout=0.0)
        x = torch.randn(1, 16, 30)
        x_mask = torch.ones(1, 1, 30)
        out = crn(x, x_mask)
        assert out.shape == x.shape

    def test_dds_conv(self):
        from rvc_platform.engine.rvc.models.modules import DDSConv

        dds = DDSConv(channels=16, kernel_size=3, n_layers=2)
        x = torch.randn(1, 16, 30)
        x_mask = torch.ones(1, 1, 30)
        out = dds(x, x_mask)
        assert out.shape == x.shape
        # with a conditioning vector g (broadcast over time)
        out_g = dds(x, x_mask, g=torch.randn(1, 16, 1))
        assert out_g.shape == x.shape

    def test_wn_forward_and_remove_weight_norm(self):
        from rvc_platform.engine.rvc.models.modules import WN

        wn = WN(hidden_channels=16, kernel_size=3, dilation_rate=1,
                n_layers=2, gin_channels=0, p_dropout=0.0)
        x = torch.randn(1, 16, 30)
        x_mask = torch.ones(1, 1, 30)
        out = wn(x, x_mask)
        assert out.shape == x.shape
        wn.remove_weight_norm()                 # inference conversion path
        out2 = wn(x, x_mask)
        assert out2.shape == x.shape

    def test_resblock1(self):
        from rvc_platform.engine.rvc.models.modules import ResBlock1

        rb = ResBlock1(channels=8, kernel_size=3, dilation=(1, 3, 5))
        x = torch.randn(1, 8, 64)
        out = rb(x)
        assert out.shape == x.shape
        rb.remove_weight_norm()
        assert rb(x).shape == x.shape

    def test_resblock2(self):
        from rvc_platform.engine.rvc.models.modules import ResBlock2

        rb = ResBlock2(channels=8, kernel_size=3, dilation=(1, 3))
        x = torch.randn(1, 8, 64)
        out = rb(x)
        assert out.shape == x.shape
        rb.remove_weight_norm()
        assert rb(x).shape == x.shape


# --------------------------------------------------------------------------- #
# attentions — encoder / decoder / MHA
# --------------------------------------------------------------------------- #
class TestAttentions:
    def _enc(self):
        from rvc_platform.engine.rvc.models.attentions import Encoder

        return Encoder(hidden_channels=16, filter_channels=32,
                       n_heads=2, n_layers=1, kernel_size=3,
                       p_dropout=0.0)

    def test_encoder(self):
        enc = self._enc()
        x = torch.randn(1, 16, 40)
        x_mask = torch.ones(1, 1, 40)
        out = enc(x, x_mask)
        assert out.shape == x.shape             # returns x only

    def test_encoder_masks_zero_region(self):
        enc = self._enc()
        x = torch.randn(1, 16, 40)
        mask = torch.ones(1, 1, 40)
        mask[:, :, 30:] = 0                     # last 10 frames padded
        out = enc(x, mask)
        assert float(out[:, :, 30:].abs().max()) == 0.0

    def test_decoder(self):
        from rvc_platform.engine.rvc.models.attentions import Decoder

        dec = Decoder(hidden_channels=16, filter_channels=32, n_heads=2,
                      n_layers=1, kernel_size=3, p_dropout=0.0)
        # x and h share the channel count (self-attention over x,
        # cross-attention against the encoder output h)
        z = torch.randn(1, 16, 30)
        z_mask = torch.ones(1, 1, 30)
        h = torch.randn(1, 16, 30)
        h_mask = torch.ones(1, 1, 30)
        out = dec(z, z_mask, h, h_mask)
        assert out.shape == z.shape

    def test_multi_head_attention(self):
        from rvc_platform.engine.rvc.models.attentions import \
            MultiHeadAttention

        mha = MultiHeadAttention(channels=16, out_channels=16, n_heads=2)
        x = torch.randn(1, 16, 20)
        c = torch.randn(1, 16, 20)
        out = mha(x, c)                           # self-attn: c == x
        assert out.shape == x.shape
        out_qk = mha(x, c, attn_mask=torch.ones(1, 1, 20, 20))
        assert out_qk.shape == x.shape


# --------------------------------------------------------------------------- #
# best_model — tracking + splits
# --------------------------------------------------------------------------- #
class TestBestModelTracker:
    def test_mode_validation(self, tmp_path):
        from rvc_platform.training.best_model import BestModelTracker

        with pytest.raises(ValueError):
            BestModelTracker(tmp_path / "b.json", mode="sideways")

    def test_min_mode_tracking(self, tmp_path):
        from rvc_platform.training.best_model import BestModelInfo, \
            BestModelTracker

        tr = BestModelTracker(tmp_path / "best.json", metric="val_mel",
                              mode="min")
        assert tr.load() is False               # nothing on disk yet
        first = tr.update({"val_mel": 0.5}, epoch=1, global_step=10)
        assert first is not None and first.score == 0.5
        assert tr.update({"val_mel": 0.6}, 2, 20) is None   # worse -> no
        better = tr.update({"val_mel": 0.4}, 3, 30)
        assert better is not None and better.epoch == 3
        # NaN / missing / non-numeric are all ignored
        assert tr.update({"val_mel": float("nan")}, 4, 40) is None
        assert tr.update({"other": 1.0}, 5, 50) is None
        assert tr.update({"val_mel": "junk"}, 6, 60) is None
        # persist + restore: the worker saves the state file itself
        state = tmp_path / "best.json"
        state.write_text(json.dumps(tr.best.to_dict()), encoding="utf-8")
        tr2 = BestModelTracker(state, metric="val_mel")
        assert tr2.load() is True
        assert tr2.best.score == 0.4
        back = BestModelInfo.from_dict(tr2.best.to_dict())
        assert back.score == 0.4 and back.epoch == 3

    def test_max_mode_tracking(self, tmp_path):
        from rvc_platform.training.best_model import BestModelTracker

        tr = BestModelTracker(tmp_path / "b.json", metric="sim", mode="max")
        assert tr.update({"sim": 0.2}, 1, 10) is not None
        assert tr.update({"sim": 0.1}, 2, 20) is None
        assert tr.update({"sim": 0.9}, 3, 30) is not None
        tr.reset()
        assert tr.best is None

    def test_load_corrupt_file(self, tmp_path):
        from rvc_platform.training.best_model import BestModelTracker

        p = tmp_path / "best.json"
        p.write_text("{corrupt", encoding="utf-8")
        tr = BestModelTracker(p, metric="val_mel")
        assert tr.load() is False
        assert tr.best is None

    def test_load_wrong_schema(self, tmp_path):
        from rvc_platform.training.best_model import BestModelTracker

        p = tmp_path / "best.json"
        p.write_text(json.dumps({"nonsense": True}), encoding="utf-8")
        tr = BestModelTracker(p, metric="val_mel")
        assert tr.load() is False

    def test_best_model_info_roundtrip(self):
        from rvc_platform.training.best_model import BestModelInfo

        info = BestModelInfo(metric="val_mel", score=0.42, epoch=7,
                             global_step=700, improved_at=1234.5)
        d = info.to_dict()
        assert d["metric"] == "val_mel"
        assert d["epoch"] == 7
        back = BestModelInfo.from_dict(d)
        assert back.score == 0.42 and back.epoch == 7


class TestDeriveValidationSplit:
    def _filelist(self, tmp_path, n=20):
        p = tmp_path / "filelist.txt"
        p.write_text("\n".join(f"line{i}" for i in range(n)) + "\n",
                     encoding="utf-8")
        return p

    def test_no_split_when_zero(self, tmp_path):
        from rvc_platform.training.best_model import derive_validation_split

        fl = self._filelist(tmp_path)
        assert derive_validation_split(fl, 0.0) is None

    def test_no_split_when_missing_file(self, tmp_path):
        from rvc_platform.training.best_model import derive_validation_split

        assert derive_validation_split(tmp_path / "ghost.txt", 0.1) is None

    def test_no_split_when_too_few_lines(self, tmp_path):
        from rvc_platform.training.best_model import derive_validation_split

        fl = self._filelist(tmp_path, n=1)
        assert derive_validation_split(fl, 0.2) is None

    def test_split_writes_and_pins(self, tmp_path):
        from rvc_platform.training.best_model import derive_validation_split

        fl = self._filelist(tmp_path, n=20)
        res = derive_validation_split(fl, 0.2)
        assert res is not None and res.rederived is True
        assert res.n_train + res.n_val == 20
        assert res.n_val == 4
        assert res.train_path.is_file() and res.val_path.is_file()
        # deterministic: same seed -> same membership
        val_lines = res.val_path.read_text(encoding="utf-8").splitlines()
        res2 = derive_validation_split(fl, 0.2)     # pinned -> reused
        assert res2.rederived is False
        assert res2.val_path.read_text(
            encoding="utf-8").splitlines() == val_lines

    def test_split_rederives_when_filelist_changes(self, tmp_path):
        from rvc_platform.training.best_model import derive_validation_split

        fl = self._filelist(tmp_path, n=20)
        derive_validation_split(fl, 0.2)
        fl.write_text("\n".join(f"changed{i}" for i in range(20)) + "\n",
                      encoding="utf-8")
        res = derive_validation_split(fl, 0.2)
        assert res.rederived is True

    def test_split_rederives_when_ratio_changes(self, tmp_path):
        from rvc_platform.training.best_model import derive_validation_split

        fl = self._filelist(tmp_path, n=20)
        derive_validation_split(fl, 0.2)
        res = derive_validation_split(fl, 0.3)
        assert res.rederived is True and res.n_val == 6

    def test_split_corrupt_meta_rederives(self, tmp_path):
        from rvc_platform.training.best_model import derive_validation_split

        fl = self._filelist(tmp_path, n=10)
        derive_validation_split(fl, 0.2)
        # the actual metadata filename is validation_split.json
        (tmp_path / "validation_split.json").write_text("{bad",
                                                        encoding="utf-8")
        res = derive_validation_split(fl, 0.2)
        assert res.rederived is True


# --------------------------------------------------------------------------- #
# preprocess — silence / noise heuristics
# --------------------------------------------------------------------------- #
class TestPreprocessHeuristics:
    def test_silence_ratio(self):
        from rvc_platform.training.preprocess import _silence_ratio

        loud = np.full(16000, 0.5, np.float32)
        quiet = np.full(16000, 1e-5, np.float32)
        assert _silence_ratio(loud) == pytest.approx(0.0)
        assert _silence_ratio(quiet) == pytest.approx(1.0)

    def test_noise_floor_db(self):
        from rvc_platform.training.preprocess import _noise_floor_db

        loud = np.full(16000, 0.5, np.float32)
        floor = _noise_floor_db(loud)
        assert floor > -20                      # loud signal -> high floor
        quiet = np.full(16000, 1e-5, np.float32)
        assert _noise_floor_db(quiet) < -80

    def test_file_md5(self, tmp_path):
        from rvc_platform.training.preprocess import file_md5

        p = tmp_path / "x.bin"
        p.write_bytes(b"hello")
        h1 = file_md5(p)
        assert h1 == file_md5(p) and len(h1) == 32
        p.write_bytes(b"world")
        assert file_md5(p) != h1
