"""Unit tests — batch 9: engine internals.

RVCInferenceEngine (with a synthetic random-weight model + fake HuBERT),
RealTimeRVC, FeatureIndex (real faiss round trip), hubert helpers,
f0 utilities, change_rms, and the generator/discriminator forward paths.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import SYNTH_CONFIG_40K


def _speech(sr=16_000, seconds=2.0, f0=150.0, seed=9):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    sig = (0.4 * np.sin(2 * np.pi * f0 * t)
           + 0.2 * np.sin(2 * np.pi * 2 * f0 * t)
           + 0.005 * rng.standard_normal(len(t)))
    return sig.astype(np.float32)


# --------------------------------------------------------------------------- #
# f0 utilities
# --------------------------------------------------------------------------- #
class TestF0Utils:
    def test_interp_unvoiced(self):
        from rvc_platform.engine.rvc.f0 import interp_unvoiced

        f0 = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 200.0, 0.0], np.float32)
        out = interp_unvoiced(f0)
        assert (out > 0).all()                    # gaps filled
        # leading zeros get the first voiced value
        assert out[0] == pytest.approx(100.0)

    def test_median_filter_f0(self):
        from rvc_platform.engine.rvc.f0 import median_filter_f0

        f0 = np.array([100, 100, 100, 500, 100, 100], np.float32)
        out = median_filter_f0(f0, 3)
        # the spike is smoothed by the median of its neighbourhood
        assert float(np.max(np.abs(out - 100))) < 400

    def test_median_filter_radius_zero_passthrough(self):
        from rvc_platform.engine.rvc.f0 import median_filter_f0

        f0 = np.array([100.0, 300.0, 100.0], np.float32)
        assert np.array_equal(median_filter_f0(f0, 0), f0)

    def test_shift_f0(self):
        from rvc_platform.engine.rvc.f0 import shift_f0

        f0 = np.array([100.0, 200.0, 0.0], np.float32)
        up = shift_f0(f0, 12)                      # +1 octave
        assert up[0] == pytest.approx(200.0)
        assert up[1] == pytest.approx(400.0)
        assert up[2] == 0.0                        # unvoiced stays zero
        down = shift_f0(f0, -12)
        assert down[0] == pytest.approx(50.0)

    def test_f0_to_coarse(self):
        from rvc_platform.engine.rvc.f0 import f0_to_coarse

        coarse, fine = f0_to_coarse(np.array([100.0, 200.0], np.float32))
        assert coarse.dtype == torch.long or coarse[0] >= 1
        assert 1 <= int(np.min(coarse)) and int(np.max(coarse)) <= 255
        assert np.allclose(fine, [100.0, 200.0], atol=1e-4)

    def test_get_f0_extractor_factory(self, project_env):
        from rvc_platform.engine.rvc.f0 import get_f0_extractor

        ex = get_f0_extractor("pm", "cpu", False)
        assert ex.name == "pm"
        f0 = ex.extract(_speech())
        assert f0.shape[0] >= 100 and (f0 > 0).all()

    def test_get_f0_extractor_unknown_raises(self, project_env):
        from rvc_platform.engine.rvc.f0 import get_f0_extractor

        with pytest.raises(Exception):
            get_f0_extractor("nonexistent", "cpu", False)

    def test_rmvpe_extractor_needs_asset(self, project_env):
        """rmvpe requires assets/rmvpe.pt — absent in unit envs -> the
        clear AssetMissingError pointing at doctor --fix."""
        from rvc_platform.core.errors import AssetMissingError
        from rvc_platform.engine.rvc.f0 import get_f0_extractor
        from rvc_platform.infrastructure.paths import get_paths

        if get_paths().rmvpe.is_file():
            pytest.skip("rmvpe.pt present — asset branch not reachable")
        with pytest.raises(AssetMissingError):
            get_f0_extractor("rmvpe", "cpu", False)


# --------------------------------------------------------------------------- #
# change_rms
# --------------------------------------------------------------------------- #
class TestChangeRms:
    def _env(self, audio, sr=16_000):
        import librosa

        return np.abs(librosa.feature.rms(
            y=audio, frame_length=sr // 2 * 2, hop_length=sr // 2)[0])

    def _change(self, src, dst, rate, sr=16_000):
        from rvc_platform.engine.rvc.inference import change_rms

        return change_rms(src, sr, dst.copy(), sr, rate)

    def _signals(self):
        sr = 16_000
        t = np.arange(sr * 3) / sr
        src = (0.5 * (1 + np.sin(2 * np.pi * 0.5 * t))
               * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        dst = (0.5 * (1 + np.sin(2 * np.pi * 0.3 * t + 2))
               * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        return src, dst

    def test_rate_one_is_identity(self):
        src, dst = self._signals()
        out = self._change(src, dst, rate=1.0)
        assert np.allclose(out, dst, atol=1e-4)

    def test_rate_zero_adopts_source_envelope(self):
        src, dst = self._signals()
        out = self._change(src, dst, rate=0.0)
        assert not np.allclose(out, dst)
        # output envelope now tracks the SOURCE envelope
        assert np.corrcoef(self._env(out), self._env(src))[0, 1] > 0.8

    def test_mid_rate_blends(self):
        src, dst = self._signals()
        out = self._change(src, dst, rate=0.5)
        assert not np.allclose(out, dst)
        # between the two envelopes
        c_src = np.corrcoef(self._env(out), self._env(src))[0, 1]
        c_dst = np.corrcoef(self._env(out), self._env(dst))[0, 1]
        assert c_src > 0.5 and c_dst > 0.5


# --------------------------------------------------------------------------- #
# FeatureIndex — real faiss round trip
# --------------------------------------------------------------------------- #
class TestFeatureIndex:
    def _build(self, tmp_path, n=32, dim=16, name="added_IVF4_test.index"):
        import faiss

        vecs = np.random.default_rng(3).standard_normal((n, dim))
        vecs = vecs.astype("float32")
        quant = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quant, dim, 4)
        index.train(vecs)
        index.add(vecs)
        p = tmp_path / name
        faiss.write_index(index, str(p))
        return p, vecs

    def test_roundtrip(self, tmp_path):
        from rvc_platform.engine.rvc.index import FeatureIndex

        p, vecs = self._build(tmp_path)
        fi = FeatureIndex(p)
        assert fi.size == 32
        assert fi.dim == 16
        # blend on a NON-indexed query vector differs from the query
        out = fi.blend(vecs[:4].copy(), index_rate=0.0)
        assert np.allclose(out, vecs[:4], atol=1e-5)
        # blend a vector far outside the stored set
        far = np.full((4, 16), 50.0, dtype=np.float32)
        out1 = fi.blend(far.copy(), index_rate=1.0)
        assert out1.shape == (4, 16)
        # result is pulled toward stored vectors, not equal to the query
        assert float(np.abs(out1 - far).max()) > 1.0

    def test_missing_file_raises(self, tmp_path):
        from rvc_platform.core.errors import CheckpointError
        from rvc_platform.engine.rvc.index import FeatureIndex

        with pytest.raises(CheckpointError):
            FeatureIndex(tmp_path / "nope.index")

    def test_newest_added_index(self, tmp_path):
        from rvc_platform.engine.rvc.index import newest_added_index

        assert newest_added_index(tmp_path, "x", "v2") is None
        import os

        p1 = tmp_path / "added_IVF4_Flat_nprobe_1_x_v2.index"
        p1.write_bytes(b"a" * 1024)
        # the "(1)" name does NOT match the glob pattern (parentheses are
        # literal, 'nprobe_1_' requires digits there) — only the canonical
        # name matches; verify the newest WINS by touching two valid names
        p2 = tmp_path / "added_IVF4_Flat_nprobe_2_x_v2.index"
        p2.write_bytes(b"b" * 1024)
        os.utime(p1, (0, 0))              # force p1 older
        assert newest_added_index(tmp_path, "x", "v2") == p2

    def test_build_index_from_feature_dir(self, tmp_path):
        """build_index(feature_dir, ...) reads *.npy feature files from a
        3_feature768-style directory (768-D for v2 — the dims are fixed by
        the version, not read from the files)."""
        from rvc_platform.engine.rvc.index import build_index

        feat_dir = tmp_path / "3_feature768"
        feat_dir.mkdir()
        rng = np.random.default_rng(5)
        for i in range(3):
            np.save(feat_dir / f"c{i}.npy",
                    rng.standard_normal((40, 768)).astype("float32"))
        out = build_index(feat_dir, tmp_path, "spk", "v2")
        assert Path(out).is_file()
        assert "spk" in Path(out).name and "v2" in Path(out).name
        # re-build produces a second index (different IVF count or same name
        # overwritten) — the function returns a valid path either way
        out2 = build_index(feat_dir, tmp_path, "spk", "v2")
        assert Path(out2).is_file()

    def test_build_index_empty_dir_raises(self, tmp_path):
        from rvc_platform.core.errors import CheckpointError
        from rvc_platform.engine.rvc.index import build_index

        empty = tmp_path / "3_feature768"
        empty.mkdir()
        with pytest.raises(CheckpointError):
            build_index(empty, tmp_path, "spk", "v2")


# --------------------------------------------------------------------------- #
# hubert helpers (no 374 MB asset — stub the transformer)
# --------------------------------------------------------------------------- #
class TestHubertHelpers:
    def test_hubert_model_path(self, project_env):
        from rvc_platform.engine.rvc.hubert import _hubert_model_path
        from rvc_platform.infrastructure.paths import get_paths

        assert _hubert_model_path() == get_paths().hubert

    def test_hubert_requires_normalization_flag(self, project_env):
        """Without the 374 MB asset AutoFeatureExtractor raises (OSError /
        HFValidationError) — the caller treats that as normalize=True."""
        from rvc_platform.infrastructure.paths import get_paths

        if (get_paths().hubert / "config.json").is_file():
            from rvc_platform.engine.rvc.hubert import \
                hubert_audio_requires_normalization

            assert isinstance(
                hubert_audio_requires_normalization(), bool)
        else:
            from rvc_platform.engine.rvc.hubert import \
                hubert_audio_requires_normalization

            with pytest.raises((OSError, ValueError)):
                hubert_audio_requires_normalization()

    def test_load_hubert_missing_asset_raises(self, project_env):
        from rvc_platform.infrastructure.paths import get_paths

        if (get_paths().hubert / "config.json").is_file():
            pytest.skip("hubert asset present")
        with pytest.raises((FileNotFoundError, OSError)):
            from rvc_platform.engine.rvc.hubert import load_hubert_model

            load_hubert_model("cpu", False)

    def test_extract_hubert_features_shapes(self, monkeypatch):
        """With a fake transformers-style model the extractor runs its full
        reshape/layer-selection path (v1 -> final_proj of layer 9 = 256-D;
        v2 -> last_hidden_state = 768-D)."""
        from rvc_platform.engine.rvc import hubert as hb

        class FakeOutput:
            def __init__(self, hs, last):
                self.hidden_states = hs
                self.last_hidden_state = last

        class FakeModel(torch.nn.Module):
            device = torch.device("cpu")

            def __init__(self):
                super().__init__()
                self.final_proj = torch.nn.Linear(1024, 256)

            def forward(self, input_values, attention_mask=None,
                        output_hidden_states=False, return_dict=True,
                        **kw):                       # transformers kwargs
                n = input_values.shape[-1]
                t = max(1, n // 320)
                hs = tuple(torch.randn(1, t, 1024)
                           for _ in range(13))
                return FakeOutput(hs, hs[-1])

        fake = FakeModel()
        src = torch.zeros(1, 16_000)
        v2 = hb.extract_hubert_features(fake, src, "v2")
        assert v2.shape[-1] == 1024          # v2: last_hidden_state passthrough
        v1 = hb.extract_hubert_features(fake, src, "v1")
        assert v1.shape[-1] == 256           # v1: final_proj(hidden_states[9])

    def test_device_type_normalisation(self):
        from rvc_platform.engine.rvc.hubert import _device_type

        assert _device_type("cpu") == "cpu"
        assert _device_type("cuda:0") == "cuda"


# --------------------------------------------------------------------------- #
# models: generator + discriminator forwards (synthetic weights)
# --------------------------------------------------------------------------- #
class TestModelForwards:
    def test_multi_period_discriminator(self):
        from rvc_platform.engine.rvc.models.models import \
            MultiPeriodDiscriminator

        d = MultiPeriodDiscriminator(use_spectral_norm=False)
        y = torch.randn(1, 1, 8192)
        y_hat = torch.randn(1, 1, 8192)
        out = d(y, y_hat)
        # 4 lists (real/fake feats + real/fake maps), one entry per period
        assert len(out) == 4
        assert all(len(lst) == 7 for lst in out)

    def test_posterior_encoder(self):
        from rvc_platform.engine.rvc.models.models import PosteriorEncoder

        enc = PosteriorEncoder(1025, 192, 8, 3, 1, 2)
        x = torch.randn(1, 1025, 100)
        z, m, logs, x_mask = enc(x, torch.tensor([100]))
        assert z.shape[1] == 192
        assert x_mask.shape == (1, 1, 100)

    def test_generator_infer_small(self, rvc_hparams_40k):
        """The real v2 synthesizer with random weights — infer() path (the
        conversion entry point; forward() is training-only)."""
        from rvc_platform.engine.rvc.models.models import \
            SynthesizerTrnMs768NSFsid

        net_g = SynthesizerTrnMs768NSFsid(
            *SYNTH_CONFIG_40K[:17], is_half=False, sr=48_000)
        net_g.eval()
        with torch.no_grad():
            phone = torch.randn(1, 50, 768)
            phone_lengths = torch.tensor([50])
            pitch = torch.randint(1, 255, (1, 50))
            pitchf = torch.rand(1, 50) * 200 + 100
            sid = torch.zeros(1, dtype=torch.long)
            out, x_mask, (z, z_p, m_p, logs_p) = net_g.infer(
                phone, phone_lengths, pitch, pitchf.float(), sid)
        assert out.shape[-1] > 0 and out.shape[0] == 1

    def test_residual_coupling_layer(self):
        from rvc_platform.engine.rvc.models.modules import \
            ResidualCouplingLayer

        rcl = ResidualCouplingLayer(channels=8, hidden_channels=16,
                                    kernel_size=3, dilation_rate=1,
                                    n_layers=2, mean_only=False)
        x = torch.randn(1, 8, 40)
        x_mask = torch.ones(1, 1, 40)
        y, logdet = rcl(x, x_mask)               # forward returns a 2-tuple
        assert y.shape == x.shape
        assert logdet.shape == (1,)
        back, _ = rcl(y, x_mask, reverse=True)
        assert (back - x).abs().max() < 1e-4


# --------------------------------------------------------------------------- #
# RVCInferenceEngine — with synthetic model + fake hubert
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic_model(rvc_hparams_40k, tmp_path_factory):
    from rvc_platform.engine.rvc.checkpoint import export_inference_checkpoint
    from rvc_platform.engine.rvc.models.models import \
        SynthesizerTrnMs768NSFsid

    root = tmp_path_factory.mktemp("engine9")
    net_g = SynthesizerTrnMs768NSFsid(
        *SYNTH_CONFIG_40K[:17], is_half=False, sr=48_000)
    out = root / "synth_v2_48k.pth"
    export_inference_checkpoint(net_g.state_dict(), rvc_hparams_40k, 3,
                                "v2", out, 48_000, 1)
    return out


@pytest.fixture()
def engine(project_env, monkeypatch, synthetic_model):
    """Real engine with: pm f0 (asset-free) + fake HuBERT features."""
    from rvc_platform.engine.rvc.inference import RVCInferenceEngine
    from rvc_platform.utils.device import DeviceProfile

    profile = DeviceProfile(device="cpu", is_half=False, gpu_name=None,
                              gpu_memory_gb=0.0, sm=0.0)
    from rvc_platform.infrastructure.audio.pipeline import \
        AudioProcessingEngine

    eng = RVCInferenceEngine(profile, AudioProcessingEngine(ffmpeg=False))
    # inject a fake hubert so _ensure_hubert skips the 374 MB asset check
    eng._hubert = object()
    eng.load_model(synthetic_model, None)

    # fake hubert: deterministic features, correct time grid
    def fake_extract(model, source, version, padding_mask=None):
        n = source.shape[-1]
        t = max(1, n // 320)
        feats = torch.zeros(1, t, 768)
        feats[0, :, 0] = torch.linspace(0, 1, t)
        return feats

    import rvc_platform.engine.rvc.inference as inf
    monkeypatch.setattr(inf, "extract_hubert_features", fake_extract)
    # pm extractor is dependency-free — keep the real one
    return eng


class TestInferenceEngine:
    def test_load_model_metadata(self, engine, synthetic_model):
        info = engine.loaded_model
        assert info is not None
        assert info.sample_rate == 48_000
        assert info.f0 is True
        assert info.version == "v2"

    def test_status_lifecycle(self, engine):
        assert engine.status == "ready"
        engine.unload()
        assert engine.status == "unavailable"
        from rvc_platform.core.errors import ConversionError

        with pytest.raises(ConversionError):
            engine._require_model()

    def test_get_f0(self, engine):
        audio = _speech()
        coarse, fine = engine.get_f0(audio, audio.shape[0] // 320, 0,
                                     "pm", 3)
        assert coarse.shape[0] == audio.shape[0] // 320
        assert 1 <= int(coarse.min()) and int(coarse.max()) <= 255

    def test_convert_file_end_to_end(self, engine, project_env, tmp_path):
        import soundfile as sf

        from rvc_platform.core.entities import ConversionSettings

        src = tmp_path / "in.wav"
        sf.write(str(src), _speech(seconds=1.5), 16_000, subtype="PCM_16")
        out = tmp_path / "out.wav"
        settings = ConversionSettings(pitch_shift=0, f0_method="pm")
        result = engine.convert_file(src, out, settings)
        assert result.output_path.is_file()
        assert result.duration_seconds == pytest.approx(1.5, abs=0.2)
        assert result.processing_seconds > 0
        # output is real audio (non-silent)
        data, sr = sf.read(str(result.output_path))
        assert sr == 48_000 and float(np.abs(data).max()) > 0.001

    def test_convert_file_requires_model(self, project_env, tmp_path):
        from rvc_platform.core.entities import ConversionSettings
        from rvc_platform.core.errors import ConversionError
        from rvc_platform.engine.rvc.inference import RVCInferenceEngine
        from rvc_platform.utils.device import DeviceProfile

        eng = RVCInferenceEngine(
            DeviceProfile(device="cpu", is_half=False, gpu_name=None,
                              gpu_memory_gb=0.0, sm=0.0), None)
        import soundfile as sf

        src = tmp_path / "x.wav"
        sf.write(str(src), _speech(seconds=0.5), 16_000)
        with pytest.raises(ConversionError):
            eng.convert_file(src, tmp_path / "o.wav", ConversionSettings())

    def test_share_hubert(self, engine):
        fake = torch.nn.Linear(2, 2)
        engine.share_hubert(fake)          # a real module gets stored
        assert engine._hubert is fake
        engine.share_hubert(None)          # clearing is a no-op
        assert engine._hubert is fake


# --------------------------------------------------------------------------- #
# RealTimeRVC — load + latency bookkeeping (synthetic model)
# --------------------------------------------------------------------------- #
class TestRealTimeRVC:
    @pytest.fixture()
    def rt(self, project_env, monkeypatch, synthetic_model):
        from rvc_platform.engine.rvc import realtime as rt_mod
        from rvc_platform.engine.rvc.realtime import RealTimeRVC
        from rvc_platform.utils.device import DeviceProfile

        def fake_extract(model, source, version, padding_mask=None):
            n = source.shape[-1]
            t = max(1, n // 320)
            feats = torch.zeros(1, t, 768)
            feats[0, :, 0] = torch.linspace(0, 1, t)
            return feats

        monkeypatch.setattr(rt_mod, "extract_hubert_features", fake_extract)
        profile = DeviceProfile(device="cpu", is_half=False, gpu_name=None,
                                gpu_memory_gb=0.0, sm=0.0)
        # f0_method=pm (asset-free); inject a fake hubert so load_model
        # skips the 374 MB asset requirement
        engine = RealTimeRVC(profile, block_ms=160, crossfade_ms=40,
                             f0_method="pm")
        engine.hubert = object()
        engine.load_model(synthetic_model, None)
        return engine

    def test_load_and_status(self, rt):
        assert rt.status == "ready"
        assert rt.model_info is not None
        assert rt.model_info.f0 is True

    def test_process_blocks_and_latency(self, rt):
        t = np.arange(rt.block_frame_16k) / 16_000
        for i in range(6):
            wave = 0.2 * np.sin(2 * np.pi * 150 * (t + i * 0.16))
            out = rt.process_block(wave.astype(np.float32))
            assert out.shape[0] > 0
        stats = rt.latency()
        assert stats.count == 6
        assert stats.avg_ms >= 0

    def test_reset_clears_state(self, rt):
        t = np.arange(rt.block_frame_16k) / 16_000
        rt.process_block((0.2 * np.sin(2 * np.pi * 150 * t)).astype(np.float32))
        rt.reset()
        # buffers are zeroed again
        assert float(rt.sola_buffer.abs().max()) == 0.0
        assert float(rt.input_wav.abs().max()) == 0.0
        # latency records are NOT reset by reset() (separate bookkeeping)
        stats = rt.latency()
        assert stats.count == 1

    def test_update_settings(self, rt):
        rt.update_settings(pitch_shift=2, index_rate=0.3,
                           rms_mix_rate=0.4, silence_threshold_db=-50.0)
        assert rt.pitch_shift == 2
        assert rt.index_rate == pytest.approx(0.3)
        assert rt.rms_mix_rate == pytest.approx(0.4)
        assert rt.silence_threshold_db == -50.0
        # unknown keys are ignored safely
        rt.update_settings(nonexistent=1)

    def test_get_f0_path(self, rt):
        f0 = _speech(seconds=0.5)
        pitch, pitchf = rt._get_f0(f0)
        assert pitch.shape[0] > 0 and pitchf.shape[0] > 0
