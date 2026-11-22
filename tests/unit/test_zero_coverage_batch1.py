"""Unit tests — previously-uncovered modules, batch 1 (the 0% files).

Targets (all were 0% in the round-10 coverage baseline):
  * training/soup.py      — checkpoint averaging (uniform + metric-weighted)
  * engine/rvc/fcpe.py    — FCPE adapter incl. DirectML branches
  * engine/rvc/rmvpe.py   — full RMVPE stack with a real random-weight E2E
  * engine/rvc/models/transforms.py — piecewise rational-quadratic transform
  * infrastructure/audio/_ffmpeg_load.py — vendored any-format loader
  * core/interfaces.py    — runtime-checkable Protocols
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

# --------------------------------------------------------------------------- #
# training/soup.py
# --------------------------------------------------------------------------- #
from rvc_platform.training.soup import (
    _softmax_mwa,
    _score_from_payload,
    _step_key,
    average_state_dicts,
    build_soup,
    list_g_checkpoints,
)


def _g_ckpt(path: Path, step: int, scale: float, metric: float | None = None) -> None:
    payload = {
        "model": {
            "w": torch.full((4,), scale),
            "counter": torch.tensor(step),          # non-float -> newest wins
        },
        "iteration": step,
    }
    if metric is not None:
        payload["best"] = {"composite": metric}
    torch.save(payload, path)


class TestSoup:
    def test_step_key_variants(self):
        assert _step_key("G_23400.pth") == (23400, 0)
        assert _step_key("G_final.pth") == (10**12, 0)
        assert _step_key("junk") == (-1, 0)

    def test_list_g_checkpoints_filters_and_orders(self, tmp_path):
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        for name in ["G_100.pth", "G_300.pth", "G_200.pth", "D_400.pth",
                     "G_soup.pth", "G_23400.pth", "notes.txt"]:
            (ck / name).write_bytes(b"x")
        got = [p.name for p in list_g_checkpoints(ck)]
        # G_soup excluded by the G_<digits> regex; D_/txt excluded
        assert got == ["G_23400.pth", "G_300.pth", "G_200.pth", "G_100.pth"]
        assert [p.name for p in list_g_checkpoints(ck, k=2)] == [
            "G_23400.pth", "G_300.pth"]
        # missing dir -> empty
        assert list_g_checkpoints(tmp_path / "nope") == []

    def test_average_uniform_and_weighted(self):
        a = {"w": torch.tensor([1.0]), "n": torch.tensor(3)}   # n is int-like
        b = {"w": torch.tensor([3.0]), "n": torch.tensor(7)}
        out = average_state_dicts([a, b])
        assert out["w"].item() == pytest.approx(2.0)
        assert out["n"].item() == 7            # newest non-float wins
        out = average_state_dicts([a, b], [3.0, 1.0])
        assert out["w"].item() == pytest.approx(1.5)
        # missing key in one state -> renormalised over present ones
        c = {"w": torch.tensor([10.0])}
        out = average_state_dicts([a, c], [1.0, 1.0])
        assert out["w"].item() == pytest.approx(5.5)

    def test_average_errors(self):
        with pytest.raises(ValueError):
            average_state_dicts([])
        a = {"w": torch.tensor([1.0])}
        with pytest.raises(ValueError):
            average_state_dicts([a, a], [1.0])          # length mismatch
        with pytest.raises(ValueError):
            average_state_dicts([a, a], [0.0, 0.0])     # non-positive sum

    def test_softmax_mwa_prefers_best(self):
        w_min = _softmax_mwa([0.1, 0.5, 0.9], "min")    # lower better
        assert w_min[0] > w_min[1] > w_min[2]
        assert sum(w_min) == pytest.approx(1.0)
        w_max = _softmax_mwa([0.1, 0.5, 0.9], "max")    # higher better
        assert w_max[2] > w_max[1] > w_max[0]
        # identical scores -> equal weights
        assert _softmax_mwa([0.5, 0.5]) == [0.5, 0.5]

    def test_score_from_payload_priority(self):
        assert _score_from_payload({"best": {"val_mel": 3.0}}, ("val_mel",)) == 3.0
        assert _score_from_payload({"composite": 2.0}, ("composite",)) == 2.0
        # 'best' container wins over top level
        p = {"best": {"composite": 1.0}, "composite": 9.0}
        assert _score_from_payload(p, ("composite",)) == 1.0
        assert _score_from_payload({"x": 1}, ("composite",)) is None
        assert _score_from_payload({"composite": "bad"}, ("composite",)) is None

    def test_build_soup_uniform(self, tmp_path):
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        _g_ckpt(ck / "G_100.pth", 100, 1.0)
        _g_ckpt(ck / "G_200.pth", 200, 3.0)
        s = build_soup(tmp_path, k=5)
        assert s["ok"] is True
        assert s["checkpoints"] == ["G_200.pth", "G_100.pth"]
        assert s["newest_epoch"] == 200
        assert s["metric_weighted"] is False
        merged = torch.load(ck / "G_soup.pth", map_location="cpu",
                            weights_only=False)
        assert merged["model"]["w"].flatten().tolist() == pytest.approx(
            [2.0] * 4)
        # non-float entries come from states[-1]; list is newest-first, so
        # this is the oldest checkpoint's counter (upstream semantics)
        assert merged["model"]["counter"].item() == 100
        assert merged["soup"]["sources"] == ["G_200.pth", "G_100.pth"]

    def test_build_soup_metric_weighted(self, tmp_path):
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        _g_ckpt(ck / "G_100.pth", 100, 1.0, metric=0.2)   # better (lower)
        _g_ckpt(ck / "G_200.pth", 200, 3.0, metric=0.8)
        s = build_soup(tmp_path, weighted=True)
        assert s["ok"] is True and s["metric_weighted"] is True
        # paths are newest-first: [G_200(metric .8), G_100(metric .2)]
        # -> the GOOD checkpoint (metric .2) must get the LARGER weight
        w_newest, w_best = s["weights"]
        assert w_best > w_newest
        # weighted average leans toward the good checkpoint
        got = torch.load(ck / "G_soup.pth", map_location="cpu",
                         weights_only=False)["model"]["w"].flatten()[0].item()
        assert got < 2.0

    def test_build_soup_weighted_falls_back_when_metrics_useless(self, tmp_path):
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        _g_ckpt(ck / "G_100.pth", 100, 1.0, metric=0.5)
        _g_ckpt(ck / "G_200.pth", 200, 3.0, metric=0.5)   # identical -> uniform
        s = build_soup(tmp_path, weighted=True)
        assert s["ok"] is True and s["metric_weighted"] is False
        _g_ckpt(ck / "G_300.pth", 300, 5.0)               # no metric at all
        s2 = build_soup(tmp_path, weighted=True, k=3)
        assert s2["ok"] is True and s2["metric_weighted"] is False

    def test_build_soup_failure_paths(self, tmp_path, caplog):
        # <2 checkpoints
        s = build_soup(tmp_path)
        assert s["ok"] is False and "found 0" in s["reason"]
        # checkpoints exist but unreadable
        ck = tmp_path / "checkpoints"
        ck.mkdir()
        (ck / "G_1.pth").write_bytes(b"garbage")
        (ck / "G_2.pth").write_bytes(b"garbage")
        with caplog.at_level(logging.WARNING):
            s = build_soup(tmp_path)
        assert s["ok"] is False and "readable" in s["reason"]


# --------------------------------------------------------------------------- #
# engine/rvc/fcpe.py — DirectML branches (CPU-safe exercises)
# --------------------------------------------------------------------------- #
class TestFcpeHelpers:
    def test_is_directml_device(self):
        from rvc_platform.engine.rvc.fcpe import _is_directml_device

        assert _is_directml_device(torch.device("privateuseone")) is True
        assert _is_directml_device("privateuseone:0") is True
        assert _is_directml_device(torch.device("cpu")) is False
        assert _is_directml_device("cuda:0") is False

    @pytest.mark.skipif(
        pytest.importorskip("torchfcpe") is None, reason="torchfcpe missing"
    )
    def test_fcpe_infer_cpu_end_to_end(self):
        from rvc_platform.engine.rvc.fcpe import FCPEInfer

        inf = FCPEInfer(torch.device("cpu"))
        assert inf.is_directml is False
        assert not hasattr(inf, "cent_table_cpu")
        wav = torch.randn(1, 16000)
        f0 = inf.infer(wav, sr=16000, decoder_mode="local_argmax",
                       threshold=0.006)
        # (batch, frames, 1) — 101 frames for 1 s of 16 kHz audio
        assert f0.shape[1] == 101
        assert torch.isfinite(f0).all()

    def test_fcpe_decode_on_cpu_formula(self):
        """_decode_on_cpu mirrors torchfcpe formulas; run it directly with a
        synthetic infer_model to cover the DML decode branch on CPU."""
        from rvc_platform.engine.rvc.fcpe import FCPEInfer

        inf = FCPEInfer.__new__(FCPEInfer)
        inf.out_dims = 8
        inf.cent_table_cpu = torch.linspace(1800.0, 2400.0, 8)
        latent = torch.rand(1, 3, 8)
        with pytest.raises(ValueError):
            inf._decode_on_cpu(latent, decoder_mode="bogus", threshold=0.0)
        f0 = inf._decode_on_cpu(latent, decoder_mode="argmax", threshold=0.0)
        assert f0.shape == (1, 3, 1)
        # f0 = 10 * 2^(cents/1200) — with cents ~2000 all values are positive
        assert (f0 > 0).all()

    def test_fcpe_graphable_model_infer_local_argmax(self):
        from rvc_platform.engine.rvc.fcpe import FCPEInfer

        class _M(torch.nn.Module):
            cent_table = torch.nn.Parameter(torch.linspace(1800., 2400., 8))
            out_dims = 8

            def forward(self, mel):
                return torch.rand(mel.shape[0], mel.shape[1], 8)

        inf = FCPEInfer.__new__(FCPEInfer)
        inf.infer_model = torch.nn.Module()
        inf.infer_model.model = _M()
        inf.local_offsets = torch.arange(9).view(1, 1, 9)
        mel = torch.rand(1, 3, 16)
        f0 = inf._graphable_model_infer(mel, "local_argmax", 0.0)
        assert f0.shape == (1, 3, 1) and torch.isfinite(f0).all()
        f0a = inf._graphable_model_infer(mel, "argmax", 0.0)
        assert torch.isfinite(f0a).all()
        with pytest.raises(ValueError):
            inf._graphable_model_infer(mel, "bogus", 0.0)


# --------------------------------------------------------------------------- #
# engine/rvc/rmvpe.py — the full stack with a random-weight E2E checkpoint
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rmvpe_instance(tmp_path_factory):
    pytest.importorskip("rvc_platform.engine.rvc.rmvpe")
    from rvc_platform.engine.rvc.rmvpe import E2E, RMVPE

    model = E2E(4, 1, (2, 2))
    p = tmp_path_factory.mktemp("rmvpe") / "rmvpe_random.pt"
    torch.save(model.state_dict(), p)
    return RMVPE(str(p), is_half=False, device="cpu")


class TestRmvpe:
    def test_is_half_string_coercion(self, rmvpe_instance):
        # constructor accepts "false"/"true" strings (upstream behaviour)
        from rvc_platform.engine.rvc.rmvpe import E2E, RMVPE
        import tempfile
        import os

        model = E2E(4, 1, (2, 2))
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        torch.save(model.state_dict(), path)
        try:
            r = RMVPE(path, is_half="False", device="cpu")
            assert r.is_half is False
        finally:
            os.unlink(path)

    def test_infer_from_audio(self, rmvpe_instance):
        audio = (np.random.default_rng(1).standard_normal(16000)
                 * 0.1).astype(np.float32)
        f0 = np.asarray(rmvpe_instance.infer_from_audio(audio, thred=0.03))
        assert f0.shape == (101,)

    def test_mel2hidden_pads_to_32(self, rmvpe_instance):
        mel = torch.randn(1, 128, 50)     # not multiple of 32
        hidden = rmvpe_instance.mel2hidden(mel)
        assert hidden.shape[1] >= 50

    def test_to_local_average_cents(self, rmvpe_instance):
        salience = np.zeros((5, 360), dtype=np.float32)
        salience[:, 100] = 0.5            # a clear peak every frame
        cents = rmvpe_instance.to_local_average_cents(salience, thred=0.05)
        assert cents.shape == (5,)
        assert (cents > 0).all()
        # below-threshold frames -> exactly 0
        salience[:] = 0.01
        assert (rmvpe_instance.to_local_average_cents(salience, thred=0.05) == 0).all()

    def test_stft_transform_inverse_roundtrip(self):
        from rvc_platform.engine.rvc.rmvpe import STFT

        stft = STFT(filter_length=1024, hop_length=512)
        x = torch.randn(1, 16000)              # (batch, samples) — 2-D input
        mag, phase = stft.transform(x, return_phase=True)
        assert mag.shape[1] == 513             # filter_length // 2 + 1
        back = stft.inverse(mag, phase)
        assert back.shape[-1] >= 16000 - 1024  # reconstructed length sane

    def test_unet_blocks_forward_shapes(self):
        from rvc_platform.engine.rvc.rmvpe import (
            BiGRU,
            ConvBlockRes,
            Decoder,
            DeepUnet,
            E2E,
            Encoder,
            Intermediate,
            MelSpectrogram,
            ResDecoderBlock,
            ResEncoderBlock,
        )

        # BiGRU is batch-first (B, T, F) bidirectional
        x_t = torch.randn(2, 10, 16)
        assert BiGRU(16, 8, 2)(x_t).shape == (2, 10, 16)
        # ConvBlockRes / Encoder operate on (B, C, F, T) spectrograms
        x = torch.randn(1, 16, 32, 64)
        assert ConvBlockRes(16, 16)(x).shape == x.shape
        # Encoder: (in_channels, in_size, n_encoders, kernel, n_blocks)
        enc = Encoder(16, 32, 2, 3, 1)
        y, skips = enc(x)                     # forward returns (x, concat_list)
        # blocks output 16 then 32 channels; attrs overshoot by one doubling
        assert y.shape[1] == 32
        assert enc.latent_channels == [[16, 32], [32, 16]]
        assert len(skips) == 2
        resb = ResEncoderBlock(16, 8, 2, n_blocks=2)  # in 16 -> out 8 channels
        y2, pooled = resb(x)
        assert y2.shape[1] == 8
        # AvgPool2d(kernel=2) halves BOTH spatial dims of the pooled output
        assert pooled.shape[-1] == x.shape[-1] // 2
        assert pooled.shape[-2] == x.shape[-2] // 2
        inter = Intermediate(16, 8, 1, 1)      # (in_ch, out_ch, ...)
        assert inter(x).shape[1] == 8
        # decoder blocks upsample via ConvTranspose2d(stride) then concat the
        # matching-width skip (real DeepUnet wiring uses stride=(1, 2))
        dec_blk = ResDecoderBlock(16, 8, stride=(1, 2))
        x16 = torch.randn(1, 16, 32, 32)
        skip8 = torch.randn(1, 8, 32, 64)
        out = dec_blk(x16, skip8)
        assert out.shape == (1, 8, 32, 64)     # time axis upsampled 2x
        dec = Decoder(in_channels=16, n_decoders=1, stride=(1, 2), n_blocks=1)
        assert dec(x16, [skip8]).shape == (1, 8, 32, 64)
        # DeepUnet standalone (geometry must be a multiple of 2^en_de_layers
        # in BOTH dims for the concat to line up)
        unet = DeepUnet(kernel_size=(2, 2), n_blocks=4, en_de_layers=5,
                        inter_layers=4, in_channels=1, en_out_channels=16)
        y_u = unet(torch.randn(1, 1, 128, 128))
        assert y_u.shape == (1, 16, 128, 128)
        # E2E takes (B, frames, 128); frames must be a multiple of 32 (the
        # wrapper pads to 32) AND compatible with the 5-layer unet halving,
        # e.g. 128. Output: (B, frames, 360 salience bins).
        e2e = E2E(4, 1, (2, 2))
        assert e2e(torch.randn(1, 128, 128)).shape == (1, 128, 360)
        mel = MelSpectrogram(False, 128, 16000, 1024, 160, None, 30, 8000)
        out = mel(torch.randn(1, 16000))
        assert out.shape[1] == 128

    def test_mel_spectrogram_keyshift_speed(self):
        from rvc_platform.engine.rvc.rmvpe import MelSpectrogram

        mel = MelSpectrogram(False, 128, 16000, 1024, 160, None, 30, 8000)
        out = mel(torch.randn(1, 16000), keyshift=2, speed=1.5)
        assert out.shape[1] == 128 and torch.isfinite(out).all()

    def test_extract_mel_resamples(self, rmvpe_instance):
        audio_32k = np.random.default_rng(2).standard_normal(32000) * 0.1
        mel = rmvpe_instance.extract_mel(audio_32k.astype(np.float32))
        # mel extractor: (1, 128, frames) tensor
        assert mel.dim() == 3
        assert mel.shape[1] == 128


# --------------------------------------------------------------------------- #
# engine/rvc/models/transforms.py
# --------------------------------------------------------------------------- #
class TestTransforms:
    @staticmethod
    def _params(shape, bins=5):
        w = torch.randn(*shape, bins)
        h = torch.randn(*shape, bins)
        d = torch.randn(*shape, bins)
        return w, h, d

    def test_unconstrained_spline_roundtrip(self):
        """The exact call pattern of Flip in models/modules.py (num_bins,
        linear tails) — forward/inverse round trip to float precision."""
        from rvc_platform.engine.rvc.models.transforms import (
            piecewise_rational_quadratic_transform,
        )

        x = torch.randn(1, 1, 64) * 3.0        # some points outside tail bound
        w, h, d = self._params((1, 1, 64))
        y, logdet = piecewise_rational_quadratic_transform(
            x, w, h, d, inverse=False, tails="linear", tail_bound=5.0)
        assert torch.isfinite(y).all() and torch.isfinite(logdet).all()
        x_back, logdet_inv = piecewise_rational_quadratic_transform(
            y, w, h, d, inverse=True, tails="linear", tail_bound=5.0)
        assert (x_back - x).abs().max() < 1e-3   # float32 inverse tolerance
        assert (logdet + logdet_inv).abs().max() < 1e-3

    def test_flip_module(self):
        """Flip — the simple channel-flip flow used inside the RVC flows."""
        from rvc_platform.engine.rvc.models.modules import Flip

        x = torch.randn(2, 16, 50)
        m = torch.ones_like(x)
        f = Flip()
        y, logdet = f(x, m, reverse=False)
        assert torch.equal(y, torch.flip(x, [1]))
        assert torch.equal(logdet, torch.zeros(2))
        back, _ = f(y, m, reverse=True)      # reverse returns (x, dummy)
        assert torch.equal(back, x)


# --------------------------------------------------------------------------- #
# infrastructure/audio/_ffmpeg_load.py — vendored any-format loader
# (guarded at CLASS level: a missing optional dep skips only this section,
# never the whole module)
# --------------------------------------------------------------------------- #
try:
    from rvc_platform.infrastructure.audio import _ffmpeg_load as ffmpeg_load

    _FFMPEG_LOAD_MISSING = False
except Exception:  # ffmpeg-python / av not installed in this env
    ffmpeg_load = None
    _FFMPEG_LOAD_MISSING = True


@pytest.mark.skipif(_FFMPEG_LOAD_MISSING,
                    reason="requires ffmpeg-python + av")
class TestFfmpegLoad:
    def test_module_constants(self):
        assert ffmpeg_load.AUDIO_LOAD_BACKEND in ("ffmpeg", "torchaudio_cuda")
        assert ffmpeg_load.TORCHAUDIO_GPU_ENABLED is False  # no CUDA here

    def test_wav2_and_transcode_roundtrip(self, tmp_path):
        import soundfile as sf

        src = tmp_path / "in.wav"
        sf.write(str(src), np.zeros(1600, dtype=np.float32), 16000)
        out = tmp_path / "out.mp4"          # mp4 container -> aac codec path
        ffmpeg_load.wav2(str(src), str(out), "m4a")
        assert out.is_file() and out.stat().st_size > 0
        out2 = tmp_path / "out.mp3"
        ffmpeg_load.transcode_audio_file(str(src), str(out2), "mp3")
        assert out2.is_file() and out2.stat().st_size > 0

    def test_wav2_no_audio_stream_raises(self, tmp_path):
        p = tmp_path / "silent.mp4"
        # an empty file has no audio stream -> ValueError from wav2
        p.write_bytes(b"")
        with pytest.raises(Exception):
            ffmpeg_load.wav2(str(p), str(tmp_path / "o.mp3"), "mp3")

    def test_probe_and_decode(self, tmp_path):
        import soundfile as sf

        src = tmp_path / "p.wav"
        sf.write(str(src), np.zeros(8000, dtype=np.float32), 16000)
        sr, ch = ffmpeg_load._probe_audio(str(src))
        assert sr == 16000 and ch == 1
        audio, src_sr = ffmpeg_load._decode_audio_ffmpeg(str(src))
        assert audio.dtype == np.float32      # (channels, samples)
        assert audio.shape[0] == 1 and src_sr == 16000

    def test_load_audio_ffmpeg_resamples(self, tmp_path):
        import soundfile as sf

        src = tmp_path / "r.wav"
        sf.write(str(src), np.linspace(-0.5, 0.5, 8000, dtype=np.float32), 16000)
        audio = ffmpeg_load._load_audio_ffmpeg(str(src), 32000)
        assert audio.dtype == np.float32

    def test_channel_first_tensor_and_format(self):
        # numpy 2-D (samples, ch) -> tensor keeps shape (2, 100)
        a = ffmpeg_load._channel_first_tensor(
            np.zeros((2, 100), dtype=np.float32))
        assert a.shape == (2, 100)
        # 1-D tensor gets a leading channel dim
        t = ffmpeg_load._channel_first_tensor(torch.zeros(100))
        assert list(t.shape) == [1, 100]
        # 3-D input is rejected
        with pytest.raises(ValueError):
            ffmpeg_load._channel_first_tensor(torch.zeros(1, 2, 3))
        # _format_audio_tensor without keep_on_device returns a numpy 1-D
        # mono track (upstream convention: flat float32 samples)
        one = ffmpeg_load._format_audio_tensor(torch.zeros(1, 100))
        assert one.shape == (100,) and one.dtype == np.float32
        # keep_on_device keeps a torch tensor and returns flat mono
        t_dev = ffmpeg_load._format_audio_tensor(
            torch.zeros(1, 100), keep_on_device=True)
        assert torch.is_tensor(t_dev)

    def test_resample_audio(self):
        a = np.zeros((1, 16000), dtype=np.float32)
        out = ffmpeg_load.resample_audio(a, 16000, 32000)
        assert out.shape[-1] == 32000
        out_t = ffmpeg_load.resample_audio_tensor(
            torch.zeros(1, 16000), 16000, 8000)
        assert out_t.shape[-1] == 8000

    def test_load_audio_tensor_and_load_audio(self, tmp_path):
        import soundfile as sf

        src = tmp_path / "l.wav"
        sf.write(str(src), np.zeros(16000, dtype=np.float32), 16000)
        t = ffmpeg_load.load_audio_tensor(str(src), 16000)
        assert t.shape[-1] >= 15000
        a = ffmpeg_load.load_audio(str(src), 16000)
        assert a.shape[-1] >= 15000
        # non-mono collapse
        sf.write(str(src), np.zeros((16000, 2), dtype=np.float32), 16000)
        mono = ffmpeg_load.load_audio(str(src), 16000, force_mono=True)
        assert mono.ndim == 1

    def test_get_gpu_resampler_is_cached(self):
        """Without CUDA the helper still builds (CPU Resample) and caches."""
        r1 = ffmpeg_load._get_gpu_resampler(16000, 32000)
        r2 = ffmpeg_load._get_gpu_resampler(16000, 32000)
        assert r1 is r2                     # cached per (src, dst) pair
        assert ffmpeg_load._get_gpu_resampler(8000, 8000) is not None


# --------------------------------------------------------------------------- #
# core/interfaces.py — runtime-checkable Protocol contracts
# --------------------------------------------------------------------------- #
class TestInterfaceProtocols:
    def test_runtime_checkable_contracts(self):
        from rvc_platform.core.interfaces import (
            AudioProcessor,
            FeatureIndex,
            TrainingRunner,
            TunnelManager,
            VoiceModelEngine,
        )

        class Dummy:
            def status(self): return "ok"

            def load_model(self, model_path, index_path): return None

            def unload(self): return None

            @property
            def loaded_model(self): return None

            def convert_file(self, *a, **k): return None

            def load_any(self, path, target_sr): return None

            def resample(self, audio, src_sr, dst_sr): return None

            def normalize(self, audio, peak=0.9): return None

            def denoise(self, audio, sr, strength=1.0): return None

            def detect_silence(self, audio, sr): return []

            def trim_silence(self, audio, sr, margin_ms=120): return None

            def validate_audio_file(self, path): return {}

            def search(self, feats, k=8): return None, None

            def size(self): return 0

            def start(self, config=None): return None

            def stop(self): return None

            def state(self): return None

            def progress(self): return None

            def list_checkpoints(self): return []

            def resume(self, checkpoint=None): return None

            def export(self, epoch=None): return None

        d = Dummy()
        assert isinstance(d, VoiceModelEngine)
        assert isinstance(d, AudioProcessor)
        assert isinstance(d, FeatureIndex)
        assert isinstance(d, TrainingRunner)
        assert isinstance(d, TunnelManager)

    def test_protocols_reject_partial_implementations(self):
        from rvc_platform.core.interfaces import FeatureIndex

        class HalfDone:
            def search(self, feats, k=8): return None, None

        assert not isinstance(HalfDone(), FeatureIndex)
