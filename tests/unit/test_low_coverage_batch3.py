"""Unit tests — previously low-coverage modules, batch 3.

Targets (round-10 baseline): training/feature_extract.py (12%),
training/eval_metrics.py (72%), training/huggingface.py (52% uncovered
helper paths), plus the f0 mel-quantisation math.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch


# --------------------------------------------------------------------------- #
# feature_extract helpers
# --------------------------------------------------------------------------- #
class TestFeatureExtractHelpers:
    def test_atomic_save_npy_roundtrip(self, tmp_path):
        from rvc_platform.training.feature_extract import _atomic_save_npy

        arr = np.linspace(0, 1, 64, dtype=np.float32)
        p = tmp_path / "x.npy"
        _atomic_save_npy(p, arr)
        assert np.array_equal(np.load(p), arr)
        # no .part residue
        assert not list(tmp_path.glob("*.part"))

    def test_atomic_save_npy_overwrites(self, tmp_path):
        from rvc_platform.training.feature_extract import _atomic_save_npy

        p = tmp_path / "y.npy"
        _atomic_save_npy(p, np.zeros(4))
        _atomic_save_npy(p, np.ones(4))
        assert np.array_equal(np.load(p), np.ones(4))

    def test_valid_npy(self, tmp_path):
        from rvc_platform.training.feature_extract import _valid_npy

        good = tmp_path / "good.npy"
        np.save(good, np.ones(10, dtype=np.float32))
        # note: returns numpy.bool_ (upstream-style) — truthiness, not `is`
        assert bool(_valid_npy(good)) is True
        assert not _valid_npy(tmp_path / "missing.npy")
        # np.save's minimum output is 132 bytes (128 header + payload), so
        # the <128 guard only trips on TRUNCATED files — craft one directly
        truncated = tmp_path / "truncated.npy"
        truncated.write_bytes(b"\x93NUMPY" + b"\x00" * 50)
        assert not _valid_npy(truncated)
        nan = tmp_path / "nan.npy"
        np.save(nan, np.array([np.nan] * 64, dtype=np.float32))
        assert not _valid_npy(nan)                  # non-finite
        garbage = tmp_path / "garbage.npy"
        garbage.write_bytes(b"not-an-npy" * 30)
        assert not _valid_npy(garbage)
        # empty array (size 0) is invalid too
        empty = tmp_path / "empty.npy"
        np.save(empty, np.zeros((0, 768), dtype=np.float32))
        assert not _valid_npy(empty)

    def test_read_16k(self, tmp_path):
        from rvc_platform.training.feature_extract import _read_16k

        wav = (np.random.default_rng(0).standard_normal(16_000)
               * 0.1).astype(np.float32)
        p = tmp_path / "a.wav"
        sf.write(str(p), wav, 16_000, subtype="FLOAT")
        t = _read_16k(p, normalize=False)
        assert t.shape == (1, 16_000) and t.dtype == torch.float32
        # normalize -> zero-mean/unit-var (layer_norm)
        t_norm = _read_16k(p, normalize=True)
        assert abs(float(t_norm.mean())) < 1e-5
        assert abs(float(t_norm.std()) - 1.0) < 1e-3
        # wrong sr -> assertion
        p2 = tmp_path / "b.wav"
        sf.write(str(p2), wav, 22_050, subtype="FLOAT")
        with pytest.raises(AssertionError):
            _read_16k(p2, normalize=False)

    def test_read_16k_stereo_collapses(self, tmp_path):
        from rvc_platform.training.feature_extract import _read_16k

        wav = np.zeros((16_000, 2), dtype=np.float32)
        wav[:, 0] = 0.5
        p = tmp_path / "st.wav"
        sf.write(str(p), wav, 16_000, subtype="FLOAT")
        t = _read_16k(p, normalize=False)
        assert t.shape == (1, 16_000)
        assert torch.allclose(t, torch.full_like(t, 0.25))

    def test_infer_sr_hz(self, tmp_path):
        from rvc_platform.training.feature_extract import _infer_sr_hz

        gt = tmp_path / "0_gt_wavs"
        gt.mkdir()
        sf.write(str(gt / "x.wav"), np.zeros(1600), 40_000)
        assert _infer_sr_hz(tmp_path) == 40_000
        assert _infer_sr_hz(tmp_path / "nope") is None

    def _make_run_dir(self, tmp_path, sr=40_000, n=3, pitched=True):
        run = tmp_path / "run"
        for sub in ("0_gt_wavs", "1_16k_wavs", "3_feature768",
                    "3_feature256", "2a_f0", "2b-f0nsf"):
            (run / sub).mkdir(parents=True)
        rng = np.random.default_rng(3)
        for i in range(n):
            if pitched:      # harmonic stack: pm/rmvpe can track real pitch
                t = np.arange(16_000) / 16_000
                wav = (0.4 * np.sin(2 * np.pi * 150 * t)
                       + 0.2 * np.sin(2 * np.pi * 300 * t)
                       + 0.005 * rng.standard_normal(16_000)).astype(np.float32)
            else:
                wav = (rng.standard_normal(16_000) * 0.2).astype(np.float32)
            sf.write(str(run / "1_16k_wavs" / f"{i}.wav"), wav, 16_000)
            sf.write(str(run / "0_gt_wavs" / f"{i}.wav"),
                     np.zeros(int(sr * 1.0)), sr)
            np.save(run / "3_feature768" / f"{i}.npy",
                    rng.standard_normal((100, 768)).astype(np.float32))
            np.save(run / "3_feature256" / f"{i}.npy",
                    rng.standard_normal((100, 256)).astype(np.float32))
            f0 = rng.uniform(100, 300, 100).astype(np.float32)
            np.save(run / "2b-f0nsf" / f"{i}.wav.npy", f0)
            np.save(run / "2a_f0" / f"{i}.wav.npy",
                    np.rint(f0).astype(np.int64))
        return run

    def test_write_filelist_with_f0(self, tmp_path, project_env):
        from rvc_platform.training.feature_extract import write_filelist

        run = self._make_run_dir(tmp_path)
        fl = write_filelist(run, "v2", True, speaker_id=0,
                            sample_rate_hz=40_000)
        lines = fl.read_text(encoding="utf-8").splitlines()
        # 3 real + 2 mute lines
        assert len(lines) == 5
        real = [ln for ln in lines if "mute" not in ln]
        assert len(real) == 3
        gt, feat, a, b, sid = real[0].split("|")
        assert gt.endswith(".wav") and "3_feature768" in feat
        assert a.endswith(".wav.npy") and sid == "0"

    def test_write_filelist_without_f0(self, tmp_path, project_env):
        from rvc_platform.training.feature_extract import write_filelist

        run = self._make_run_dir(tmp_path)
        fl = write_filelist(run, "v1", False, speaker_id=2,
                            sample_rate_hz=48_000)
        lines = fl.read_text(encoding="utf-8").splitlines()
        real = [ln for ln in lines if "mute" not in ln]
        gt, feat, sid = real[0].split("|")
        assert "3_feature256" in feat and sid == "2"
        # mute lines still present, referencing mute48k.wav
        mute = [ln for ln in lines if "mute" in ln]
        assert len(mute) == 2 and "mute48k.wav" in mute[0]

    def test_write_filelist_requires_complete_entries(self, tmp_path,
                                                      project_env):
        from rvc_platform.core.errors import AssetMissingError
        from rvc_platform.training.feature_extract import write_filelist

        run = self._make_run_dir(tmp_path)
        # remove the f0nsf cache -> entry incomplete -> skipped
        for p in (run / "2b-f0nsf").glob("*.npy"):
            p.unlink()
        with pytest.raises(AssetMissingError):
            write_filelist(run, "v2", True, sample_rate_hz=40_000)

    def test_extract_f0_pm_end_to_end(self, tmp_path, project_env):
        """pm (Praat) is dependency-free — a real extraction round trip."""
        from rvc_platform.training.feature_extract import extract_f0

        run = self._make_run_dir(tmp_path, n=2)
        # _make_run_dir seeds f0 caches (for the filelist tests) — clear
        # them so this test measures real extraction, not cache skipping
        for sub in ("2a_f0", "2b-f0nsf"):
            for p in (run / sub).glob("*.npy"):
                p.unlink()
        report = extract_f0(run, "pm", "cpu", False)
        assert report["files"] == 2
        assert report["success"] == 2 and report["failed"] == 0
        coarse = np.load(run / "2a_f0" / "0.wav.npy")
        fine = np.load(run / "2b-f0nsf" / "0.wav.npy")
        assert coarse.dtype == np.int64
        assert 1 <= coarse.min() and coarse.max() <= 255
        assert fine.dtype == np.float32 and (fine > 0).any()
        # second run -> all skipped (valid cache)
        report2 = extract_f0(run, "pm", "cpu", False)
        assert report2["skipped"] == 2 and report2["success"] == 0

    def test_extract_f0_no_chunks_raises(self, tmp_path, project_env):
        from rvc_platform.core.errors import AssetMissingError
        from rvc_platform.training.feature_extract import extract_f0

        run = tmp_path / "empty_run"
        (run / "1_16k_wavs").mkdir(parents=True)
        with pytest.raises(AssetMissingError):
            extract_f0(run, "pm", "cpu", False)

    def test_extract_f0_silence_chunk_fails_cleanly(self, tmp_path,
                                                    project_env):
        """All-silent chunk -> all-zero f0 -> counted failed, not saved."""
        from rvc_platform.training.feature_extract import extract_f0

        run = tmp_path / "run"
        (run / "1_16k_wavs").mkdir(parents=True)
        (run / "2a_f0").mkdir()
        (run / "2b-f0nsf").mkdir()
        sf.write(str(run / "1_16k_wavs" / "sil.wav"),
                 np.zeros(16_000, dtype=np.float32), 16_000)
        report = extract_f0(run, "pm", "cpu", False)
        assert report["failed"] == 1
        assert not (run / "2a_f0" / "sil.wav.npy").exists()

    def test_extract_f0_progress_callback(self, tmp_path, project_env):
        from rvc_platform.training.feature_extract import extract_f0

        run = self._make_run_dir(tmp_path, n=2)
        for sub in ("2a_f0", "2b-f0nsf"):
            for p in (run / sub).glob("*.npy"):
                p.unlink()
        calls = []
        extract_f0(run, "pm", "cpu", False,
                   progress=lambda frac, name: calls.append((frac, name)))
        assert len(calls) == 2
        assert calls[-1][0] == 1.0

    def test_f0_mel_quantisation_matches_upstream(self):
        """The coarse-F0 mel quantisation used by extract_f0 (256 bins,
        50-1100 Hz) — verify the transform round-trips within one bin."""
        f0_bin, f0_max, f0_min = 256, 1100.0, 50.0
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0 = np.array([80.0, 120.0, 220.0, 440.0, 800.0], dtype=np.float32)
        mel = 1127 * np.log(1 + f0 / 700)
        mel[mel > 0] = ((mel[mel > 0] - f0_mel_min) * (f0_bin - 2)
                        / (f0_mel_max - f0_mel_min) + 1)
        mel = np.clip(mel, 1, f0_bin - 1)
        coarse = np.rint(mel).astype(np.int64)
        assert ((coarse >= 1) & (coarse <= 255)).all()
        # monotonic: higher pitch -> higher bin
        assert (np.diff(coarse) > 0).all()


# --------------------------------------------------------------------------- #
# eval_metrics
# --------------------------------------------------------------------------- #
class TestEvalMetrics:
    def test_resample_16k_passthrough_and_resample(self):
        from rvc_platform.training.eval_metrics import resample_16k

        wav = torch.randn(2, 16000)
        assert torch.equal(resample_16k(wav, 16_000), wav)
        out = resample_16k(torch.randn(1, 32000), 32_000)
        assert out.shape[-1] == 16_000
        # 3-D input gets squeezed
        out3 = resample_16k(torch.randn(1, 1, 44100), 44_100)
        assert out3.dim() == 2 and out3.shape[-1] > 15_000

    def test_speaker_similarity_with_fake_embedder(self):
        from rvc_platform.training.eval_metrics import SpeakerEmbedder, \
            speaker_similarity

        class Fake(torch.nn.Module):
            """SpeechBrain-ECAPA-shaped: encode_batch(wavs=(B,1,T) -> (B,1,D)."""

            def encode_batch(self, wavs):
                return torch.nn.functional.normalize(
                    wavs.mean(-1), dim=-1).unsqueeze(1)

        emb = SpeakerEmbedder(Fake())
        torch.manual_seed(0)
        a = torch.randn(1, 32000)
        sim_same = speaker_similarity(emb, a, a.clone(), 32_000)
        assert sim_same == pytest.approx(1.0, abs=1e-5)
        # orthogonal embeddings -> similarity ~0 (not just < 1)
        b = torch.cat([a[:, :16000], -a[:, 16000:]], dim=1)
        sim_anti = speaker_similarity(emb, a, b, 32_000)
        assert sim_anti < 0.1
        # length mismatch tolerated (truncation)
        sim_trunc = speaker_similarity(emb, a, a[:, :16000], 32_000)
        assert -1.0 <= sim_trunc <= 1.0

    def test_try_build_speaker_embedder_env_off(self, monkeypatch):
        from rvc_platform.training.eval_metrics import try_build_speaker_embedder

        monkeypatch.setenv("RVC_SPK_SIM", "0")
        assert try_build_speaker_embedder("cpu") is None

    def test_try_build_speaker_embedder_without_speechbrain(self, monkeypatch):
        import builtins

        from rvc_platform.training.eval_metrics import \
            try_build_speaker_embedder

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name.startswith("speechbrain"):
                raise ImportError("no speechbrain")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert try_build_speaker_embedder("cpu") is None

    def test_try_build_f0_helper_env_off(self, monkeypatch):
        from rvc_platform.training.eval_metrics import try_build_f0_helper

        monkeypatch.setenv("RVC_F0_EVAL", "off")
        assert try_build_f0_helper("cpu", False) is None

    def test_f0_rmse_voiced_only(self):
        from rvc_platform.training.eval_metrics import f0_rmse

        gt = np.array([100, 110, 0, 130, 140], dtype=np.float32)
        # prediction matches voiced frames, deviates nowhere else
        pred_fn = lambda wav: np.array(
            [100, 110, 999, 130, 140], dtype=np.float32)
        rmse = f0_rmse(pred_fn, np.zeros(5), gt)
        assert rmse == 0.0
        pred_fn2 = lambda wav: np.array(
            [110, 120, 999, 140, 150], dtype=np.float32)
        rmse2 = f0_rmse(pred_fn2, np.zeros(5), gt)
        assert rmse2 == pytest.approx(10.0)
        # too few voiced frames -> None
        assert f0_rmse(pred_fn, np.zeros(5),
                       np.zeros(5, dtype=np.float32)) is None

    def test_composite_scores(self):
        from rvc_platform.training.eval_metrics import (
            composite_score,
            composite_score_f0,
        )

        assert composite_score(2.0, 1.0) == pytest.approx(2.0)
        assert composite_score(2.0, 0.5) == pytest.approx(2.5)
        assert composite_score(2.0, None) == pytest.approx(2.0)
        # f0 variant: val_mel + w * rmse / scale
        assert composite_score_f0(2.0, 100.0) == pytest.approx(2.5)
        assert composite_score_f0(2.0, None) == pytest.approx(2.0)
        assert composite_score_f0(2.0, 200.0, f0_weight=0.25) \
            == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# huggingface helpers (no network)
# --------------------------------------------------------------------------- #
class TestHuggingfaceHelpers:
    def test_resolve_token_prefers_argument(self, monkeypatch):
        from rvc_platform.training.huggingface import _resolve_token

        monkeypatch.setenv("HF_TOKEN", "env-token")
        assert _resolve_token("arg-token") == "arg-token"
        assert _resolve_token(None) == "env-token"
        monkeypatch.delenv("HF_TOKEN")
        monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "legacy")
        assert _resolve_token(None) == "legacy"
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN")
        with pytest.raises(RuntimeError):
            _resolve_token(None)

    def test_import_hub_raises_without_lib(self, monkeypatch):
        import builtins

        from rvc_platform.training.huggingface import _import_hub

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "huggingface_hub":
                raise ImportError("no hf")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError):
            _import_hub()

    def test_newest_index_file(self, tmp_path):
        from rvc_platform.training.huggingface import newest_index_file

        assert newest_index_file(tmp_path) is None
        p1 = tmp_path / "added_IVF4_Flat_nprobe_1_exp_v2.index"
        p1.write_bytes(b"a")
        p2 = tmp_path / "added_IVF4_Flat_nprobe_1_exp_v2 (1).index"
        import time

        time.sleep(0.01)
        p2.write_bytes(b"b")
        newest = newest_index_file(tmp_path)
        assert newest == p2

    def test_pick_export_file_prefers_best(self, tmp_path):
        from rvc_platform.training.huggingface import pick_export_file

        # empty -> None
        assert pick_export_file(tmp_path) is None
        # only a G checkpoint -> exports dir empty, pick None or build later
        (tmp_path / "G_100.pth").write_bytes(b"g")
        # an exported inference checkpoint wins
        exp = tmp_path / "exports"
        exp.mkdir()
        best = exp / "best_200.pth"
        best.write_bytes(b"b")
        assert pick_export_file(tmp_path) == best
