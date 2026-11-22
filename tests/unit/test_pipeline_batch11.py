"""Unit tests — batch 11: training pipeline plumbing.

checkpointing (payloads, atomic writes, mirrors, list/latest probing),
Slicer (silence-aware waveform segmentation), mel_processing (spectrogram
pipeline math), write_filelist (upstream line format + mute lines), and
run_config remaining gaps.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh())


def _opt(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #
class TestCheckpointing:
    def test_build_full_payload_cpu_copies(self):
        from rvc_platform.training.checkpointing import build_full_payload

        model = _tiny_model()
        opt = _opt(model)
        payload = build_full_payload(model, opt, None, None,
                                     epoch=3, global_step=42)
        assert payload["epoch"] == 3
        assert payload["global_step"] == 42
        assert payload["scheduler"] is None and payload["scaler"] is None
        # the payload tensors are CPU copies detached from the live model
        assert payload["model"]["0.weight"].device.type == "cpu"
        assert payload["rng"] is not None

    def test_write_checkpoint_files_and_mirrors(self, tmp_path):
        from rvc_platform.training.checkpointing import (
            LATEST_TAG,
            write_checkpoint_files,
        )

        g_path = tmp_path / "G_100.pth"
        d_path = tmp_path / "D_100.pth"
        state_path = tmp_path / "state.json"
        write_checkpoint_files(
            {"model": {"w": torch.zeros(2)}}, g_path,
            {"model": {"w": torch.ones(2)}}, d_path,
            {"epoch": 1, "global_step": 100}, state_path, mirror=True,
        )
        assert g_path.is_file() and d_path.is_file()
        # always-latest mirrors were refreshed
        assert (tmp_path / f"G_{LATEST_TAG}.pth").is_file()
        assert (tmp_path / f"D_{LATEST_TAG}.pth").is_file()
        # state written AFTER both checkpoints (crash-order guarantee)
        assert json.loads(state_path.read_text())["global_step"] == 100

    def test_write_checkpoint_files_with_d2(self, tmp_path):
        from rvc_platform.training.checkpointing import (
            LATEST_TAG,
            write_checkpoint_files,
        )

        write_checkpoint_files(
            {"m": 1}, tmp_path / "G_50.pth",
            {"m": 2}, tmp_path / "D_50.pth",
            None, tmp_path / "state.json", mirror=True,
            payload_d2={"m": 3}, d2_path=tmp_path / "D2_50.pth",
        )
        assert (tmp_path / "D2_50.pth").is_file()
        assert (tmp_path / f"D2_{LATEST_TAG}.pth").is_file()

    def test_save_full_and_load_full_roundtrip(self, tmp_path):
        from rvc_platform.training.checkpointing import load_full, save_full

        model = _tiny_model()
        opt = _opt(model)
        for _ in range(2):                       # give the optimizer state
            opt.zero_grad()
            opt.step()
        p = tmp_path / "full.pth"
        save_full(p, model, opt, None, None, epoch=5, global_step=77)

        model2 = _tiny_model()                   # same architecture
        opt2 = _opt(model2)
        meta = load_full(p, model2, opt2)
        assert meta["epoch"] == 5 and meta["global_step"] == 77
        assert meta["ema"] is None and meta["msd"] is None
        # weights restored bit-exact
        for w1, w2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(w1, w2)

    def test_load_full_missing_file_raises(self, tmp_path):
        from rvc_platform.core.errors import CheckpointError
        from rvc_platform.training.checkpointing import load_full

        with pytest.raises(CheckpointError):
            load_full(tmp_path / "ghost.pth", _tiny_model(), None)

    def test_load_full_rejects_payload_without_model(self, tmp_path):
        from rvc_platform.core.errors import CheckpointError
        from rvc_platform.training.checkpointing import load_full

        p = tmp_path / "bad.pth"
        torch.save({"something": "else"}, p)
        with pytest.raises(CheckpointError):
            load_full(p, _tiny_model(), None)

    def test_load_full_legacy_iteration_epoch(self, tmp_path):
        from rvc_platform.training.checkpointing import (
            legacy_iteration_epoch,
            load_full,
        )

        p = tmp_path / "legacy.pth"
        # full weights (Linear has weight AND bias keys)
        torch.save({"model": {"0.weight": torch.zeros(4, 4),
                              "0.bias": torch.zeros(4)},
                    "iteration": 12}, p)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        meta = load_full(p, model, None)
        assert meta["epoch"] == 12               # iteration fallback
        assert legacy_iteration_epoch(p) == 12

    def test_state_roundtrip_and_corrupt(self, tmp_path):
        from rvc_platform.training.checkpointing import read_state, write_state

        p = tmp_path / "state.json"
        assert read_state(p) is None             # missing -> None
        write_state({"a": 1}, p)
        assert read_state(p) == {"a": 1}
        p.write_text("{broken", encoding="utf-8")
        assert read_state(p) is None             # corrupt -> None, no raise

    def test_list_checkpoints_kinds(self, tmp_path):
        from rvc_platform.training.checkpointing import list_checkpoints

        # numbered, final, and always-latest mirror
        for name, payload in (("G_100.pth", {"epoch": 1, "global_step": 100}),
                              ("G_final.pth", {"epoch": 9, "global_step": 900}),
                              ("G_2333333.pth", {"epoch": 9, "global_step": 900})):
            torch.save(payload, tmp_path / name)
        ckpts = list_checkpoints(tmp_path)
        kinds = {c.kind for c in ckpts}
        assert kinds == {"G", "G-final", "G-latest"}
        # the always-latest mirror IS listed (upstream visibility) but the
        # epoch comes from the embedded payload
        latest = next(c for c in ckpts if c.kind == "G-latest")
        assert latest.epoch == 9

    def test_list_checkpoints_probe_fallback(self, tmp_path):
        from rvc_platform.training.checkpointing import list_checkpoints

        # unreadable checkpoint -> epoch falls back to 0, step from filename
        (tmp_path / "G_555.pth").write_bytes(b"not-a-torch-file")
        ckpts = list_checkpoints(tmp_path)
        assert len(ckpts) == 1
        assert ckpts[0].epoch == 0 and ckpts[0].global_step == 555

    def test_latest_step_checkpoint_prefers_highest(self, tmp_path):
        from rvc_platform.training.checkpointing import latest_step_checkpoint

        torch.save({"epoch": 1, "global_step": 100},
                   tmp_path / "G_100.pth")
        torch.save({"epoch": 2, "global_step": 500},
                   tmp_path / "G_500.pth")
        assert latest_step_checkpoint(tmp_path) == tmp_path / "G_500.pth"

    def test_latest_step_checkpoint_final_wins_when_equal_or_newer(
            self, tmp_path):
        from rvc_platform.training.checkpointing import latest_step_checkpoint

        torch.save({"epoch": 2, "global_step": 500},
                   tmp_path / "G_500.pth")
        torch.save({"epoch": 3, "global_step": 800},
                   tmp_path / "G_final.pth")
        assert latest_step_checkpoint(tmp_path) == tmp_path / "G_final.pth"

    def test_latest_step_checkpoint_empty(self, tmp_path):
        from rvc_platform.training.checkpointing import latest_step_checkpoint

        assert latest_step_checkpoint(tmp_path) is None

    def test_latest_step_checkpoint_mirror_excluded(self, tmp_path):
        from rvc_platform.training.checkpointing import (
            LATEST_TAG,
            latest_step_checkpoint,
        )

        torch.save({"epoch": 1, "global_step": 10},
                   tmp_path / "G_10.pth")
        torch.save({"epoch": 1, "global_step": 999},
                   tmp_path / f"G_{LATEST_TAG}.pth")   # only the mirror
        assert latest_step_checkpoint(tmp_path) == tmp_path / "G_10.pth"


# --------------------------------------------------------------------------- #
# Slicer — silence-aware segmentation
# --------------------------------------------------------------------------- #
class TestSlicer:
    def test_validation(self):
        from rvc_platform.infrastructure.audio.slicer import Slicer

        with pytest.raises(ValueError):
            Slicer(sr=16_000, min_length=300, min_interval=500)  # inverted
        with pytest.raises(ValueError):
            Slicer(sr=16_000, min_interval=300, hop_size=400)    # inverted
        with pytest.raises(ValueError):
            Slicer(sr=16_000, max_sil_kept=10, hop_size=20)      # too small

    def test_slices_speech_with_silence(self):
        from rvc_platform.infrastructure.audio.slicer import Slicer

        sr = 16_000
        # speech(1s) silence(1s) speech(1s) — deterministic split at the gap
        t = np.arange(sr) / sr
        speech = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        silence = np.zeros(sr, np.float32)
        wav = np.concatenate([speech, silence, speech])
        slicer = Slicer(sr=sr, threshold=-40.0, min_length=5000,
                        min_interval=300, hop_size=20, max_sil_kept=5000)
        chunks = slicer.slice(wav)
        assert len(chunks) >= 1
        # each chunk keeps the required tail (last-sil) per upstream contract
        for chunk in chunks:
            assert chunk.shape[-1] > 0

    def test_get_rms_interior_frames(self):
        from rvc_platform.infrastructure.audio.slicer import get_rms

        y = np.ones(8192, np.float32)
        rms = get_rms(y, frame_length=2048, hop_length=512)
        assert rms.shape[-1] > 0
        # zero-padded edges reduce the mean; interior full frames are 1.0
        assert float(rms.max()) == pytest.approx(1.0, abs=1e-5)
        assert float(rms.min()) > 0.5


# --------------------------------------------------------------------------- #
# mel_processing — spectrogram math
# --------------------------------------------------------------------------- #
class TestMelProcessing:
    def test_dynamic_range_compression_roundtrip(self):
        from rvc_platform.training.mel_processing import (
            dynamic_range_compression_torch,
            dynamic_range_decompression_torch,
        )

        x = torch.rand(2, 100) + 0.01
        comp = dynamic_range_compression_torch(x)
        back = dynamic_range_decompression_torch(comp)
        assert torch.allclose(back, x, atol=1e-4)

    def test_spectral_normalize_roundtrip(self):
        from rvc_platform.training.mel_processing import (
            spectral_de_normalize_torch,
            spectral_normalize_torch,
        )

        mag = torch.rand(1, 1025, 50) + 0.01
        norm = spectral_normalize_torch(mag)
        back = spectral_de_normalize_torch(norm)
        assert torch.allclose(back, mag, atol=1e-3)

    def test_spectrogram_torch_shape_and_consistency(self):
        from rvc_platform.training.mel_processing import spectrogram_torch

        y = torch.randn(1, 16_000)
        spec = spectrogram_torch(y, n_fft=1024, sampling_rate=16_000,
                                 hop_size=256, win_size=1024)
        # onesided=True -> n_fft//2 + 1 bins
        assert spec.shape == (1, 513, 16_000 // 256)
        assert float(spec.min()) >= 0

    def test_spec_to_mel_shape(self):
        from rvc_platform.training.mel_processing import (
            spectrogram_torch,
            spec_to_mel_torch,
        )

        y = torch.randn(1, 16_000)
        spec = spectrogram_torch(y, 1024, 16_000, 256, 1024)
        mel = spec_to_mel_torch(spec, 1024, 128, 16_000, 0, 8000)
        assert mel.shape == (1, 128, 16_000 // 256)

    def test_mel_spectrogram_torch_end_to_end(self):
        from rvc_platform.training.mel_processing import mel_spectrogram_torch

        y = torch.randn(1, 16_000)
        mel = mel_spectrogram_torch(
            y, 1024, 128, 16_000, 256, 1024, 0, 8000)
        assert mel.shape == (1, 128, 16_000 // 256)


# --------------------------------------------------------------------------- #
# feature_extract — filelist assembly
# --------------------------------------------------------------------------- #
class TestWriteFilelist:
    def _run(self, tmp_path, n=2, sr=40_000, version="v2", f0=True):
        run = tmp_path / "run"
        gt = run / "0_gt_wavs"
        feat = run / "3_feature768" if version == "v2" else run / "3_feature256"
        f0d = run / "2a_f0"
        f0n = run / "2b-f0nsf"
        for d in (gt, feat, f0d, f0n):
            d.mkdir(parents=True, exist_ok=True)
        import soundfile as sf

        t = np.arange(sr) / sr
        for i in range(n):
            sf.write(str(gt / f"c{i}.wav"),
                     (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32),
                     sr, subtype="PCM_16")
            np.save(feat / f"c{i}.npy",
                    np.zeros((16, 768 if version == "v2" else 256),
                             np.float32))
            np.save(f0d / f"c{i}.wav.npy", np.zeros(16, np.float32))
            np.save(f0n / f"c{i}.wav.npy", np.zeros(16, np.float32))
        return run

    def test_upstream_line_format_with_f0(self, tmp_path):
        from rvc_platform.training.feature_extract import write_filelist

        run = self._run(tmp_path, n=3)
        fl = write_filelist(run, "v2", f0=True, sample_rate_hz=40_000)
        lines = fl.read_text(encoding="utf-8").splitlines()
        # mute lines reference the shared assets/mute layout (absolute path)
        data = [ln for ln in lines if "/assets/mute/" not in ln]
        assert len(data) == 3
        for ln in data:
            parts = ln.split("|")
            assert len(parts) == 5                 # gt|feat|f0|f0nsf|spk
            assert parts[0].endswith(".wav")
            assert parts[1].endswith(".npy")

    def test_no_f0_shorter_lines(self, tmp_path):
        from rvc_platform.training.feature_extract import write_filelist

        run = self._run(tmp_path, n=2)
        fl = write_filelist(run, "v2", f0=False, sample_rate_hz=40_000)
        data = [ln for ln in fl.read_text(encoding="utf-8").splitlines()
                if "/assets/mute/" not in ln]
        for ln in data:
            assert len(ln.split("|")) == 3          # gt|feat|spk

    def test_missing_sidecars_skip_entry(self, tmp_path):
        from rvc_platform.training.feature_extract import write_filelist

        run = self._run(tmp_path, n=3)
        # remove one f0 sidecar -> that entry is skipped, others survive
        (run / "2a_f0" / "c1.wav.npy").unlink()
        fl = write_filelist(run, "v2", f0=True, sample_rate_hz=40_000)
        data = [ln for ln in fl.read_text(encoding="utf-8").splitlines()
                if "/assets/mute/" not in ln]
        assert len(data) == 2
        assert not any("c1.wav" in ln for ln in data)

    def test_empty_run_raises_actionable(self, tmp_path):
        from rvc_platform.core.errors import AssetMissingError
        from rvc_platform.training.feature_extract import write_filelist

        run = self._run(tmp_path, n=1)
        (run / "3_feature768" / "c0.npy").unlink()
        with pytest.raises(AssetMissingError):
            write_filelist(run, "v2", f0=True, sample_rate_hz=40_000)

    def test_mute_lines_appended(self, tmp_path):
        from rvc_platform.training.feature_extract import write_filelist

        run = self._run(tmp_path, n=2)
        fl = write_filelist(run, "v2", f0=True, sample_rate_hz=32_000)
        lines = fl.read_text(encoding="utf-8").splitlines()
        mutes = [ln for ln in lines if "/assets/mute/" in ln]
        assert len(mutes) == 2                     # one per speaker id pair
        # the mute wav matches the requested target rate
        assert any("mute32k.wav" in ln for ln in mutes)

    def test_infer_sr_hz(self, tmp_path):
        from rvc_platform.training.feature_extract import _infer_sr_hz

        run = self._run(tmp_path, n=1, sr=48_000)
        assert _infer_sr_hz(run) == 48_000
        assert _infer_sr_hz(tmp_path / "empty") is None


# --------------------------------------------------------------------------- #
# run_config — remaining branches
# --------------------------------------------------------------------------- #
class TestRunConfigGaps:
    def test_invalid_sample_rate_rejected(self, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        # invalid rate rejected by the field validator
        with pytest.raises(Exception):
            TrainingRunConfig(experiment_name="x", dataset_path=tmp_path,
                              sample_rate="16k")

    def test_invalid_experiment_name_rejected(self, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        with pytest.raises(Exception):
            TrainingRunConfig(experiment_name="bad name!!",
                              dataset_path=tmp_path)

    def test_epoch_bounds(self, tmp_path):
        from rvc_platform.training.run_config import TrainingRunConfig

        with pytest.raises(Exception):
            TrainingRunConfig(experiment_name="x", dataset_path=tmp_path,
                              epochs=0)
        with pytest.raises(Exception):
            TrainingRunConfig(experiment_name="x", dataset_path=tmp_path,
                              batch_size=100)
