"""Regression tests for the official mute-silence training asset.

Upstream RVC (webui.py run_train_model) appends TWO mute lines per speaker
id — by reference into logs/mute — when writing the training filelist.
Our port generates a functionally-equivalent asset (assets/mute, official
layout names) and injects the same 2-line dose through
``mute_asset.mute_filelist_lines``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from rvc_platform.training import mute_asset
from rvc_platform.training.feature_extract import write_filelist


def _make_run(run_dir, n_files: int = 2, sr: int = 48_000) -> None:
    gt = run_dir / "0_gt_wavs"
    w16 = run_dir / "1_16k_wavs"
    feat = run_dir / "3_feature768"
    f0 = run_dir / "2a_f0"
    f0nsf = run_dir / "2b-f0nsf"
    for d in (gt, w16, feat, f0, f0nsf):
        d.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        sf.write(gt / f"f{i}.wav", np.zeros(sr, dtype=np.float32), sr)
        sf.write(w16 / f"f{i}.wav", np.zeros(16_000, dtype=np.float32), 16_000)
        np.save(feat / f"f{i}.npy", np.zeros((5, 768), dtype=np.float32))
        np.save(f0 / f"f{i}.wav.npy", np.zeros(5, dtype=np.float32))
        np.save(f0nsf / f"f{i}.wav.npy", np.zeros(5, dtype=np.float32))


class TestMuteAssetLayout:
    def test_official_layout_generated(self, tmp_path, monkeypatch):
        from rvc_platform.infrastructure import paths as paths_mod

        # point the module-level get_paths at a tmp-rooted layout
        fake = paths_mod.Paths(
            root=tmp_path, assets=tmp_path / "assets", models=tmp_path / "models",
            data=tmp_path / "data", runs=tmp_path / "runs",
            cache=tmp_path / "cache", logs=tmp_path / "logs", temp=tmp_path / "temp",
        )
        monkeypatch.setattr(mute_asset, "get_paths", lambda: fake)

        d = mute_asset.ensure_mute_asset()
        layout = mute_asset._official_layout(d)
        assert layout["wav_16k"].is_file()
        assert (layout["f0"]).is_file()
        assert (layout["wav_sr"] / "mute48k.wav").is_file()
        assert (layout["wav_sr"] / "mute32k.wav").is_file()
        # zero F0 frames
        f0 = np.load(layout["f0"])
        assert f0.shape[0] == 1000 and not f0.any()

    def test_two_lines_per_speaker_dose(self, tmp_path, monkeypatch):
        from rvc_platform.infrastructure import paths as paths_mod

        fake = paths_mod.Paths(
            root=tmp_path, assets=tmp_path / "assets", models=tmp_path / "models",
            data=tmp_path / "data", runs=tmp_path / "runs",
            cache=tmp_path / "cache", logs=tmp_path / "logs", temp=tmp_path / "temp",
        )
        monkeypatch.setattr(mute_asset, "get_paths", lambda: fake)

        lines = mute_asset.mute_filelist_lines(48_000, "v2", True, 0)
        assert len(lines) == 2, "upstream dose: exactly 2 mute lines per speaker"
        gt, feat, a, b, sid = lines[0].split("|")
        assert gt.endswith("mute48k.wav")
        assert feat.endswith("mute.npy") and "3_feature768" in feat
        assert a.endswith("mute.wav.npy") and "2a_f0" in a
        assert sid == "0"

    def test_non_f0_variant(self, tmp_path, monkeypatch):
        from rvc_platform.infrastructure import paths as paths_mod

        fake = paths_mod.Paths(
            root=tmp_path, assets=tmp_path / "assets", models=tmp_path / "models",
            data=tmp_path / "data", runs=tmp_path / "runs",
            cache=tmp_path / "cache", logs=tmp_path / "logs", temp=tmp_path / "temp",
        )
        monkeypatch.setattr(mute_asset, "get_paths", lambda: fake)

        lines = mute_asset.mute_filelist_lines(40_000, "v2", False, 3)
        assert len(lines) == 2
        parts = lines[0].split("|")
        assert len(parts) == 3 and parts[0].endswith("mute40k.wav")
        assert parts[-1] == "3"


class TestFilelistIntegration:
    def test_write_filelist_includes_mute_lines(self, tmp_path, monkeypatch):
        from rvc_platform.infrastructure import paths as paths_mod

        fake = paths_mod.Paths(
            root=tmp_path, assets=tmp_path / "assets", models=tmp_path / "models",
            data=tmp_path / "data", runs=tmp_path / "runs",
            cache=tmp_path / "cache", logs=tmp_path / "logs", temp=tmp_path / "temp",
        )
        monkeypatch.setattr(mute_asset, "get_paths", lambda: fake)

        run = tmp_path / "run1"
        _make_run(run, n_files=3)
        out = write_filelist(run, "v2", f0=True, speaker_id=0)
        text = out.read_text()
        lines = text.splitlines()
        mute = [l for l in lines if "mute" in l]
        assert len(mute) == 2, "filelist must carry the official 2-line mute dose"
        assert len(lines) == 5, "3 real entries + 2 mute"
        # every referenced file exists (no dangling paths in the manifest)
        for l in lines:
            for p in l.split("|")[:-1]:
                assert Path(p).is_file(), p

    def test_deterministic_shuffle_seed(self, tmp_path, monkeypatch):
        from rvc_platform.infrastructure import paths as paths_mod

        fake = paths_mod.Paths(
            root=tmp_path, assets=tmp_path / "assets", models=tmp_path / "models",
            data=tmp_path / "data", runs=tmp_path / "runs",
            cache=tmp_path / "cache", logs=tmp_path / "logs", temp=tmp_path / "temp",
        )
        monkeypatch.setattr(mute_asset, "get_paths", lambda: fake)

        run_a, run_b = tmp_path / "a", tmp_path / "b"
        _make_run(run_a, 2)
        _make_run(run_b, 2)
        ta = write_filelist(run_a, "v2", True).read_text()
        tb = write_filelist(run_b, "v2", True).read_text()
        # absolute paths necessarily differ across run dirs; the SEEDED
        # permutation must be identical — compare relative order/stems.
        stems_a = [Path(l.split("|")[0]).name for l in ta.splitlines()]
        stems_b = [Path(l.split("|")[0]).name for l in tb.splitlines()]
        assert stems_a == stems_b, "seeded shuffle must be reproducible"
        assert len(stems_a) == 4  # 2 real + 2 mute
        assert stems_a.count("mute48k.wav") == 2
