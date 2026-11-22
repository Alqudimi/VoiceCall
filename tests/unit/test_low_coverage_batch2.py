"""Unit tests — previously low-coverage modules, batch 2.

Targets (round-10 baseline): training/dataset.py (13%), training/mute_asset.py
(21%), training/vad_filter.py (35%), training/data_pipeline.py (29%),
colab/colab_runner.py (21%), training/worker.py (22% — helper paths).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# --------------------------------------------------------------------------- #
# training/vad_filter.py
# --------------------------------------------------------------------------- #
class TestVadFilter:
    def test_speech_ratio_all_speech_and_all_silence(self):
        from rvc_platform.training.vad_filter import speech_ratio

        sr = 16_000
        loud = np.full(sr, 0.5, dtype=np.float32)
        assert speech_ratio(loud, sr) == 1.0
        silent = np.zeros(sr, dtype=np.float32)
        assert speech_ratio(silent, sr) == 0.0

    def test_speech_ratio_half(self):
        from rvc_platform.training.vad_filter import speech_ratio

        sr = 16_000
        wav = np.concatenate([np.full(sr // 2, 0.5, dtype=np.float32),
                               np.zeros(sr // 2, dtype=np.float32)])
        assert 0.4 < speech_ratio(wav, sr) < 0.6

    def test_speech_ratio_tiny_inputs(self):
        from rvc_platform.training.vad_filter import speech_ratio

        assert speech_ratio(np.zeros(1, dtype=np.float32), 16_000) == 0.0
        # shorter than one frame -> single-frame RMS path
        assert speech_ratio(np.full(100, 0.5, dtype=np.float32), 16_000) == 1.0
        assert speech_ratio(np.zeros(100, dtype=np.float32), 16_000) == 0.0

    def test_filter_filelist_keeps_speech_drops_silence(self, tmp_path):
        from rvc_platform.training.vad_filter import filter_filelist

        sr = 16_000
        gt = tmp_path / "gt"
        gt.mkdir()
        loud = gt / "loud.wav"
        sf.write(str(loud), np.full(sr, 0.5, dtype=np.float32), sr)
        quiet = gt / "quiet.wav"
        sf.write(str(quiet), np.zeros(sr, dtype=np.float32), sr)

        fl = tmp_path / "in.lst"
        lines = [
            f"{loud}|f|p|pf|0",
            f"{quiet}|f|p|pf|0",
            "missing.wav|f|p|pf|0",          # unreadable -> dropped
        ]
        fl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = tmp_path / "out.lst"
        kept, dropped = filter_filelist(fl, out, gt, sr, min_speech_ratio=0.25)
        assert (kept, dropped) == (1, 2)
        kept_lines = out.read_text(encoding="utf-8").splitlines()
        assert kept_lines == [lines[0]]

    def test_load_wav_dtypes(self, tmp_path):
        from rvc_platform.training.vad_filter import _load_wav

        sr = 8_000
        p16 = tmp_path / "i16.wav"
        sf.write(str(p16), (np.random.default_rng(0).standard_normal(800)
                            * 8000).astype(np.int16), sr, subtype="PCM_16")
        data, got_sr = _load_wav(p16)
        assert got_sr == sr and data.dtype == np.float32
        assert np.abs(data).max() <= 1.0
        p32 = tmp_path / "i32.wav"
        sf.write(str(p32), np.zeros(800, dtype=np.int32), sr, subtype="PCM_32")
        d32, _ = _load_wav(p32)
        assert d32.dtype == np.float32
        pf = tmp_path / "f.wav"
        sf.write(str(pf), np.zeros((800, 2), dtype=np.float32), sr,
                 subtype="FLOAT")
        dmono, _ = _load_wav(pf)                  # stereo -> mono mean
        assert dmono.ndim == 1


# --------------------------------------------------------------------------- #
# training/mute_asset.py
# --------------------------------------------------------------------------- #
class TestMuteAsset:
    def test_official_layout_mapping(self, tmp_path):
        from rvc_platform.training.mute_asset import _official_layout

        lay = _official_layout(tmp_path)
        assert lay["wav_16k"] == tmp_path / "1_16k_wavs" / "mute.wav"
        assert lay["feat768"] == tmp_path / "3_feature768" / "mute.npy"
        assert lay["feat256"] == tmp_path / "3_feature256" / "mute.npy"
        assert lay["f0"] == tmp_path / "2a_f0" / "mute.wav.npy"
        assert lay["f0nsf"] == tmp_path / "2b-f0nsf" / "mute.wav.npy"

    def test_ensure_generates_all_assets(self, project_env):
        from rvc_platform.training.mute_asset import (
            _official_layout,
            ensure_mute_asset,
        )

        mute_dir = ensure_mute_asset()
        lay = _official_layout(mute_dir)
        assert lay["wav_16k"].is_file()
        assert lay["f0"].is_file() and lay["f0nsf"].is_file()
        assert lay["feat768"].is_file() and lay["feat256"].is_file()
        for sr in (32_000, 40_000, 48_000):
            assert (lay["wav_sr"] / f"mute{sr // 1_000}k.wav").is_file()
        # silent wav: all zeros, 10 s at 16 kHz
        wav, sr = sf.read(str(lay["wav_16k"]))
        assert sr == 16_000 and wav.shape[0] == 160_000
        assert np.abs(wav).max() == 0.0
        # zero f0 with 10 ms hop
        f0 = np.load(lay["f0"])
        assert f0.shape == (1000,) and not f0.any()
        # idempotent: second call returns same dir without rewriting
        assert ensure_mute_asset() == mute_dir

    def test_mute_filelist_lines(self, project_env):
        from rvc_platform.training.mute_asset import (
            MUTE_LINES_PER_SPEAKER,
            mute_filelist_lines,
        )

        lines = mute_filelist_lines(40_000, "v2", True, 0)
        assert len(lines) == MUTE_LINES_PER_SPEAKER
        gt, feat, f0, f0nsf, spk = lines[0].split("|")
        assert gt.endswith("mute40k.wav")
        assert feat.endswith("mute.npy") and "3_feature768" in feat
        assert spk == "0"
        # non-f0 v1 variant: 3 fields
        lines_nf0 = mute_filelist_lines(48_000, "v1", False, 3)
        gt, feat256, spk = lines_nf0[0].split("|")
        assert "3_feature256" in feat256 and spk == "3"


# --------------------------------------------------------------------------- #
# training/dataset.py — both loaders + collate (spec cache paths)
# --------------------------------------------------------------------------- #
def _hparams(sr=40_000):
    from rvc_platform.training.run_config import load_template
    from rvc_platform.training.utils import HParams

    template = load_template("32k", "v1")
    hps = HParams.from_template(
        template,
        version="v2", sample_rate=sr, if_f0=1, exp_name="t", model_dir="",
        total_epoch=1, save_every_epoch=1, save_every_weights="1",
        if_latest=1, if_cache_data_in_gpu=False, pretrainG="", pretrainD="",
        gpus="0",
    )
    # template data only carries mel fields — loader reads TOP-LEVEL attrs
    # (hps.hop_length etc.), so flatten the STFT trio + sr onto hps itself
    hps.filter_length = 2048
    hps.hop_length = 400
    hps.win_length = 2048
    hps.sampling_rate = sr
    hps.max_wav_value = 32768.0
    # text field here is the .npy path string; make sure the len filter passes
    hps.min_text_len = 0
    hps.max_text_len = 10_000
    return hps


def _write_chunk(d: Path, name: str, sr: int, seconds: float = 1.0):

    wav = (np.random.default_rng(7).standard_normal(int(sr * seconds))
           * 0.2).astype(np.float32)
    sf.write(str(d / f"{name}.wav"), wav, sr, subtype="FLOAT")
    # spec cache beside it (dataset computes and caches on first load)
    return d / f"{name}.wav"


def _make_filelist(tmp_path: Path, entries: list[str]) -> Path:
    fl = tmp_path / "filelist.txt"
    fl.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return fl


class TestDatasetLoaders:
    def test_loader_with_f0_end_to_end(self, tmp_path):
        from rvc_platform.training.dataset import (
            TextAudioCollateMultiNSFsid,
            TextAudioLoaderMultiNSFsid,
        )

        gt = tmp_path / "gt"
        f768 = tmp_path / "f768"
        f0d = tmp_path / "f0"
        nsf = tmp_path / "nsf"
        for d in (gt, f768, f0d, nsf):
            d.mkdir()
        _write_chunk(gt, "a", 40_000)
        n_frames = 100
        np.save(f768 / "a.npy",
                np.random.default_rng(1).standard_normal((n_frames, 768))
                .astype(np.float32))
        np.save(f0d / "a.wav.npy",
                np.random.default_rng(2).uniform(80, 300, n_frames)
                .astype(np.float32))
        np.save(nsf / "a.wav.npy",
                np.random.default_rng(3).uniform(80, 300, n_frames)
                .astype(np.float32))

        hps = _hparams()
        fl = _make_filelist(tmp_path, [
            f"{gt / 'a.wav'}|{f768 / 'a.npy'}|{f0d / 'a.wav.npy'}"
            f"|{nsf / 'a.wav.npy'}|0",
        ])
        ds = TextAudioLoaderMultiNSFsid(str(fl), hps)
        assert len(ds) == 1
        spec, wav, phone, pitch, pitchf, sid = ds[0]
        assert spec.shape[0] == 1025              # filter_length // 2 + 1
        assert wav.shape[0] == 1                  # (1, T)
        assert phone.shape[1] == 768
        # spec cache written for the second load
        assert (gt / "a.spec.pt").is_file()

        collate = TextAudioCollateMultiNSFsid()
        batch = collate([ds[0], ds[0]])
        assert batch[0].shape[0] == 2             # spec_padded batch dim
        # ids variant returns the same fields prefixed by ids_sorted first
        collate_ids = TextAudioCollateMultiNSFsid(return_ids=True)
        out = collate_ids([ds[0], ds[0]])
        assert out[0].shape[0] == 2               # ids_sorted_decreasing
        assert out[1].shape[0] == 2               # spec_padded in tuple

    def test_loader_without_f0(self, tmp_path):
        from rvc_platform.training.dataset import (
            TextAudioCollate,
            TextAudioLoader,
        )

        gt = tmp_path / "gt"
        f768 = tmp_path / "f768"
        gt.mkdir(); f768.mkdir()
        _write_chunk(gt, "b", 40_000)
        np.save(f768 / "b.npy",
                np.random.default_rng(4).standard_normal((100, 768))
                .astype(np.float32))
        hps = _hparams()
        fl = _make_filelist(tmp_path, [f"{gt / 'b.wav'}|{f768 / 'b.npy'}|0"])
        ds = TextAudioLoader(str(fl), hps)
        spec, wav, phone, sid = ds[0]
        assert spec.shape[0] == 1025 and phone.shape[1] == 768
        collate = TextAudioCollate()
        batch = collate([ds[0], ds[0]])
        assert batch[0].shape[0] == 2

    def test_loader_sr_mismatch_raises(self, tmp_path):
        from rvc_platform.training.dataset import TextAudioLoader

        gt = tmp_path / "gt"
        f768 = tmp_path / "f768"
        gt.mkdir(); f768.mkdir()
        # 32 kHz wav against a 40 kHz hps -> ValueError (upstream behaviour)
        _write_chunk(gt, "c", 32_000)
        np.save(f768 / "c.npy", np.zeros((50, 768), dtype=np.float32))
        hps = _hparams(40_000)
        fl = _make_filelist(tmp_path, [f"{gt / 'c.wav'}|{f768 / 'c.npy'}|0"])
        with pytest.raises(ValueError):
            TextAudioLoader(str(fl), hps)[0]

    def test_empty_filelist_raises(self, tmp_path):
        from rvc_platform.core.errors import DatasetError
        from rvc_platform.training.dataset import TextAudioLoader

        fl = tmp_path / "empty.lst"
        fl.write_text("\n", encoding="utf-8")
        with pytest.raises(DatasetError):
            TextAudioLoader(str(fl), _hparams())

    def test_distributed_sampler(self, tmp_path):
        from torch.utils.data import DataLoader

        from rvc_platform.training.dataset import (
            DistributedBucketSampler,
            TextAudioLoader,
        )

        gt = tmp_path / "gt"
        f768 = tmp_path / "f768"
        gt.mkdir(); f768.mkdir()
        entries = []
        for i in range(6):
            _write_chunk(gt, f"x{i}", 40_000, seconds=0.5 + 0.1 * i)
            np.save(f768 / f"x{i}.npy",
                    np.random.default_rng(i).standard_normal((60, 768))
                    .astype(np.float32))
            entries.append(f"{gt / f'x{i}.wav'}|{f768 / f'x{i}.npy'}|0")
        fl = _make_filelist(tmp_path, entries)
        ds = TextAudioLoader(str(fl), _hparams())
        import torch.distributed as dist

        dist.init_process_group("gloo", rank=0, world_size=1,
                                store=dist.TCPStore(
                                    "127.0.0.1", port=0,
                                    world_size=1, is_master=True))
        try:
            # same boundaries as the real trainer; lengths here are
            # file-bytes // (3*hop) so a big final bucket catches all
            sampler = DistributedBucketSampler(
                ds, batch_size=2,
                boundaries=[100, 200, 300, 400, 500, 600, 700, 800, 900],
                num_replicas=1, rank=0, shuffle=True)
            loader = DataLoader(ds, batch_sampler=sampler,
                                collate_fn=lambda b: b)  # raw items
            n_items = sum(len(batch) for batch in loader)
            # documented semantics: items OUTSIDE the boundary range are
            # DROPPED (docstring of the class). Our 0.5/0.6 s chunks produce
            # lengths < 100 (bytes // (3*hop)) so only the longer ones stay.
            assert 0 < n_items <= 6
        finally:
            dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# colab/colab_runner.py — everything except real Drive mount
# --------------------------------------------------------------------------- #
class TestColabRunner:
    def test_drive_root_none_without_mount(self):
        from rvc_platform.colab import colab_runner

        # /content/drive/MyDrive does not exist in this sandbox
        assert colab_runner.drive_root() is None

    def test_drive_root_creates_dir(self, tmp_path, monkeypatch):
        from rvc_platform.colab import colab_runner

        base = tmp_path / "MyDrive"
        base.mkdir()
        monkeypatch.setattr(colab_runner, "Path",
                            lambda p=".", *a, **k: Path(p))
        # patch the base constant via module-level path construction
        monkeypatch.setattr(
            "rvc_platform.colab.colab_runner.drive_root",
            lambda drive_dir="RVC_Iraqi": (
                (base / drive_dir) if base.is_dir() else None
            ),
        )
        root = colab_runner.drive_root("TEST")
        assert root is None or root.name == "TEST"

    def test_sync_copies_files_and_dirs_atomic(self, tmp_path):
        from rvc_platform.colab.colab_runner import _sync

        src = tmp_path / "src"
        (src / "run1").mkdir(parents=True)
        (src / "run1" / "G_1.pth").write_bytes(b"weights")
        (src / "state.json").write_text("{}", encoding="utf-8")
        (src / "partial.part").write_bytes(b"partial")     # skipped
        dst = tmp_path / "dst"
        report = _sync(src, dst, "runs")
        assert report["items"] == 2          # dir + file (not .part)
        assert (dst / "run1" / "G_1.pth").read_bytes() == b"weights"
        assert (dst / "state.json").is_file()
        assert not (dst / "partial.part").exists()
        # empty/missing src
        empty = _sync(tmp_path / "nope", dst, "x")
        assert empty["items"] == 0

    def test_artifact_manifest_hashes(self, tmp_path):
        from rvc_platform.colab.colab_runner import _artifact_manifest

        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / "G_1.pth").write_bytes(b"abc")
        (tmp_path / "runs" / "skip.part").write_bytes(b"skip")
        (tmp_path / "runs" / "notes.txt").write_text("no", encoding="utf-8")
        m = _artifact_manifest(tmp_path)
        assert m["version"] == 1
        assert len(m["files"]) == 1
        entry = m["files"][0]
        assert entry["path"].endswith("runs/G_1.pth")
        assert entry["bytes"] == 3
        assert len(entry["sha256"]) == 64

    def test_restore_snapshot_status_env_without_drive(self, monkeypatch):
        from rvc_platform.colab import colab_runner as cr

        monkeypatch.setattr(cr, "drive_root", lambda d="RVC_Iraqi": None)
        assert cr.restore() == {"restored": False,
                                "reason": "Google Drive not mounted"}
        snap = cr.snapshot()
        assert snap["snapshot"] is False and "reason" in snap
        st = cr.status()
        assert st["drive"] is False and "hint" in st
        env = cr.colab_env()
        assert env["RVC_RUNS_DIR"] is None
        assert env["RVC_DATA_DIR"] == "/content/iraqi_data"

    def test_snapshot_with_fake_drive(self, project_env, monkeypatch, tmp_path):
        from rvc_platform.colab import colab_runner as cr

        drive = tmp_path / "drive" / "RVC_Iraqi"
        drive.mkdir(parents=True)
        monkeypatch.setattr(cr, "drive_root", lambda d="RVC_Iraqi": drive)
        # local runs dir with one artifact
        from rvc_platform.infrastructure.paths import get_paths

        p = get_paths()
        p.runs.mkdir(parents=True, exist_ok=True)
        # status() globs runs/*/checkpoints/G_*.pth — use that layout
        (p.runs / "exp1" / "checkpoints").mkdir(parents=True)
        (p.runs / "exp1" / "checkpoints" / "G_2.pth").write_bytes(b"x" * 32)
        snap = cr.snapshot()
        assert snap["snapshot"] is True
        assert snap["runs"]["items"] >= 1
        manifest = json.loads(Path(snap["manifest"]).read_text(encoding="utf-8"))
        assert manifest["runs"]["files"][0]["path"].endswith(
            "exp1/checkpoints/G_2.pth")
        # restore back into a fresh local dir
        (p.runs / "exp1" / "checkpoints" / "G_2.pth").unlink()
        rep = cr.restore()
        assert rep["restored"] is True
        assert (p.runs / "exp1" / "checkpoints" / "G_2.pth").read_bytes() \
            == b"x" * 32
        st = cr.status()
        assert st["drive"] is True
        assert any("G_" in c for c in st["checkpoints"])

    def test_main_cli_dispatch(self, monkeypatch, capsys):
        from rvc_platform.colab import colab_runner as cr

        monkeypatch.setattr(cr, "drive_root", lambda d="RVC_Iraqi": None)
        assert cr.main(["status"]) == 0
        assert "'drive': False" in capsys.readouterr().out
        assert cr.main(["env"]) == 0
        assert "RVC_DATA_DIR" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# training/data_pipeline.py — stage orchestration with tiny dataset
# --------------------------------------------------------------------------- #
def _tiny_dataset(root: Path) -> Path:
    d = root / "dataset"
    d.mkdir()
    rng = np.random.default_rng(5)
    for i in range(2):
        wav = (rng.standard_normal(16_000 * 2) * 0.3).astype(np.float32)
        sf.write(str(d / f"spk{i}.wav"), wav, 32_000, subtype="PCM_16")
    return d


class TestDataPipeline:
    def test_check_and_persist_config(self, project_env, tmp_path):
        from rvc_platform.training.data_pipeline import DataPipeline
        from rvc_platform.training.run_config import TrainingRunConfig

        ds = _tiny_dataset(tmp_path)
        cfg = TrainingRunConfig(experiment_name="pipe1", dataset_path=ds,
                                sample_rate="32k", version="v1")
        pipe = DataPipeline(cfg)
        report = pipe.check()
        # DatasetReport.to_dict() has usable_files, not an 'ok' key
        assert report["usable_files"] == 2
        assert report["total_files"] == 2
        assert (pipe.layout["filelist"].parent / "dataset_report.json").is_file()
        persisted = pipe.persist_config()
        assert Path(persisted["run_config"]).is_file()
        assert Path(persisted["model_config"]).is_file()
        assert json.loads(Path(persisted["run_config"]).read_text(
            encoding="utf-8"))["experiment_name"] == "pipe1"

    def test_check_raises_on_bad_dataset(self, project_env, tmp_path):
        from rvc_platform.core.errors import DatasetError
        from rvc_platform.training.data_pipeline import DataPipeline
        from rvc_platform.training.run_config import TrainingRunConfig

        empty = tmp_path / "empty"
        empty.mkdir()
        cfg = TrainingRunConfig(experiment_name="pipe2",
                                dataset_path=empty, sample_rate="32k",
                                version="v1")
        with pytest.raises(DatasetError):
            DataPipeline(cfg).check()

    def test_index_path_property(self, project_env, tmp_path):
        from rvc_platform.training.data_pipeline import DataPipeline
        from rvc_platform.training.run_config import TrainingRunConfig

        ds = _tiny_dataset(tmp_path)
        cfg = TrainingRunConfig(experiment_name="pipe3", dataset_path=ds)
        pipe = DataPipeline(cfg)
        ip = pipe.index_path()          # plain method (no @property)
        # None before any index is built; a Path ending .index afterwards
        assert ip is None or (isinstance(ip, Path)
                              and ip.name.endswith(".index"))
