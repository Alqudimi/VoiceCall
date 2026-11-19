"""Unit tests for best-model tracking (tracker, split derivation, artifacts)."""
from __future__ import annotations

import json

import torch

from rvc_platform.training.best_model import (
    BEST_STATE_NAME,
    BestModelTracker,
    derive_validation_split,
    read_best_state,
    write_best_artifacts,
)


class _Hps:
    """Minimal hps stand-in for write_best_artifacts."""

    class data:
        filter_length = 1025
        n_mel_channels = 125
        sampling_rate = 40_000
        mel_fmin = 0
        mel_fmax = None
        inter_channels = 192
        hidden_channels = 192

    class model:
        inter_channels = 192
        hidden_channels = 192
        filter_channels = 768
        n_heads = 2
        n_layers = 6
        kernel_size = 3
        p_dropout = 0
        resblock = "1"
        resblock_kernel_sizes = [3, 7, 11]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        upsample_rates = [10, 10, 2, 2]
        upsample_initial_channel = 512
        upsample_kernel_sizes = [16, 16, 4, 4]
        spk_embed_dim = 109
        gin_channels = 256

    version = "v2"
    sample_rate = 40_000
    if_f0 = 1


def test_tracker_min_mode_records_improvements(tmp_path):
    tracker = BestModelTracker(tmp_path / BEST_STATE_NAME, metric="val_mel")
    assert tracker.update({"val_mel": 0.9}, epoch=1, global_step=10) is not None
    # worse score -> no new best
    assert tracker.update({"val_mel": 0.95}, epoch=2, global_step=20) is None
    # better score -> new best
    improved = tracker.update({"val_mel": 0.8}, epoch=3, global_step=30)
    assert improved is not None and improved.epoch == 3
    # nan/inf and missing metric are ignored
    assert tracker.update({"val_mel": float("nan")}, epoch=4, global_step=40) is None
    assert tracker.update({"other": 1.0}, epoch=5, global_step=50) is None
    assert tracker.best.score == 0.8 and tracker.best.epoch == 3


def test_tracker_persisted_best_survives_reload(tmp_path):
    path = tmp_path / BEST_STATE_NAME
    first = BestModelTracker(path, metric="val_mel")
    first.update({"val_mel": 0.7}, epoch=9, global_step=90)
    # simulate the persisted write
    path.write_text(json.dumps(first.best.to_dict()), encoding="utf-8")

    resumed = BestModelTracker(path, metric="val_mel")
    assert resumed.load() is True
    assert resumed.best.epoch == 9 and resumed.best.score == 0.7
    # a resumed run with worse epochs can never regress the bar
    assert resumed.update({"val_mel": 0.88}, epoch=10, global_step=100) is None
    assert resumed.update({"val_mel": 0.69}, epoch=11, global_step=110) is not None


def test_split_derivation_is_deterministic_and_pinned(tmp_path):
    filelist = tmp_path / "filelist.txt"
    lines = [f"/x/{i}.wav|/x/{i}.npy|/x/{i}.f0|/x/{i}.nsf|0" for i in range(20)]
    filelist.write_text("\n".join(lines), encoding="utf-8")

    first = derive_validation_split(filelist, 0.2)
    assert first is not None and first.rederived is True
    assert first.n_val == 4 and first.n_train == 16

    train_lines = first.train_path.read_text(encoding="utf-8").splitlines()
    val_lines = first.val_path.read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 16 and len(val_lines) == 4
    assert set(train_lines) | set(val_lines) == set(lines), "split must be disjoint+total"

    # second derivation with identical inputs reuses the pinned files
    second = derive_validation_split(filelist, 0.2)
    assert second is not None and second.rederived is False
    assert second.train_path.read_text(encoding="utf-8").splitlines() == train_lines

    # same content, different fraction -> re-derive
    third = derive_validation_split(filelist, 0.1)
    assert third is not None and third.rederived is True
    assert third.n_val == 2

    # changed filelist content -> re-derive (caller resets the best bar)
    filelist.write_text("\n".join(lines + ["extra|extra|extra|extra|0"]), encoding="utf-8")
    fourth = derive_validation_split(filelist, 0.1)
    assert fourth is not None and fourth.rederived is True


def test_split_refuses_impossible_cases(tmp_path):
    filelist = tmp_path / "filelist.txt"
    filelist.write_text("only|one|line|here|0", encoding="utf-8")
    assert derive_validation_split(filelist, 0.2) is None
    assert derive_validation_split(tmp_path / "missing.txt", 0.2) is None
    filelist.write_text("a|b\nc|d", encoding="utf-8")
    assert derive_validation_split(filelist, 0.0) is None


def test_write_best_artifacts_and_pruning(tmp_path):
    exports = tmp_path / "exports"
    exports.mkdir()
    snapshot = {"weight": torch.zeros(2, 2)}
    best_json = tmp_path / "checkpoints" / BEST_STATE_NAME
    best_json.parent.mkdir()
    metadata = {
        "metric": "val_mel", "score": 0.42, "epoch": 5, "global_step": 50,
        "improved_at": 1.0, "experiment": "exp", "export": "",
    }
    # create three successive "best" exports, newest wins the prune
    for epoch, step in ((1, 10), (3, 30), (5, 50)):
        metadata = {**metadata, "epoch": epoch, "global_step": step,
                    "export": f"exp_best_e{epoch}_s{step}.pth"}
        out = exports / metadata["export"]
        out.write_bytes(b"placeholder")  # pre-existing file gets overwritten
        write_best_artifacts(snapshot, _Hps(), metadata, out, best_json, keep=2)
        # mtime ordering for prune tests
        import os
        os.utime(out, (epoch * 1000, epoch * 1000))

    remaining = sorted(exports.glob("exp_best_*.pth"))
    assert len(remaining) == 2, "keep=2 must prune the oldest best exports"
    assert remaining[-1].name == "exp_best_e5_s50.pth"
    persisted = json.loads(best_json.read_text(encoding="utf-8"))
    assert persisted["epoch"] == 5 and persisted["metric"] == "val_mel"
    assert read_best_state(tmp_path) == persisted
    assert read_best_state(tmp_path / "nowhere") is None
