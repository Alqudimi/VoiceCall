"""Unit tests for the background Hugging Face publisher (fake hub, no network)."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rvc_platform.training import huggingface as hf


class FakeCommitOperationAdd:
    def __init__(self, path_in_repo, path_or_fileobj):
        self.path_in_repo = path_in_repo
        self.path_or_fileobj = path_or_fileobj


class FakeHfApi:
    """Records create_repo / repo_info / create_commit calls."""

    instances: list["FakeHfApi"] = []

    def __init__(self, token=None, *, fail_first_n_commits=0, public=False):
        self.token = token
        self.created: list[dict] = []
        self.commits: list[dict] = []
        self.fail_first_n_commits = fail_first_n_commits
        self.public = public
        FakeHfApi.instances.append(self)

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def repo_info(self, **kwargs):
        return SimpleNamespace(private=not self.public)

    def create_commit(self, **kwargs):
        if self.fail_first_n_commits > 0:
            self.fail_first_n_commits -= 1
            raise ConnectionError("simulated network hiccup")
        self.commits.append(kwargs)


@pytest.fixture()
def fake_hub(monkeypatch):
    FakeHfApi.instances = []
    fake_module = SimpleNamespace(HfApi=FakeHfApi,
                                  CommitOperationAdd=FakeCommitOperationAdd)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    monkeypatch.setattr(hf, "RETRY_BACKOFF_SECONDS", 0.01)
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    return FakeHfApi


def _run_dir_with_export(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    exports = run_dir / "exports"
    exports.mkdir(parents=True)
    model = exports / "voice_e10_s100.pth"
    model.write_bytes(b"weights")
    (run_dir / "added_IVF1.index").write_bytes(b"index")
    return run_dir, model


def test_publisher_uploads_best_atomically(fake_hub, tmp_path):
    run_dir, model = _run_dir_with_export(tmp_path)
    events: list[dict] = []
    publisher = hf.HfBackgroundPublisher("owner/private-voice",
                                         on_event=events.append)
    publisher.submit_best(model, run_dir / "added_IVF1.index", {
        "metric": "val_mel", "score": 0.4242, "epoch": 10,
        "global_step": 100, "export": model.name,
    })
    summary = publisher.close(timeout=10)
    assert summary["uploaded"] == 1 and summary["failed"] == 0

    api = FakeHfApi.instances[-1]
    assert api.created[0]["private"] is True
    assert api.created[0]["exist_ok"] is True
    (commit,) = api.commits
    ops = commit["operations"]
    paths = [op.path_in_repo for op in ops]
    assert paths == ["best/model.pth", "best/index.index", "best/metadata.json"]
    assert commit["repo_id"] == "owner/private-voice"
    assert "Best model: epoch 10" in commit["commit_message"]
    # metadata rides along as bytes in the same atomic commit
    metadata_op = ops[2]
    assert b"val_mel" in metadata_op.path_or_fileobj

    kinds = [e["topic"] for e in events]
    assert "training.hf_best_uploaded" in kinds
    best_event = next(e for e in events if e["topic"] == "training.hf_best_uploaded")
    assert best_event["best"]["epoch"] == 10
    assert publisher.last_best_key == "10:100"


def test_publisher_retries_transient_failures(fake_hub, tmp_path):
    run_dir, model = _run_dir_with_export(tmp_path)
    FakeHfApi.next_failures = 2
    events: list[dict] = []

    class FlakyApi(FakeHfApi):
        def __init__(self, token=None, **kwargs):
            super().__init__(token,
                             fail_first_n_commits=FakeHfApi.next_failures,
                             **kwargs)

    FakeHfApi.instances = []
    fake = sys.modules["huggingface_hub"]
    fake.HfApi = FlakyApi

    publisher = hf.HfBackgroundPublisher("owner/flaky", on_event=events.append)
    publisher.submit_export(run_dir, model)
    summary = publisher.close(timeout=10)
    assert summary["uploaded"] == 1 and summary["failed"] == 0, \
        "two transient failures must be retried, not fatal"
    api = FakeHfApi.instances[-1]
    assert len(api.commits) == 1
    paths = [op.path_in_repo for op in api.commits[0]["operations"]]
    assert "models/voice_e10_s100.pth" in paths
    assert "indexes/added_IVF1.index" in paths


def test_publisher_coalesces_superseded_best_jobs(fake_hub, tmp_path):
    run_dir, model = _run_dir_with_export(tmp_path)
    other = run_dir / "exports" / "voice_e20_s200.pth"
    other.write_bytes(b"weights-newer")

    # block the writer so jobs queue up instead of executing
    gate_release = __import__("threading").Event()

    class GatedApi(FakeHfApi):
        def create_commit(self, **kwargs):
            gate_release.wait(timeout=10)
            return super().create_commit(**kwargs)

    FakeHfApi.instances = []
    sys.modules["huggingface_hub"].HfApi = GatedApi

    publisher = hf.HfBackgroundPublisher("owner/coalesce")
    publisher.submit_best(model, None, {"epoch": 10, "global_step": 100})
    publisher.submit_best(model, None, {"epoch": 10, "global_step": 100})
    time.sleep(0.05)   # let the first job start (blocked in the gate)
    publisher.submit_best(other, None, {"epoch": 20, "global_step": 200})
    gate_release.set()
    summary = publisher.close(timeout=10)
    api = FakeHfApi.instances[-1]
    committed_epochs = []
    for commit in api.commits:
        for op in commit["operations"]:
            if op.path_in_repo == "best/metadata.json":
                committed_epochs.append(op.path_or_fileobj)
    assert summary["failed"] == 0
    assert len(committed_epochs) >= 1
    # the last commit always carries the newest best (epoch 20)
    assert b"20" in committed_epochs[-1]
    assert publisher.last_best_key == "20:200"


def test_publisher_reports_permanent_failure_without_raising(fake_hub, tmp_path):
    run_dir, model = _run_dir_with_export(tmp_path)
    events: list[dict] = []

    class DeadApi(FakeHfApi):
        def create_commit(self, **kwargs):
            raise ConnectionError("network down for good")

    FakeHfApi.instances = []
    sys.modules["huggingface_hub"].HfApi = DeadApi
    monkey_attempts = 1  # shrink retries for speed
    import rvc_platform.training.huggingface as mod
    original = mod.UPLOAD_ATTEMPTS
    mod.UPLOAD_ATTEMPTS = monkey_attempts
    try:
        publisher = hf.HfBackgroundPublisher("owner/dead", on_event=events.append)
        publisher.submit_export(run_dir, model)
        summary = publisher.close(timeout=10)
    finally:
        mod.UPLOAD_ATTEMPTS = original
    assert summary["failed"] == 1
    kinds = [e["topic"] for e in events]
    assert "training.hf_upload_failed" in kinds
    assert "network down for good" in events[-1]["error"]


def test_publisher_status_reflects_state(fake_hub, tmp_path):
    run_dir, model = _run_dir_with_export(tmp_path)
    publisher = hf.HfBackgroundPublisher("owner/status")
    publisher.submit_export(run_dir, model)
    publisher.close(timeout=10)
    status = publisher.status()
    assert status["uploaded"] == 1 and status["pending"] == 0
    assert status["repo_id"] == "owner/status"
    # a closed publisher refuses new work
    with pytest.raises(RuntimeError):
        publisher.submit_export(run_dir, model)


def test_sync_publish_run_uses_atomic_commit(fake_hub, tmp_path):
    run_dir, model = _run_dir_with_export(tmp_path)
    (run_dir / "exports" / "run.pth").write_bytes(b"final-weights")
    url = hf.publish_run(run_dir, "owner/private-voice")
    assert url == "https://huggingface.co/owner/private-voice"
    api = FakeHfApi.instances[-1]
    (commit,) = api.commits
    # the final-named export wins over the epoch-named one
    paths = [op.path_in_repo for op in commit["operations"]]
    assert paths[0] == "models/run.pth"
