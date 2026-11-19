"""Unit tests for the background I/O worker (ordering, flush, chaining)."""
from __future__ import annotations

import threading
import time

import torch

from rvc_platform.training.background_io import BackgroundIOWorker, snapshot_to_cpu


def test_jobs_run_in_strict_fifo_order():
    worker = BackgroundIOWorker(name="io-test-order")
    order: list[int] = []
    for i in range(20):
        worker.submit(f"j{i}", order.append, i)
    assert worker.wait_idle(timeout=10)
    assert order == list(range(20))
    assert worker.close(timeout=10)


def test_submit_never_runs_inline():
    """The whole point: submit() returns before the job's I/O happens."""
    started = threading.Event()
    release = threading.Event()

    def slow_job():
        started.set()
        release.wait(timeout=10)

    worker = BackgroundIOWorker(name="io-test-inline")
    t0 = time.time()
    worker.submit("slow", slow_job)
    submit_duration = time.time() - t0
    assert started.wait(timeout=5), "job should start on the worker thread"
    assert submit_duration < 0.5, "submit() must not block on the job body"
    release.set()
    assert worker.wait_idle(timeout=5)
    worker.close(timeout=5)


def test_on_done_chaining_and_error_callbacks():
    worker = BackgroundIOWorker(name="io-test-chain")
    results: list[str] = []
    errors: list[str] = []
    done = threading.Event()

    def chain(value):
        results.append(f"done:{value}")
        done.set()

    def fail():
        raise RuntimeError("boom")

    def on_error(exc):
        errors.append(str(exc))

    worker.submit("ok", lambda: "value", on_done=chain)
    worker.submit("bad", fail, on_error=on_error)
    assert done.wait(timeout=5)
    assert worker.wait_idle(timeout=5)
    assert results == ["done:value"]
    assert errors == ["boom"]
    assert worker.last_error is not None and "boom" in worker.last_error
    # the writer survives a failing job and keeps processing
    worker.submit("after", results.append, "after")
    assert worker.wait_idle(timeout=5)
    assert "after" in results
    worker.close(timeout=5)


def test_close_drains_pending_jobs():
    worker = BackgroundIOWorker(name="io-test-drain")
    seen: list[int] = []
    for i in range(50):
        worker.submit(f"j{i}", seen.append, i)
    assert worker.close(timeout=15), "close must drain the queued jobs"
    assert seen == list(range(50))
    try:
        worker.submit("late", lambda: None)
        raise AssertionError("submit after close must raise")
    except RuntimeError:
        pass


def test_snapshot_to_cpu_copies_are_private():
    model = torch.nn.Linear(4, 2)
    snap = snapshot_to_cpu(model.state_dict())
    before = snap["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1.0)          # mutate the live tensor in place
    assert torch.equal(snap["weight"], before), "snapshot must not alias live memory"

    # nested structures (optimizer state) also snapshotted
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(3, 4)).sum().backward()   # populate optimizer state
    optim.step()
    snap_opt = snapshot_to_cpu(optim.state_dict())
    live_exp_avg = optim.state_dict()["state"][0]["exp_avg"]
    snap_exp_avg = snap_opt["state"][0]["exp_avg"]
    assert snap_exp_avg.data_ptr() != live_exp_avg.data_ptr()

    # scalars/strings pass through untouched
    assert snapshot_to_cpu({"a": 1, "b": "x", "c": None}) == {"a": 1, "b": "x", "c": None}
