"""Background I/O: keep heavy disk/network work off the training thread.

Design — snapshot-then-queue with one FIFO writer (the same core idea as
PyTorch's official async-checkpointing recipe for ``torch.distributed.
checkpoint``: the training thread only pays the cost of *copying* live tensor
state to CPU, while serialization (``torch.save``), file copies, mirror
refreshes and JSON writes run on a single background writer thread):

* ``snapshot_to_cpu`` produces private CPU copies of model/optimizer state
  that are safe to serialize on another thread while training keeps stepping
  (the optimizer mutates the live GPU tensors in place — without the copy the
  background save would race the training loop).
* ``BackgroundIOWorker`` executes jobs strictly in submission order. A single
  FIFO writer preserves the ordering invariants the resume logic depends on:
  ``state.json`` is only written after its ``G_/D_`` checkpoints, and the
  always-latest mirrors (``G_2333333``) always match the newest numbered
  checkpoint queued before them. PyTorch's async-checkpoint docs recommend
  "one asynchronous request at a time" — a single writer gives exactly that.
* ``wait_idle`` / ``close`` provide the crash-safe flush points used at the
  end of a run (and on the failure path) so no artifact is lost when the
  process exits.

The tiny journal writes in the worker's ``emit`` (events.jsonl append +
progress.json dump, a few KB) intentionally stay synchronous: they must keep
strict ordering with training events and cost microseconds, so moving them to
a thread would add re-reading complexity for zero measurable gain.
"""
from __future__ import annotations

import copy
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


def snapshot_to_cpu(obj: Any) -> Any:
    """Recursively detach + copy tensors to CPU.

    CUDA tensors are copied host-side (``.cpu()`` always copies across
    devices); CPU tensors are ``.clone()``-ed because the optimizer mutates
    them in place. Everything else is deep-copied only when it is not an
    immutable builtin (scheduler/scaler state dicts are plain data).
    """
    if isinstance(obj, torch.Tensor):
        t = obj.detach()
        return t.cpu() if t.device.type != "cpu" else t.clone()
    if isinstance(obj, dict):
        return {key: snapshot_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [snapshot_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(snapshot_to_cpu(value) for value in obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return copy.deepcopy(obj)


@dataclass
class _Job:
    name: str
    fn: Callable[..., Any]
    args: tuple
    on_done: Callable[[Any], None] | None = None
    on_error: Callable[[BaseException], None] | None = None
    enqueued_at: float = field(default_factory=time.time)


_STOP = object()


class BackgroundIOWorker:
    """One daemon thread running I/O jobs in strict FIFO order.

    ``submit`` never blocks on I/O (only on the internal lock, held for
    microseconds). ``on_done``/``on_error`` callbacks run on the writer
    thread — keep them cheap; they are used to chain follow-up work such as
    "the export file now exists, queue its Hugging Face upload".
    """

    def __init__(self, name: str = "rvc-io") -> None:
        self._queue: queue.Queue = queue.Queue()
        self._cond = threading.Condition()
        self._pending = 0
        self._last_error: str | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    # ------------------------------------------------------------------ api
    @property
    def pending(self) -> int:
        with self._cond:
            return self._pending

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        if not self._thread.is_alive() and not self._closed:
            self._thread.start()

    def submit(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        """Enqueue ``fn(*args)`` for background execution (FIFO)."""
        self.start()
        with self._cond:
            if self._closed:
                raise RuntimeError("background I/O worker is closed")
            self._pending += 1
        self._queue.put(_Job(name, fn, args, on_done, on_error))

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until every submitted job finished. False on timeout."""
        with self._cond:
            return self._cond.wait_for(lambda: self._pending == 0, timeout)

    def close(self, timeout: float = 60.0) -> bool:
        """Stop accepting jobs, drain what is queued, join the thread.

        Returns True when the writer exited cleanly with an empty queue.
        """
        with self._cond:
            if self._closed:
                return not self._thread.is_alive() and self._pending == 0
            self._closed = True
        self.start()
        self._queue.put(_STOP)
        self._thread.join(timeout)
        drained = not self._thread.is_alive() and self.pending == 0
        if not drained:
            logger.warning(
                "background I/O worker did not drain within %.0fs (pending=%s)",
                timeout, self.pending,
            )
        return drained

    # ------------------------------------------------------------------ loop
    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is _STOP:
                self._queue.task_done()
                break
            assert isinstance(job, _Job)
            started = time.time()
            try:
                result = job.fn(*job.args)
            except BaseException as exc:  # noqa: BLE001 — jobs must never kill the writer
                self._last_error = f"{job.name}: {exc}"
                logger.exception("background job '%s' failed", job.name)
                if job.on_error is not None:
                    try:
                        job.on_error(exc)
                    except BaseException:  # noqa: BLE001
                        logger.exception("on_error callback for '%s' failed", job.name)
            else:
                logger.debug("background job '%s' done in %.2fs",
                            job.name, time.time() - started)
                if job.on_done is not None:
                    try:
                        job.on_done(result)
                    except BaseException:  # noqa: BLE001
                        logger.exception("on_done callback for '%s' failed", job.name)
            finally:
                with self._cond:
                    self._pending -= 1
                    self._cond.notify_all()
                self._queue.task_done()
