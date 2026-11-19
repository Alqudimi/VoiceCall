"""Publish trained RVC artifacts to a user-selected private Hugging Face repo.

Upload strategy (following the official huggingface_hub guidance):
* one *atomic* commit per upload via ``create_commit`` with
  ``CommitOperationAdd`` — the model, its retrieval index and the metadata
  land together or not at all, and large files go to LFS automatically;
* a *background publisher thread* (``HfBackgroundPublisher``) so uploads,
  retries and backoff never block the training loop — the same
  single-ordered-queue idea as the Hub's own non-blocking ``run_as_future``
  helpers, plus retry, coalescing of superseded best-model jobs and status
  reporting that this project needs on top.

Repository layout (all under the private repo):

    models/<export>.pth        periodic/final inference exports
    indexes/<index>.index      FAISS retrieval indexes
    best/model.pth             best-so-far model — updated in place
    best/index.index           retrieval index for the best model
    best/metadata.json         {metric, score, epoch, global_step, ...}

Training inputs, preprocessed audio/features, and full optimizer checkpoints
are intentionally excluded from uploads.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

REPO_URL_TEMPLATE = "https://huggingface.co/{repo_id}"
BEST_DIR = "best"
UPLOAD_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5.0          # 5s, 10s, 20s ...


# ----------------------------------------------------------------- helpers
def _resolve_token(token: str | None = None) -> str:
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) on the training host")
    return token


def _import_hub():
    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:  # pragma: no cover — dependency is pinned
        raise RuntimeError("huggingface_hub is missing; install project dependencies") from exc
    return HfApi, CommitOperationAdd


def newest_index_file(run_dir: Path) -> Path | None:
    """Newest FAISS retrieval index at the run-dir root, if any."""
    run_dir = Path(run_dir)
    indexes = sorted(run_dir.glob("added_IVF*.index"))
    return max(indexes, key=lambda p: p.stat().st_mtime) if indexes else None


def pick_export_file(run_dir: Path) -> Path | None:
    """The export to publish: the final model if present, else the newest."""
    run_dir = Path(run_dir)
    exports = run_dir / "exports"
    if not exports.is_dir():
        return None
    final = exports / f"{run_dir.name}.pth"
    if final.is_file():
        return final
    exported = sorted(exports.glob("*.pth"))
    return max(exported, key=lambda p: p.stat().st_mtime) if exported else None


def _ensure_private_repo(api, repo_id: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if not getattr(info, "private", False):
        raise RuntimeError(
            f"Hugging Face repo {repo_id} is public; choose a private repo or make it private first"
        )


def _commit(api, CommitOperationAdd, repo_id: str, operations: list,
            message: str) -> None:
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=message,
    )


# ------------------------------------------------------------ sync publish
def publish_run(run_dir: Path, repo_id: str, token: str | None = None) -> str:
    """Synchronously publish the newest export + index (one atomic commit).

    Kept for CLI/tooling use; the training worker uses the background
    publisher below so this path never touches the training loop.
    """
    run_dir = Path(run_dir)
    HfApi, CommitOperationAdd = _import_hub()
    model_file = pick_export_file(run_dir)
    if model_file is None:
        raise RuntimeError(f"no exported model found in {run_dir / 'exports'}")
    index_file = newest_index_file(run_dir)

    api = HfApi(token=_resolve_token(token))
    _ensure_private_repo(api, repo_id)
    operations = [
        CommitOperationAdd(
            path_in_repo=f"models/{model_file.name}",
            path_or_fileobj=str(model_file),
        )
    ]
    if index_file is not None:
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"indexes/{index_file.name}",
                path_or_fileobj=str(index_file),
            )
        )
    _commit(api, CommitOperationAdd, repo_id, operations,
            f"Publish {model_file.name} from {run_dir.name}")
    return REPO_URL_TEMPLATE.format(repo_id=repo_id)


# --------------------------------------------------------- background pub
class HfBackgroundPublisher:
    """Serialized, retrying, non-blocking uploader for one private repo.

    One daemon thread consumes a FIFO job queue: repo verification happens
    once, each job becomes a single atomic ``create_commit``. Failures retry
    with exponential backoff and are reported through ``on_event`` — they
    never propagate into the training process.

    Best-model jobs coalesce: when a *better* best model is submitted while
    an older best job is still queued (not started), the older one is dropped
    as superseded — the repo's ``best/`` area ends at the newest best either
    way, and we avoid uploading files nobody will ever download.
    """

    def __init__(self, repo_id: str, token: str | None = None,
                 on_event: Callable[[dict], None] | None = None) -> None:
        self.repo_id = repo_id
        self._token = token
        self._on_event = on_event
        self._jobs: deque = deque()
        self._lock = threading.Lock()
        self._has_work = threading.Condition(self._lock)
        self._closed = False
        self._thread: threading.Thread | None = None
        self._api = None
        self._CommitOperationAdd = None
        self._repo_verified = False
        self._uploaded = 0
        self._failed = 0
        self._last_error: str | None = None
        self._last_best_key: str | None = None

    # ---------------------------------------------------------------- api
    @property
    def last_best_key(self) -> str | None:
        with self._lock:
            return self._last_best_key

    def status(self) -> dict:
        with self._lock:
            return {
                "repo_id": self.repo_id,
                "pending": len(self._jobs),
                "uploaded": self._uploaded,
                "failed": self._failed,
                "last_error": self._last_error,
                "last_best_key": self._last_best_key,
            }

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._closed:
                raise RuntimeError("publisher already closed")
            self._thread = threading.Thread(
                target=self._run, name=f"hf-publisher:{self.repo_id}", daemon=True,
            )
            self._thread.start()

    def submit_export(self, run_dir: Path, model_path: Path) -> None:
        """Queue an upload of one exported inference checkpoint (+ index)."""
        self._enqueue({
            "kind": "export",
            "run_dir": Path(run_dir),
            "model_path": Path(model_path),
        })

    def submit_best(self, model_path: Path | str, index_path: Path | str | None,
                    metadata: dict) -> None:
        """Queue an upload updating the repo's best/ area in place."""
        key = f"{metadata.get('epoch')}:{metadata.get('global_step')}"
        self._enqueue({
            "kind": "best",
            "key": key,
            "model_path": Path(model_path),
            "index_path": Path(index_path) if index_path else None,
            "metadata": dict(metadata),
        })

    def close(self, timeout: float = 300.0) -> dict:
        """Stop accepting jobs, drain the queue, join the thread."""
        with self._lock:
            if self._closed:
                drained = not (self._thread and self._thread.is_alive())
            else:
                self._closed = True
                drained = False
            thread = self._thread
        if thread is not None:
            with self._has_work:
                self._has_work.notify_all()
            thread.join(timeout)
            drained = not thread.is_alive()
            if not drained:
                logger.warning(
                    "HF publisher did not drain within %.0fs (%s)",
                    timeout, self.status(),
                )
        status = self.status()
        if status["pending"]:
            logger.warning("HF publisher closed with %s jobs dropped",
                           status["pending"])
        return status

    # ------------------------------------------------------------- queueing
    def _enqueue(self, job: dict) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("publisher closed; cannot enqueue")
            if job["kind"] == "best":
                # drop queued (not yet started) best jobs superseded by this one
                before = len(self._jobs)
                self._jobs = deque(
                    j for j in self._jobs
                    if not (j.get("kind") == "best" and j.get("key") != job["key"])
                )
                if len(self._jobs) != before:
                    logger.info("superseded %d queued best-model upload(s)",
                                before - len(self._jobs))
            self._jobs.append(job)
            started = self._thread is not None and self._thread.is_alive()
        if not started:
            self.start()

    # ---------------------------------------------------------------- loop
    def _run(self) -> None:
        while True:
            with self._has_work:
                while not self._jobs and not self._closed:
                    self._has_work.wait()
                if not self._jobs and self._closed:
                    return
                job = self._jobs.popleft()
            self._execute(job)

    # ------------------------------------------------------------- executor
    def _execute(self, job: dict) -> None:
        attempts = int(os.environ.get("RVC_HF_UPLOAD_ATTEMPTS", UPLOAD_ATTEMPTS))
        backoff = float(os.environ.get("RVC_HF_RETRY_BACKOFF", RETRY_BACKOFF_SECONDS))
        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                self._execute_once(job)
                return
            except BaseException as exc:  # noqa: BLE001 — never kill the thread
                last_exc = exc
                if attempt < attempts:
                    delay = backoff * (2 ** (attempt - 1))
                    logger.warning(
                        "HF upload attempt %d/%d for %s failed (%s); retrying in %.0fs",
                        attempt, attempts, job["kind"], exc, delay,
                    )
                    time.sleep(delay)
        with self._lock:
            self._failed += 1
            self._last_error = f"{job['kind']}: {last_exc}"
        logger.error("HF upload for %s failed permanently: %s",
                     job["kind"], last_exc)
        self._emit({
            "topic": "training.hf_upload_failed",
            "kind": job["kind"],
            "repo_id": self.repo_id,
            "error": str(last_exc),
        })

    def _execute_once(self, job: dict) -> None:
        if not self._ensure_api():
            raise RuntimeError("Hugging Face API unavailable (missing token or library)")
        url = REPO_URL_TEMPLATE.format(repo_id=self.repo_id)
        if job["kind"] == "best":
            metadata = dict(job["metadata"])
            operations = [
                self._CommitOperationAdd(
                    path_in_repo=f"{BEST_DIR}/model.pth",
                    path_or_fileobj=str(job["model_path"]),
                )
            ]
            if job["index_path"] is not None and Path(job["index_path"]).is_file():
                operations.append(
                    self._CommitOperationAdd(
                        path_in_repo=f"{BEST_DIR}/index.index",
                        path_or_fileobj=str(job["index_path"]),
                    )
                )
            operations.append(
                self._CommitOperationAdd(
                    path_in_repo=f"{BEST_DIR}/metadata.json",
                    path_or_fileobj=json.dumps(
                        metadata, ensure_ascii=False, indent=2
                    ).encode("utf-8"),
                )
            )
            info = metadata.get("metric", "metric")
            score = metadata.get("score", 0.0)
            epoch = metadata.get("epoch", "?")
            self._commit(operations, f"Best model: epoch {epoch} "
                                     f"({info}={score:.6f})")
            with self._lock:
                self._uploaded += 1
                self._last_best_key = job["key"]
            self._emit({
                "topic": "training.hf_best_uploaded",
                "kind": "best",
                "repo_id": self.repo_id,
                "url": url,
                "best": metadata,
            })
            return

        # kind == "export"
        model_path = Path(job["model_path"])
        if not model_path.is_file():
            raise RuntimeError(f"export file disappeared before upload: {model_path}")
        operations = [
            self._CommitOperationAdd(
                path_in_repo=f"models/{model_path.name}",
                path_or_fileobj=str(model_path),
            )
        ]
        index_file = newest_index_file(job["run_dir"])
        if index_file is not None:
            operations.append(
                self._CommitOperationAdd(
                    path_in_repo=f"indexes/{index_file.name}",
                    path_or_fileobj=str(index_file),
                )
            )
        self._commit(operations, f"Publish {model_path.name} from "
                                 f"{Path(job['run_dir']).name}")
        with self._lock:
            self._uploaded += 1
        self._emit({
            "topic": "training.hf_uploaded",
            "kind": "export",
            "repo_id": self.repo_id,
            "url": url,
            "file": model_path.name,
        })

    # ---------------------------------------------------------------- util
    def _ensure_api(self) -> bool:
        if self._api is not None and self._repo_verified:
            return True
        try:
            HfApi, CommitOperationAdd = _import_hub()
            api = HfApi(token=_resolve_token(self._token))
            _ensure_private_repo(api, self.repo_id)
        except Exception as exc:  # noqa: BLE001 — reported via job failure
            logger.warning("HF API initialization failed: %s", exc)
            return False
        self._api, self._CommitOperationAdd = api, CommitOperationAdd
        self._repo_verified = True
        return True

    def _commit(self, operations: list, message: str) -> None:
        _commit(self._api, self._CommitOperationAdd, self.repo_id,
                operations, message)

    def _emit(self, event: dict) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except BaseException:  # noqa: BLE001 — observers must not break uploads
            logger.exception("HF publisher on_event callback failed")
