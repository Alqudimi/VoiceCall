"""Publish trained RVC artifacts to a user-selected private Hugging Face repo."""
from __future__ import annotations

import os
from pathlib import Path


def publish_run(run_dir: Path, repo_id: str, token: str | None = None) -> str:
    """Create/ensure a private model repo and upload exports and retrieval indexes.

    Training inputs, preprocessed audio/features, and full optimizer checkpoints
    are intentionally excluded from uploads.
    """
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) on the training host")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is missing; install project dependencies") from exc

    run_dir = Path(run_dir)
    exports = run_dir / "exports"
    exported = sorted(exports.glob("*.pth")) if exports.is_dir() else []
    final_model = exports / f"{run_dir.name}.pth"
    model_file = final_model if final_model.is_file() else (
        max(exported, key=lambda path: path.stat().st_mtime) if exported else None
    )
    available_indexes = sorted(run_dir.glob("added_IVF*.index"))
    index_file = max(available_indexes, key=lambda path: path.stat().st_mtime) if available_indexes else None
    if model_file is None:
        raise RuntimeError(f"no exported model found in {exports}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if not getattr(info, "private", False):
        raise RuntimeError(
            f"Hugging Face repo {repo_id} is public; choose a private repo or make it private first"
        )
    uploaded = []
    for file_path in ([model_file, index_file] if index_file else [model_file]):
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=f"models/{file_path.name}" if file_path == model_file
            else f"indexes/{file_path.name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Publish {file_path.name} from {run_dir.name}",
        )
        uploaded.append(file_path.name)
    return f"https://huggingface.co/{repo_id}"
