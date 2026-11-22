#!/usr/bin/env python3
"""Background daemon for asynchronous Hugging Face uploads without slowing down RVC training."""
import os
import sys
import time
import json
import logging
from pathlib import Path

# Set low scheduling priority to never steal CPU from training data loaders
try:
    os.nice(15)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [HF-Sync] %(message)s",
    handlers=[
        logging.FileHandler("/content/iraqi-voice-platform/runs/hf_sync.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("hf_sync")

EXPERIMENT = os.environ.get("EXPERIMENT", "iraqi_voice")
REPO_ID = os.environ.get("HF_REPO", "eng7mi/Iraqi-v2")
RUN_DIR = Path(f"/content/iraqi-voice-platform/runs/{EXPERIMENT}")
EXPORTS_DIR = RUN_DIR / "exports"
SYNC_STATE_FILE = RUN_DIR / "hf_synced_state.json"

def get_synced_state():
    if SYNC_STATE_FILE.exists():
        try:
            return json.loads(SYNC_STATE_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_synced_state(state):
    try:
        SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        logger.warning(f"Failed to save sync state: {e}")

def run_sync_once():
    if not EXPORTS_DIR.exists():
        return False

    synced_state = get_synced_state()
    last_synced_mtime = synced_state.get("last_mtime", 0)

    # Find candidate model files
    exports = sorted(EXPORTS_DIR.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not exports:
        return False

    newest_export = exports[-1]
    newest_mtime = newest_export.stat().st_mtime

    # Find best model
    best_exports = sorted(EXPORTS_DIR.glob("*best*.pth"), key=lambda p: p.stat().st_mtime)
    best_export = best_exports[-1] if best_exports else newest_export

    # Only sync if newer files exist
    if newest_mtime <= last_synced_mtime:
        return False

    # Find index
    v2_indices = sorted(RUN_DIR.glob("*v2.index"), key=lambda p: p.stat().st_mtime)
    v2_index = v2_indices[-1] if v2_indices else None

    config_file = RUN_DIR / "config.json"
    preview_wav = Path("/content/test_preview.wav")

    logger.info(f"New export detected: {newest_export.name} (mtime {newest_mtime})")
    
    try:
        from huggingface_hub import HfApi, CommitOperationAdd
        api = HfApi()
        
        # Verify repo
        api.create_repo(repo_id=REPO_ID, repo_type="model", private=True, exist_ok=True)
        
        operations = [
            CommitOperationAdd(path_in_repo=f"models/{newest_export.name}", path_or_fileobj=str(newest_export)),
            CommitOperationAdd(path_in_repo="best/model.pth", path_or_fileobj=str(best_export)),
        ]
        if v2_index and v2_index.exists():
            operations.append(CommitOperationAdd(path_in_repo="best/index.index", path_or_fileobj=str(v2_index)))
            operations.append(CommitOperationAdd(path_in_repo=f"indexes/{v2_index.name}", path_or_fileobj=str(v2_index)))
        if config_file.exists():
            operations.append(CommitOperationAdd(path_in_repo="best/config.json", path_or_fileobj=str(config_file)))
        if preview_wav.exists():
            operations.append(CommitOperationAdd(path_in_repo="best/sample_preview.wav", path_or_fileobj=str(preview_wav)))

        logger.info(f"Uploading {len(operations)} files to Hugging Face ({REPO_ID})...")
        api.create_commit(
            repo_id=REPO_ID,
            repo_type="model",
            operations=operations,
            commit_message=f"Auto-sync: {newest_export.name} + v2 index + best model",
        )
        logger.info("HF Auto-Sync successfully completed!")
        save_synced_state({
            "last_mtime": newest_mtime,
            "last_synced_file": newest_export.name,
            "best_file": best_export.name,
            "synced_at": time.time(),
        })
        return True
    except Exception as e:
        logger.error(f"Sync error: {e}")
        return False

def daemon_loop():
    logger.info(f"Starting HF background sync daemon for {REPO_ID} (interval: 60s)...")
    while True:
        try:
            run_sync_once()
        except Exception as e:
            logger.error(f"Loop error: {e}")
        time.sleep(60)

if __name__ == "__main__":
    if "--once" in sys.argv:
        run_sync_once()
    else:
        daemon_loop()
