"""Best-so-far model tracking with held-out validation.

Semantics mirror the industry-standard checkpoint callbacks (Keras
``ModelCheckpoint(save_best_only=True)``, Lightning ``ModelCheckpoint`` with
``monitor``/``mode``): a monitored metric with an explicit direction, and a
persisted best-so-far record so a *resumed* run can never regress the bar —
whenever any epoch reaches the best score seen so far, its inference export
is written and (via the worker layer) published to the ``best/`` area of the
Hugging Face repo, updating the previous best in place.

Metric choice:
* With a validation split (``val_split > 0``): held-out mel-reconstruction
  L1 loss (``val_mel``) — the same objective the generator trains on and the
  standard perceptual proxy used across HiFi-GAN/VITS-style vocoders, but
  computed on audio the model never trained on. GAN adversarial terms are
  unbounded game signals, not quality signals, so they are reported as
  diagnostics only.
* Without a split: the epoch-mean training mel loss (``train_mel``, the
  ``c_mel``-weighted term rescaled to raw L1) — a monotone-ish fallback that
  requires zero changes to the data pipeline.

Files (all inside the run directory):
    filelist.txt             original training manifest (prepared data)
    filelist_train.txt       training lines after the split (pinned on disk)
    filelist_val.txt         held-out lines (deterministic, seeded shuffle)
    validation_split.json    {filelist_sha256, val_split, counts} — the split
                             is re-derived (and the best bar reset) only when
                             the underlying filelist content changes
    checkpoints/best.json    best-so-far metadata
    exports/<exp>_best_e{E}_s{S}.pth   best-model inference exports (pruned)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rvc_platform.engine.rvc import checkpoint as engine_ckpt
from rvc_platform.engine.rvc.models import commons
from rvc_platform.training import losses as L
from rvc_platform.training import mel_processing
from rvc_platform.training.dataset import (
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)

logger = logging.getLogger(__name__)

BEST_STATE_NAME = "best.json"
SPLIT_META_NAME = "validation_split.json"
TRAIN_LIST_NAME = "filelist_train.txt"
VAL_LIST_NAME = "filelist_val.txt"
BEST_EXPORT_KEEP = 3          # how many best-model exports to keep on disk
DEFAULT_SPLIT_SEED = 1234


# --------------------------------------------------------------------- info
@dataclass
class BestModelInfo:
    metric: str
    score: float
    epoch: int
    global_step: int
    improved_at: float

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "score": round(float(self.score), 6),
            "epoch": int(self.epoch),
            "global_step": int(self.global_step),
            "improved_at": float(self.improved_at),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BestModelInfo":
        return cls(
            metric=str(data["metric"]),
            score=float(data["score"]),
            epoch=int(data["epoch"]),
            global_step=int(data["global_step"]),
            improved_at=float(data.get("improved_at", 0.0)),
        )


# ------------------------------------------------------------------ tracker
class BestModelTracker:
    """Monitor one metric, remember the best score, persist it to best.json."""

    def __init__(self, state_path: Path, metric: str = "val_mel",
                 mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.state_path = Path(state_path)
        self.metric = metric
        self.mode = mode
        self.best: BestModelInfo | None = None

    # -- persistence ------------------------------------------------------
    def load(self) -> bool:
        """Restore a previous best (returns True when one was restored)."""
        if not self.state_path.is_file():
            return False
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.best = BestModelInfo.from_dict(data)
            return True
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning("unreadable %s; starting the best bar fresh",
                           self.state_path, exc_info=True)
            self.best = None
            return False

    def reset(self) -> None:
        self.best = None

    def _better(self, candidate: float, current: float) -> bool:
        return candidate < current if self.mode == "min" else candidate > current

    # -- scoring ------------------------------------------------------------
    def update(self, scores: dict, epoch: int, global_step: int) -> BestModelInfo | None:
        """Record a new score. Returns BestModelInfo when it is a new best."""
        if self.metric not in scores:
            return None
        try:
            score = float(scores[self.metric])
        except (TypeError, ValueError):
            return None
        if math.isnan(score) or math.isinf(score):
            return None
        if self.best is not None and not self._better(score, self.best.score):
            return None
        info = BestModelInfo(
            metric=self.metric, score=score,
            epoch=int(epoch), global_step=int(global_step),
            improved_at=time.time(),
        )
        self.best = info
        return info


# ------------------------------------------------------------------- split
@dataclass
class SplitResult:
    train_path: Path
    val_path: Path
    n_train: int
    n_val: int
    rederived: bool


def derive_validation_split(
    training_files: Path,
    val_split: float,
    seed: int = DEFAULT_SPLIT_SEED,
) -> SplitResult | None:
    """Split the prepared filelist into deterministic train/val manifests.

    The split is *pinned on disk*: as long as ``filelist.txt`` content and
    ``val_split`` are unchanged the existing files are reused, so resumed
    runs keep training on exactly the same lines. When the filelist content
    changes (re-prepared dataset) the split is re-derived — callers should
    reset the best-model bar then, because scores across different data are
    not comparable. Returns None when no split is possible/needed.
    """
    training_files = Path(training_files)
    if val_split <= 0 or not training_files.is_file():
        return None
    lines = [ln for ln in training_files.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    n = len(lines)
    if n < 2:
        return None

    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    meta_path = training_files.parent / SPLIT_META_NAME
    train_path = training_files.parent / TRAIN_LIST_NAME
    val_path = training_files.parent / VAL_LIST_NAME

    if meta_path.is_file() and train_path.is_file() and val_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (meta.get("filelist_sha256") == digest
                    and float(meta.get("val_split", -1)) == float(val_split)):
                return SplitResult(
                    train_path=train_path, val_path=val_path,
                    n_train=int(meta.get("n_train", 0)),
                    n_val=int(meta.get("n_val", 0)),
                    rederived=False,
                )
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning("unreadable %s; re-deriving the split", meta_path)

    n_val = max(1, min(n - 1, int(round(n * val_split))))
    order = list(range(n))
    random.Random(seed).shuffle(order)          # stable, reproducible split
    val_idx = set(order[:n_val])
    val_lines = [ln for i, ln in enumerate(lines) if i in val_idx]
    train_lines = [ln for i, ln in enumerate(lines) if i not in val_idx]

    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(".txt.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    _atomic_write(train_path, "\n".join(train_lines) + "\n")
    _atomic_write(val_path, "\n".join(val_lines) + "\n")
    meta_path.write_text(
        json.dumps({
            "filelist_sha256": digest,
            "val_split": val_split,
            "n_train": len(train_lines),
            "n_val": len(val_lines),
            "seed": seed,
            "derived_at": time.time(),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("validation split: %d train / %d val lines (pinned to disk)",
                len(train_lines), len(val_lines))
    return SplitResult(
        train_path=train_path, val_path=val_path,
        n_train=len(train_lines), n_val=len(val_lines),
        rederived=True,
    )


# --------------------------------------------------------------- evaluator
class ValidationEvaluator:
    """Held-out generator losses under ``no_grad`` + eval mode.

    The mel-reconstruction L1 (raw, without the ``c_mel`` weight) is the
    monitored quality proxy; KL and adversarial diagnostics are collected for
    observability. Evaluation is forward-only and bounded by ``max_batches``,
    so it costs a small fraction of one training epoch.
    """

    def __init__(self, hps, device: str, is_half: bool, val_filelist: Path,
                 max_batches: int | None = None) -> None:
        import os

        if max_batches is None:
            max_batches = int(os.environ.get("RVC_MAX_VAL_BATCHES", "64"))
        self.hps = hps
        self.device = device
        self.is_half = is_half
        self.max_batches = max(1, max_batches)
        if int(hps.if_f0) == 1:
            dataset = TextAudioLoaderMultiNSFsid(str(val_filelist), hps.data)
            collate = TextAudioCollateMultiNSFsid()
        else:
            dataset = TextAudioLoader(str(val_filelist), hps.data)
            collate = TextAudioCollate()
        batch_size = max(1, min(int(hps.train.get("batch_size", 1) or 1), 8))
        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate,
        )

    def evaluate(self, net_g: torch.nn.Module) -> dict:
        hps = self.hps
        was_training = net_g.training
        net_g.eval()
        sums = {"val_mel": 0.0, "val_kl": 0.0}
        items = 0
        batches = 0
        try:
            with torch.no_grad():
                for batch_idx, info in enumerate(self.loader):
                    if batch_idx >= self.max_batches:
                        break
                    if int(hps.if_f0) == 1:
                        (phone, phone_lengths, pitch, pitchf, spec, spec_lengths,
                         wave, wave_lengths, sid) = info
                    else:
                        phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
                    if self.device.startswith("cuda"):
                        phone = phone.cuda(non_blocking=True)
                        phone_lengths = phone_lengths.cuda(non_blocking=True)
                        if int(hps.if_f0) == 1:
                            pitch = pitch.cuda(non_blocking=True)
                            pitchf = pitchf.cuda(non_blocking=True)
                        sid = sid.cuda(non_blocking=True)
                        spec = spec.cuda(non_blocking=True)
                        spec_lengths = spec_lengths.cuda(non_blocking=True)
                    with torch.autocast("cuda" if self.is_half else "cpu",
                                        enabled=self.is_half):
                        if int(hps.if_f0) == 1:
                            (y_hat, ids_slice, x_mask, z_mask,
                             (z, z_p, m_p, logs_p, m_q, logs_q)) = net_g(
                                phone, phone_lengths, pitch, pitchf,
                                spec, spec_lengths, sid,
                            )
                        else:
                            (y_hat, ids_slice, x_mask, z_mask,
                             (z, z_p, m_p, logs_p, m_q, logs_q)) = net_g(
                                phone, phone_lengths, spec, spec_lengths, sid,
                            )
                        mel = mel_processing.spec_to_mel_torch(
                            spec, hps.data["filter_length"],
                            hps.data["n_mel_channels"], hps.data["sampling_rate"],
                            hps.data["mel_fmin"], hps.data["mel_fmax"],
                        )
                        y_mel = commons.slice_segments(
                            mel, ids_slice,
                            hps.train["segment_size"] // hps.data["hop_length"],
                        )
                        with torch.autocast("cuda" if self.is_half else "cpu",
                                            enabled=False):
                            y_hat_mel = mel_processing.mel_spectrogram_torch(
                                y_hat.float().squeeze(1), hps.data["filter_length"],
                                hps.data["n_mel_channels"], hps.data["sampling_rate"],
                                hps.data["hop_length"], hps.data["win_length"],
                                hps.data["mel_fmin"], hps.data["mel_fmax"],
                            )
                            if self.is_half:
                                y_hat_mel = y_hat_mel.half()
                            mel_l1 = torch.nn.functional.l1_loss(y_mel, y_hat_mel)
                            kl = L.kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                    n = int(y_hat.shape[0])
                    sums["val_mel"] += float(mel_l1) * n
                    sums["val_kl"] += float(kl) * n
                    items += n
                    batches += 1
        finally:
            if was_training:
                net_g.train()
        if items == 0:
            return {}
        return {
            "val_mel": sums["val_mel"] / items,
            "val_kl": sums["val_kl"] / items,
            "val_batches": batches,
            "val_items": items,
        }


# ------------------------------------------------------------- artifacts
def write_best_artifacts(
    snapshot: dict,
    hps,
    metadata: dict,
    out_path: Path,
    best_json_path: Path,
    keep: int = BEST_EXPORT_KEEP,
) -> Path:
    """Write the best-model inference export + best.json, prune old exports.

    Runs on the background I/O writer from a CPU snapshot of the generator,
    so producing the export never stalls the training loop.
    """
    out_path = Path(out_path)
    export = engine_ckpt.export_inference_checkpoint(
        snapshot, hps, int(metadata["epoch"]), hps.version,
        out_path, hps.sample_rate, hps.if_f0,
    )
    tmp = best_json_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(best_json_path)
    experiment = metadata.get("experiment")
    if experiment:
        olds = sorted(
            out_path.parent.glob(f"{experiment}_best_e*_s*.pth"),
            key=lambda p: p.stat().st_mtime,
        )
        for stale in olds[:-keep] if keep > 0 else olds:
            try:
                stale.unlink()
            except OSError:
                logger.warning("could not prune old best export %s", stale)
    return export


def read_best_state(run_dir: Path) -> dict | None:
    """Read checkpoints/best.json for API/UI surfaces (never raises)."""
    path = Path(run_dir) / "checkpoints" / BEST_STATE_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
