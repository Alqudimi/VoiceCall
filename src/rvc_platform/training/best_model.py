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
from dataclasses import dataclass
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
BEST_HISTORY_NAME = "best_history.json"
SPLIT_META_NAME = "validation_split.json"
TRAIN_LIST_NAME = "filelist_train.txt"
VAL_LIST_NAME = "filelist_val.txt"
BEST_EXPORT_KEEP = 5          # how many best-model exports to keep on disk (used for Top-K soup)
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
    reconstruction proxy; KL and adversarial diagnostics are collected for
    observability. On top of that (researched composite — see
    UPGRADE_REPORT_AR.md):
      * ``val_spk``      — ECAPA speaker-embedding cosine similarity between
                          reconstruction and ground truth (timbre preservation),
                          automatically skipped when ``speechbrain`` is
                          unavailable;
      * ``val_f0_rmse``  — RMVPE F0 RMSE diagnostic, skipped when the rmvpe
                          asset is unavailable;
      * ``composite``    — val_mel + spk_weight * (1 - val_spk): the
                          best-model selection metric (lower is better).
    Evaluation is forward-only and bounded by ``max_batches``, so it costs a
    small fraction of one training epoch. No optional metric can fail a run.
    """

    def __init__(self, hps, device: str, is_half: bool, val_filelist: Path,
                 max_batches: int | None = None,
                 speaker_embedder=None) -> None:
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
        batch_size = max(1, min(int(hps.train.get("batch_size", 1) or 1), 4))
        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate,
        )
        # ---- optional objective metrics (guarded: absence degrades to
        # val_mel-only selection exactly as before)
        from rvc_platform.training import eval_metrics

        self._metrics = eval_metrics
        self.spk_weight = float(hps.train.get("spk_weight", 1.0) or 0.0)
        self.speaker = speaker_embedder
        if self.speaker is None and self.spk_weight > 0.0:
            try:
                self.speaker = eval_metrics.try_build_speaker_embedder(device)
            except Exception:
                self.speaker = None
        self.f0_fn = None
        if float(hps.train.get("f0_eval", 1) or 0) > 0 and int(hps.if_f0) == 1:
            try:
                self.f0_fn = eval_metrics.try_build_f0_helper(device, is_half)
            except Exception:
                self.f0_fn = None

    @property
    def has_speaker(self) -> bool:
        return self.speaker is not None

    def evaluate(self, net_g: torch.nn.Module) -> dict:
        hps = self.hps
        was_training = net_g.training
        net_g.eval()
        sums = {
            "val_mel": 0.0, "val_infer_mel": 0.0,
            "val_kl": 0.0, "val_spk": 0.0, "val_f0_rmse": 0.0,
        }
        items = 0
        infer_items = 0
        spk_items = 0
        f0_items = 0
        batches = 0
        has_infer = hasattr(net_g, "infer") and callable(getattr(net_g, "infer"))
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
                        wave = wave.cuda(non_blocking=True)
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
                            n_batch = int(y_hat.shape[0])

                            # ---- TRUE inference-path evaluation (enc_p -> flow_reverse -> dec) ----
                            # Measures actual voice conversion fidelity, unlike posterior enc_q!
                            y_eval_pred = None
                            y_eval_true = None
                            if has_infer:
                                try:
                                    if int(hps.if_f0) == 1:
                                        y_inf = net_g.infer(
                                            phone, phone_lengths, pitch, pitchf, sid,
                                        )[0]
                                    else:
                                        y_inf = net_g.infer(
                                            phone, phone_lengths, sid,
                                        )[0]
                                    inf_len = min(y_inf.shape[-1], wave.shape[-1])
                                    if inf_len >= int(hps.data["filter_length"]):
                                        y_inf_clip = y_inf[..., :inf_len].float()
                                        w_clip = wave[..., :inf_len].float()
                                        inf_mel = mel_processing.mel_spectrogram_torch(
                                            y_inf_clip.squeeze(1), hps.data["filter_length"],
                                            hps.data["n_mel_channels"], hps.data["sampling_rate"],
                                            hps.data["hop_length"], hps.data["win_length"],
                                            hps.data["mel_fmin"], hps.data["mel_fmax"],
                                        )
                                        m_frames = min(inf_mel.shape[-1], mel.shape[-1])
                                        inf_mel_l1 = torch.nn.functional.l1_loss(
                                            mel[..., :m_frames].float(),
                                            inf_mel[..., :m_frames].float(),
                                        )
                                        sums["val_infer_mel"] += float(inf_mel_l1) * n_batch
                                        infer_items += n_batch
                                        y_eval_pred = y_inf_clip
                                        y_eval_true = w_clip
                                except Exception:
                                    logger.debug("inference-path val skipped", exc_info=True)

                            wave_sliced = commons.slice_segments(
                                wave, ids_slice * hps.data["hop_length"],
                                hps.train["segment_size"],
                            )
                            if y_eval_pred is None:
                                y_eval_pred = y_hat.detach().float()
                                y_eval_true = wave_sliced.detach().float()

                            if self.speaker is not None:
                                try:
                                    sim = self._metrics.speaker_similarity(
                                        self.speaker, y_eval_pred, y_eval_true,
                                        int(hps.data["sampling_rate"]),
                                    )
                                    sums["val_spk"] += sim * n_batch
                                    spk_items += n_batch
                                except Exception:
                                    logger.warning("speaker-similarity failed; "
                                                   "metric disabled for this run",
                                                   exc_info=True)
                                    self.speaker = None
                            if self.f0_fn is not None:
                                try:
                                    if has_infer and y_eval_pred.shape[-1] > y_hat.shape[-1]:
                                        rmse = self._full_f0_rmse(y_eval_pred, pitchf)
                                    else:
                                        rmse = self._batch_f0_rmse(y_hat, pitchf, ids_slice)
                                    if rmse is not None:
                                        sums["val_f0_rmse"] += rmse * n_batch
                                        f0_items += n_batch
                                except Exception:
                                    logger.warning("f0 RMSE failed; metric "
                                                   "disabled for this run",
                                                   exc_info=True)
                                    self.f0_fn = None
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
        out = {
            "val_mel": sums["val_mel"] / items,
            "val_kl": sums["val_kl"] / items,
            "val_batches": batches,
            "val_items": items,
        }
        if infer_items > 0:
            out["val_infer_mel"] = sums["val_infer_mel"] / infer_items
        mel_proxy = (
            0.4 * out["val_mel"] + 0.6 * out["val_infer_mel"]
            if "val_infer_mel" in out else out["val_mel"]
        )
        if spk_items > 0:
            out["val_spk"] = sums["val_spk"] / spk_items
            out["composite"] = self._metrics.composite_score(
                mel_proxy, out["val_spk"], self.spk_weight)
        if f0_items > 0:
            out["val_f0_rmse"] = sums["val_f0_rmse"] / f0_items
            if "composite" in out:
                # Slight F0 prosody penalty on top of timbre + mel composite
                out["composite"] += 0.15 * (out["val_f0_rmse"] / 100.0)
        if spk_items == 0 and f0_items > 0:
            f0_weight = float(self.hps.train.get("f0_weight", 0.5) or 0.5)
            out["composite_f0"] = self._metrics.composite_score_f0(
                mel_proxy, out["val_f0_rmse"], f0_weight)
        return out

    def _full_f0_rmse(self, y_pred, pitchf) -> float | None:
        """Mean voiced F0 RMSE (Hz) across full-utterance inference outputs."""
        sr = int(self.hps.data["sampling_rate"])
        gt = pitchf.detach().cpu().numpy()
        pred16 = self._metrics.resample_16k(
            y_pred.detach().float().squeeze(1), sr,
        ).cpu().numpy()
        rmses = []
        for b in range(gt.shape[0]):
            value = self._metrics.f0_rmse(self.f0_fn, pred16[b], gt[b])
            if value is not None:
                rmses.append(value)
        return float(sum(rmses) / len(rmses)) if rmses else None

    def _batch_f0_rmse(self, y_hat, pitchf, ids_slice) -> float | None:
        """Mean voiced F0 RMSE (Hz) across the batch's sliced segments.

        The ground-truth F0 for the sliced segment is taken from the loader's
        ``pitchf`` frames at the same 10 ms grid (frame indices ``ids_slice``),
        and the prediction is extracted (RMVPE) from the 16 kHz-resampled
        reconstruction — the standard voiced RPE alignment used in pitch
        literature (CREPE's evaluation protocol).
        """
        hps = self.hps
        seg_frames = max(1, hps.train["segment_size"] // hps.data["hop_length"])
        sr = int(hps.data["sampling_rate"])
        starts = ids_slice.detach().cpu().numpy().ravel().astype(int)
        gt = pitchf.detach().cpu().numpy()
        pred16 = self._metrics.resample_16k(
            y_hat.detach().float().squeeze(1), sr,
        ).cpu().numpy()
        rmses = []
        for b in range(gt.shape[0]):
            f0_true = gt[b, starts[b]:starts[b] + seg_frames]
            if f0_true.size == 0:
                continue
            value = self._metrics.f0_rmse(self.f0_fn, pred16[b], f0_true)
            if value is not None:
                rmses.append(value)
        if not rmses:
            return None
        return float(sum(rmses) / len(rmses))


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
    # Track top-K best models in best_history.json for metric-weighted soup
    hist_path = best_json_path.with_name(BEST_HISTORY_NAME)
    try:
        history = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.is_file() else []
        if not isinstance(history, list):
            history = []
    except (OSError, json.JSONDecodeError):
        history = []
    history.append(dict(metadata))
    history.sort(key=lambda item: float(item.get("score", float("inf"))))
    if keep > 0:
        history = history[:keep]
    htmp = hist_path.with_suffix(".json.tmp")
    htmp.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    htmp.replace(hist_path)
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
