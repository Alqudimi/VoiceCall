"""GAN trainer for RVC v2 with complete resume + observable progress.

Single-process implementation (single GPU or CPU) — the deliberate scope
decision for Colab (1x T4/L4/A100) and desktop setups; see
docs/ARCHITECTURE.md "Why single-process training".

Loop = upstream ``train/train.py`` numerics (identical losses, AMP policy,
ExponentialLR decay, AdamW betas) with an engineered shell around it:
  * full-state checkpointing (model/optim/scheduler/scaler/epoch/step/rng);
  * dataset-signature guard before resume;
  * periodic inference-checkpoint export;
  * cooperative stop (finishes current step, saves, exits 0);
  * progress events (epoch/step/losses/lr/eta) via callback.
"""
from __future__ import annotations

import logging
import threading
import time
import warnings
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from rvc_platform.core.entities import JobState
from rvc_platform.core.errors import CheckpointError, TrainingError
from rvc_platform.engine.rvc import checkpoint as engine_ckpt
from rvc_platform.engine.rvc.models import commons
from rvc_platform.training import checkpointing
from rvc_platform.training import losses as L
from rvc_platform.training import mel_processing
from rvc_platform.training.dataset import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)
from rvc_platform.training.run_config import TrainingRunConfig
from rvc_platform.training.utils import HParams

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated.*")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.*")


class Trainer:
    def __init__(
        self,
        config: TrainingRunConfig,
        hps: HParams,
        device: str,
        training_dtype: torch.dtype,
        event_cb: Callable[[dict], None] | None = None,
    ) -> None:
        self.config = config
        self.hps = hps
        self.device = device
        self.dtype = training_dtype
        self.is_half = training_dtype == torch.float16
        self._event = event_cb or (lambda payload: None)
        self._stop = threading.Event()
        self.state: JobState = JobState.IDLE
        self.global_step = 0
        self.epoch = 0
        self.steps_per_epoch = 0
        self.last_losses: dict[str, float] = {}
        self.checkpoint_dir = Path(hps.model_dir) / "checkpoints"
        self.dataset_signature: dict | None = None
        self.exports_dir = Path(hps.model_dir) / "exports"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ events
    def emit(self, topic: str, **payload) -> None:
        self._event({"topic": topic, **payload})

    def progress_payload(self) -> dict:
        return {
            "state": self.state.value,
            "epoch": self.epoch,
            "total_epochs": self.hps.total_epoch,
            "global_step": self.global_step,
            "steps_per_epoch": self.steps_per_epoch,
            "latest_losses": self.last_losses,
        }

    # ------------------------------------------------------------------ build
    def _build(self):
        from rvc_platform.engine.rvc.models import models as rvc_models

        hps = self.hps
        if hps.if_f0 == 1:
            dataset = TextAudioLoaderMultiNSFsid(hps.data["training_files"], hps.data)
            collate = TextAudioCollateMultiNSFsid()
            net_g = rvc_models.SynthesizerTrnMs768NSFsid(
                hps.data["filter_length"] // 2 + 1,
                hps.train["segment_size"] // hps.data["hop_length"],
                **hps.model, is_half=self.is_half, sr=hps.sample_rate,
            ) if hps.version == "v2" else rvc_models.SynthesizerTrnMs256NSFsid(
                hps.data["filter_length"] // 2 + 1,
                hps.train["segment_size"] // hps.data["hop_length"],
                **hps.model, is_half=self.is_half, sr=hps.sample_rate,
            )
            disc_cls = (rvc_models.MultiPeriodDiscriminatorV2 if hps.version == "v2"
                        else rvc_models.MultiPeriodDiscriminator)
        else:
            dataset = TextAudioLoader(hps.data["training_files"], hps.data)
            collate = TextAudioCollate()
            net_g = (rvc_models.SynthesizerTrnMs768NSFsid_nono if hps.version == "v2"
                     else rvc_models.SynthesizerTrnMs256NSFsid_nono)(
                hps.data["filter_length"] // 2 + 1,
                hps.train["segment_size"] // hps.data["hop_length"],
                **hps.model, is_half=self.is_half,
            )
            disc_cls = (rvc_models.MultiPeriodDiscriminatorV2 if hps.version == "v2"
                        else rvc_models.MultiPeriodDiscriminator)
        sampler = DistributedBucketSampler(
            dataset, hps.train["batch_size"],
            [100, 200, 300, 400, 500, 600, 700, 800, 900],
            num_replicas=1, rank=0, shuffle=True,
        )
        loader = DataLoader(
            dataset,
            # forked workers multiply the parent's RSS (torch allocator pages)
            # and can OOM small RAM/Colab boxes; single-process on CPU, 2 on CUDA.
            num_workers=2 if self.device.startswith("cuda") else 0,
            shuffle=False,
            pin_memory=self.device.startswith("cuda"),
            collate_fn=collate,
            batch_sampler=sampler,
            persistent_workers=False,
        )
        net_g = net_g.to(self.device)
        net_d = disc_cls(hps.model["use_spectral_norm"]).to(self.device)
        optim_g = torch.optim.AdamW(net_g.parameters(), hps.train["learning_rate"],
                                    betas=hps.train["betas"], eps=hps.train["eps"])
        optim_d = torch.optim.AdamW(net_d.parameters(), hps.train["learning_rate"],
                                     betas=hps.train["betas"], eps=hps.train["eps"])
        return dataset, loader, net_g, net_d, optim_g, optim_d

    # ------------------------------------------------------------------ resume
    def _resolve_explicit(self) -> Path | None:
        """Resolve config.resume_from (absolute path, run-relative name, or
        'latest') into a concrete G_*.pth inside this run's checkpoints."""
        raw = self.config.resume_from
        if raw is None:
            return None
        candidate = Path(raw)
        if candidate.is_file():
            return candidate
        in_run = self.checkpoint_dir / candidate.name
        if in_run.is_file():
            return in_run
        if str(raw).strip().lower() in ("latest", "final"):
            latest = checkpointing.latest_step_checkpoint(self.checkpoint_dir)
            if latest is not None:
                return latest
        raise CheckpointError(
            f"resume_from checkpoint not found: {raw} "
            f"(looked at {candidate} and {in_run})"
        )

    def _load_pretrained(self, net_g, net_d, optim_d) -> None:
        hps = self.hps
        if hps.pretrainG and Path(hps.pretrainG).is_file():
            saved = torch.load(hps.pretrainG, map_location="cpu")["model"]
            current = net_g.state_dict()
            emb = "emb_g.weight"
            if (emb in saved and emb in current
                    and saved[emb].shape != current[emb].shape
                    and saved[emb].dim() == current[emb].dim()
                    and saved[emb].shape[1:] == current[emb].shape[1:]):
                expanded = current[emb].clone()
                rows = min(saved[emb].shape[0], current[emb].shape[0])
                expanded[:rows] = saved[emb][:rows]
                saved[emb] = expanded
            net_g.load_state_dict(saved, strict=False)
            logger.info("loaded pretrained G: %s", hps.pretrainG)
        if hps.pretrainD and Path(hps.pretrainD).is_file():
            net_d.load_state_dict(
                torch.load(hps.pretrainD, map_location="cpu")["model"]
            )
            logger.info("loaded pretrained D: %s", hps.pretrainD)

    def _restore(self, net_g, net_d, optim_g, optim_d, sched_g, sched_d, scaler,
                 dataset_signature: dict | None) -> bool:
        """Explicit > full resume > legacy resume > pretrained start.
        Returns resumed?"""
        from rvc_platform.training.checkpointing import read_state

        if self.config.clean_start:
            self.config.resume = False
            self.config.resume_from = None
            logger.info("clean start requested; ignoring all existing checkpoints")
            return False

        state = read_state(self.checkpoint_dir / "state.json")

        explicit = self._resolve_explicit()
        if explicit is not None:
            # policy 1: an explicit checkpoint wins over auto-discovery
            meta = checkpointing.load_full(explicit, net_g, optim_g, sched_g, scaler)
            d_path = (self.checkpoint_dir / explicit.name.replace("G_", "D_")
                      if explicit.name.startswith("G_") else None)
            if d_path is not None and d_path.is_file():
                checkpointing.load_full(d_path, net_d, optim_d, sched_d)
            self.epoch = max(meta["epoch"], 1)
            self.global_step = meta.get("global_step") or (self.epoch - 1) * self.steps_per_epoch
            logger.info("resumed (explicit) from %s (epoch=%s step=%s)",
                        explicit, self.epoch, self.global_step)
            self.emit("training.resumed", checkpoint=str(explicit), epoch=self.epoch,
                      global_step=self.global_step, explicit=True)
            return True

        resume_path = checkpointing.latest_step_checkpoint(self.checkpoint_dir)
        latest_g = self.checkpoint_dir / f"G_{checkpointing.LATEST_TAG}.pth"
        latest_d = self.checkpoint_dir / f"D_{checkpointing.LATEST_TAG}.pth"

        if self.config.resume and state and (resume_path or latest_g.is_file()):
            sig = state.get("dataset_signature") or {}
            if dataset_signature and sig and sig != dataset_signature:
                raise CheckpointError(
                    "dataset changed since the checkpoint was written "
                    f"(saved {sig} vs now {dataset_signature}); "
                    "use a new experiment name or set clean_start"
                )
            chosen = resume_path or latest_g
            if resume_path and latest_g.is_file():
                chosen = latest_g if latest_g.stat().st_mtime > resume_path.stat().st_mtime else resume_path
            meta = checkpointing.load_full(
                chosen, net_g, optim_g, sched_g, scaler,
            )
            d_path = latest_d if latest_d.is_file() else Path(str(chosen).replace("G_", "D_"))
            if d_path.is_file():
                checkpointing.load_full(d_path, net_d, optim_d, sched_d)
            self.epoch = max(meta["epoch"], 1)
            self.global_step = meta.get("global_step") or (self.epoch - 1) * self.steps_per_epoch
            logger.info("resumed from %s (epoch=%s step=%s)", chosen, self.epoch, self.global_step)
            self.emit("training.resumed", checkpoint=str(chosen), epoch=self.epoch,
                      global_step=self.global_step)
            return True

        if self.config.resume and latest_g.is_file() and latest_d.is_file():
            # upstream-style checkpoints: model + optimizer + iteration(epoch)
            for path, model, optim in ((latest_g, net_g, optim_g), (latest_d, net_d, optim_d)):
                payload = torch.load(path, map_location="cpu", weights_only=False)
                model.load_state_dict(payload["model"])
                if "optimizer" in payload:
                    optim.load_state_dict(payload["optimizer"])
            epoch = checkpointing.legacy_iteration_epoch(latest_g)
            self.epoch = max(epoch, 1)
            self.global_step = (self.epoch - 1) * self.steps_per_epoch
            logger.info("legacy resume from %s (epoch=%s)", latest_g, self.epoch)
            self.emit("training.resumed.legacy", epoch=self.epoch)
            return True

        self._load_pretrained(net_g, net_d, optim_d)
        return False

    # ------------------------------------------------------------------ run
    def run(self, dataset_signature: dict | None = None) -> bool:
        """Train to configured epoch count. Returns completed (not stopped)."""
        if self.state == JobState.RUNNING:
            raise TrainingError("trainer already running")
        self.dataset_signature = dataset_signature
        torch.manual_seed(self.hps.train.get("seed", 1234))
        (dataset, loader, net_g, net_d,
         optim_g, optim_d) = self._build()
        self.steps_per_epoch = len(loader)
        sched_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=self.hps.train["lr_decay"]
        )
        sched_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=self.hps.train["lr_decay"]
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.is_half)
        self.state = JobState.RUNNING
        started = time.time()
        self.emit("training.started", steps_per_epoch=self.steps_per_epoch,
                  device=self.device, precision=str(self.dtype))

        self._restore(net_g, net_d, optim_g, optim_d, sched_g, sched_d, scaler,
                      dataset_signature)
        start_epoch = self.epoch + 1
        if start_epoch > self.hps.total_epoch:
            logger.info("already past total_epoch (%s > %s) — nothing to do",
                        start_epoch, self.hps.total_epoch)
            self._finalize(net_g, self.epoch, completed=True)
            self.state = JobState.COMPLETED
            return True

        try:
            completed = self._loop(
                net_g, net_d, optim_g, optim_d, sched_g, sched_d, scaler,
                loader, start_epoch,
            )
        except Exception as exc:
            # crash-save: persist an emergency checkpoint so the run stays
            # resumable even after an unexpected failure (OOM, bad batch...)
            self.state = JobState.FAILED
            try:
                self._checkpoint(net_g, net_d, optim_g, optim_d, sched_g, sched_d,
                                 scaler, self.epoch, final=False, emergency=True)
            except Exception:
                logger.exception("emergency crash-save also failed")
            self.emit("training.failed", error=str(exc),
                      emergency_checkpoint=bool(self.global_step))
            logger.exception("training failed")
            raise
        self._finalize(net_g, self.epoch, completed=completed)
        self.state = JobState.COMPLETED if completed else JobState.STOPPED
        self.emit("training.finished", completed=completed,
                  epoch=self.epoch, seconds=round(time.time() - started, 1))
        return completed

    def _loop(self, net_g, net_d, optim_g, optim_d, sched_g, sched_d, scaler,
              loader, start_epoch: int) -> bool:
        hps = self.hps
        for epoch in range(start_epoch, hps.total_epoch + 1):
            self.epoch = epoch
            if self._stop.is_set():
                return False
            loader.batch_sampler.set_epoch(epoch)
            net_g.train()
            net_d.train()
            epoch_start = time.time()
            for batch_idx, info in enumerate(loader):
                if self._stop.is_set():
                    logger.info("stop requested at epoch %s batch %s", epoch, batch_idx)
                    self._checkpoint(net_g, net_d, optim_g, optim_d, sched_g, sched_d,
                                     scaler, epoch, final=False)
                    return False
                if hps.if_f0 == 1:
                    (phone, phone_lengths, pitch, pitchf, spec, spec_lengths,
                     wave, wave_lengths, sid) = info
                else:
                    phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
                if self.device.startswith("cuda"):
                    phone = phone.cuda(non_blocking=True)
                    phone_lengths = phone_lengths.cuda(non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.cuda(non_blocking=True)
                        pitchf = pitchf.cuda(non_blocking=True)
                    sid = sid.cuda(non_blocking=True)
                    spec = spec.cuda(non_blocking=True)
                    spec_lengths = spec_lengths.cuda(non_blocking=True)
                    wave = wave.cuda(non_blocking=True)

                with torch.autocast("cuda" if self.is_half else "cpu",
                                    enabled=self.is_half):
                    if hps.if_f0 == 1:
                        (y_hat, ids_slice, x_mask, z_mask,
                         (z, z_p, m_p, logs_p, m_q, logs_q)) = net_g(
                            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
                        )
                    else:
                        (y_hat, ids_slice, x_mask, z_mask,
                         (z, z_p, m_p, logs_p, m_q, logs_q)) = net_g(
                            phone, phone_lengths, spec, spec_lengths, sid
                        )
                    mel = mel_processing.spec_to_mel_torch(
                        spec, hps.data["filter_length"], hps.data["n_mel_channels"],
                        hps.data["sampling_rate"], hps.data["mel_fmin"],
                        hps.data["mel_fmax"],
                    )
                    y_mel = commons.slice_segments(
                        mel, ids_slice, hps.train["segment_size"] // hps.data["hop_length"]
                    )
                    with torch.autocast("cuda" if self.is_half else "cpu", enabled=False):
                        y_hat_mel = mel_processing.mel_spectrogram_torch(
                            y_hat.float().squeeze(1), hps.data["filter_length"],
                            hps.data["n_mel_channels"], hps.data["sampling_rate"],
                            hps.data["hop_length"], hps.data["win_length"],
                            hps.data["mel_fmin"], hps.data["mel_fmax"],
                        )
                    if self.is_half:
                        y_hat_mel = y_hat_mel.half()
                    wave_sliced = commons.slice_segments(
                        wave, ids_slice * hps.data["hop_length"],
                        hps.train["segment_size"],
                    )
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(wave_sliced, y_hat.detach())
                    with torch.autocast("cuda" if self.is_half else "cpu", enabled=False):
                        loss_disc, losses_disc_r, losses_disc_g = L.discriminator_loss(
                            y_d_hat_r, y_d_hat_g
                        )
                optim_d.zero_grad()
                scaler.scale(loss_disc).backward()
                scaler.unscale_(optim_d)
                commons.clip_grad_value_(net_d.parameters(), None)
                scaler.step(optim_d)

                with torch.autocast("cuda" if self.is_half else "cpu",
                                    enabled=self.is_half):
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave_sliced, y_hat)
                    with torch.autocast("cuda" if self.is_half else "cpu", enabled=False):
                        loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * hps.train["c_mel"]
                        loss_kl = L.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train["c_kl"]
                        loss_fm = L.feature_loss(fmap_r, fmap_g)
                        loss_gen, losses_gen = L.generator_loss(y_d_hat_g)
                        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
                optim_g.zero_grad()
                scaler.scale(loss_gen_all).backward()
                scaler.unscale_(optim_g)
                commons.clip_grad_value_(net_g.parameters(), None)
                scaler.step(optim_g)
                scaler.update()

                self.global_step += 1
                if self.global_step % hps.train["log_interval"] == 0 or batch_idx == 0:
                    lr = optim_g.param_groups[0]["lr"]
                    self.last_losses = {
                        "disc": round(float(loss_disc), 4),
                        "gen": round(float(loss_gen_all), 4),
                        "mel": round(min(float(loss_mel), 75.0), 4),
                        "kl": round(min(float(loss_kl), 9.0), 4),
                        "fm": round(float(loss_fm), 4),
                        "lr": lr,
                    }
                    self.emit("training.progress", **self.progress_payload(),
                              epoch_fraction=epoch / max(1, hps.total_epoch),
                              epoch_seconds=round(time.time() - epoch_start, 1),
                              batch_idx=batch_idx)

            sched_g.step()
            sched_d.step()
            if epoch % hps.save_every_epoch == 0:
                self._checkpoint(net_g, net_d, optim_g, optim_d, sched_g, sched_d,
                                 scaler, epoch, final=False)
            self.emit("training.epoch_done", epoch=epoch,
                      epoch_seconds=round(time.time() - epoch_start, 1))
        # final full checkpoint regardless of save_every_epoch alignment
        self._checkpoint(net_g, net_d, optim_g, optim_d, sched_g, sched_d,
                         scaler, self.epoch, final=True)
        return True

    # ------------------------------------------------------------------ checkpointing
    def _checkpoint(self, net_g, net_d, optim_g, optim_d, sched_g, sched_d,
                    scaler, epoch: int, final: bool, emergency: bool = False) -> dict:
        step_name = self.global_step if not final else "final"
        g_path = self.checkpoint_dir / f"G_{step_name}.pth"
        d_path = self.checkpoint_dir / f"D_{step_name}.pth"
        checkpointing.save_full(g_path, net_g, optim_g, sched_g, scaler,
                                 epoch, self.global_step)
        checkpointing.save_full(d_path, net_d, optim_d, sched_d, None,
                                epoch, self.global_step)
        if int(self.hps.if_latest) == 1:
            # upstream-compatible always-latest mirrors (G_2333333/D_2333333)
            checkpointing.write_latest_mirrors(self.checkpoint_dir, g_path, d_path)
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "saved_at": time.time(),
            "experiment": self.config.experiment_name,
            "dataset_signature": self.dataset_signature,
            "last_losses": self.last_losses,
            "run_config": self.config.model_dump(mode="json"),
        }
        checkpointing.write_state(state, self.checkpoint_dir / "state.json")
        if self.hps.save_every_weights == "1" or final:
            export = engine_ckpt.export_inference_checkpoint(
                net_g.state_dict(), self.hps, epoch, self.hps.version,
                self.exports_dir / f"{self.config.experiment_name}_e{epoch}_s{self.global_step}.pth",
                self.hps.sample_rate, self.hps.if_f0,
            )
            self.emit("training.exported", path=str(export), epoch=epoch)
        logger.info("checkpoint saved: epoch=%s step=%s emergency=%s -> %s", epoch,
                    self.global_step, emergency, g_path)
        return state

    def _finalize(self, net_g, epoch: int, completed: bool) -> None:
        # the loop already saved a full checkpoint at stop/completion;
        # here we only export the final inference model when completed.
        if completed:
            export = engine_ckpt.export_inference_checkpoint(
                net_g.state_dict(), self.hps, epoch, self.hps.version,
                self.exports_dir / f"{self.config.experiment_name}.pth",
                self.hps.sample_rate, self.hps.if_f0,
            )
            self.emit("training.final_model", path=str(export))

    # ------------------------------------------------------------------ control
    def request_stop(self) -> None:
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()
