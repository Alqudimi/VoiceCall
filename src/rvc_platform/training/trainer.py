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
  * progress events (epoch/step/losses/lr/eta) via callback;
  * ALL disk writes (checkpoints, mirrors, state.json, exports) and the
    best-model bookkeeping run through a background I/O worker — the loop
    only pays for a CPU snapshot of live state, never for serialization;
  * best-so-far model tracking: per-epoch held-out validation (or the
    epoch-mean training mel loss as fallback) with a persisted best bar
    (checkpoints/best.json) and best-model inference exports.
"""
from __future__ import annotations

import logging
import os
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
from rvc_platform.training import background_io
from rvc_platform.training import best_model
from rvc_platform.training import checkpointing
from rvc_platform.training import losses as L
from rvc_platform.training import mel_processing
from rvc_platform.training import stft_loss
from rvc_platform.training.augment import build_augmenter
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
        best_export_cb: Callable[[str, dict], None] | None = None,
        io_worker: background_io.BackgroundIOWorker | None = None,
    ) -> None:
        self.config = config
        self.hps = hps
        self.device = device
        self.dtype = training_dtype
        self.is_half = training_dtype == torch.float16
        self._event = event_cb or (lambda payload: None)
        self._best_export_cb = best_export_cb
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
        # background I/O: every disk write leaves the training thread here
        self._io = io_worker or background_io.BackgroundIOWorker(
            name=f"io-{getattr(config, 'experiment_name', 'trainer')}",
        )
        # best-model tracking state
        self._tracker: best_model.BestModelTracker | None = None
        self._validator: best_model.ValidationEvaluator | None = None
        self._epoch_mel_acc: torch.Tensor | None = None
        self._epoch_start_step = 0
        # GAN-health watcher state (see _after_epoch)
        self._val_kl_history: list[float] = []
        # researched quality levers (see UPGRADE_REPORT_AR.md):
        # generator EMA shadow, multi-scale discriminator + its optimizer/
        # scheduler (D2), early-stop bookkeeping
        self._ema = None                     # ema.GeneratorEMA | None
        self._msd = None                     # msd.MultiScaleDiscriminator | None
        self._optim_d2: torch.optim.Optimizer | None = None
        self._sched_d2: torch.optim.lr_scheduler.LRScheduler | None = None
        self._early_stopped = False
        # Phase-2 researched quality levers: online augmentation + the
        # warmup/cosine factor schedules layered on G/D
        self._augmenter = None               # augment.WaveformAugmenter | None
        self._factor_g: torch.optim.lr_scheduler.LRScheduler | None = None
        self._factor_d: torch.optim.lr_scheduler.LRScheduler | None = None
        self._factor_d2: torch.optim.lr_scheduler.LRScheduler | None = None

    # ------------------------------------------------------------------ events
    def emit(self, topic: str, **payload) -> None:
        self._event({"topic": topic, **payload})

    def progress_payload(self) -> dict:
        best = self._tracker.best.to_dict() if self._tracker and self._tracker.best else None
        return {
            "state": self.state.value,
            "epoch": self.epoch,
            "total_epochs": self.hps.total_epoch,
            "global_step": self.global_step,
            "steps_per_epoch": self.steps_per_epoch,
            "latest_losses": self.last_losses,
            "best": best,
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
        use_cuda = self.device.startswith("cuda")
        # Fast defaults merged with env-var overrides: workers auto-scale with
        # the machine (Colab free T4 has 2 vCPU -> 2 workers; L4/A100 runtimes
        # get more), stay alive across epochs (fork cost paid once) and
        # prefetch ahead. RVC_NUM_WORKERS / RVC_PERSISTENT_WORKERS /
        # RVC_PREFETCH_FACTOR let anyone benchmark or clamp these per session
        # without touching code (sampler order is unaffected either way).
        cpu = os.cpu_count() or 2
        default_workers = min(6, max(2, cpu)) if use_cuda else 0
        num_workers = int(os.environ.get("RVC_NUM_WORKERS", str(default_workers)))
        num_workers = max(0, min(num_workers, 16))
        persistent = (
            os.environ.get("RVC_PERSISTENT_WORKERS", "1") == "1" and num_workers > 0
        )
        loader_kwargs = dict(num_workers=num_workers, persistent_workers=persistent)
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = max(
                1, min(int(os.environ.get("RVC_PREFETCH_FACTOR", "4")), 8)
            )
        loader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=use_cuda,
            collate_fn=collate,
            batch_sampler=sampler,
            **loader_kwargs,
        )
        net_g = net_g.to(self.device)
        net_d = disc_cls(hps.model["use_spectral_norm"]).to(self.device)
        optim_g = self._make_optimizer(net_g, hps)
        d_lr = float(hps.train["learning_rate"]) * self._d_lr_scale(hps)
        optim_d = self._make_optimizer(net_d, hps, lr=d_lr)
        # extra coarse-scale raw discriminators (HiFi-GAN MSD scales 2/4) with
        # their own optimizer — G/D checkpoints and pretrained bases untouched
        if self._use_msd():
            from rvc_platform.training import msd as msd_mod

            self._msd = msd_mod.MultiScaleDiscriminator().to(self.device)
            self._optim_d2 = self._make_optimizer(self._msd, hps, lr=d_lr)
            logger.info("multi-scale discriminator enabled (scales=%s) as D2",
                        getattr(self._msd, "scales", (2, 4)))
        return dataset, loader, net_g, net_d, optim_g, optim_d

    def _use_msd(self) -> bool:
        """use_msd from run config, overridable per session with RVC_MSD=0."""
        env = os.environ.get("RVC_MSD", "").strip().lower()
        if env in ("0", "off", "false", "no"):
            return False
        return bool(int(self.hps.train.get("use_msd", 1)))

    @staticmethod
    def _ema_decay(hps) -> float:
        try:
            decay = float(hps.train.get("ema_decay", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return decay if 0.0 < decay < 1.0 else 0.0

    @staticmethod
    def _d_lr_scale(hps) -> float:
        """Discriminator LR multiplier (1.0 = upstream parity).

        Research (see UPGRADE_REPORT_AR.md §2.10): RVC fine-tuning starts
        from a pretrained G/D pair; a slightly slower discriminator keeps
        the inherited G/D balance longer at the start of fine-tuning (the
        same rationale as lower-D-lr experiments in GAN literature)."""
        try:
            scale = float(hps.train.get("d_lr_scale", 1.0) or 1.0)
        except (TypeError, ValueError):
            return 1.0
        return scale if 0.0 < scale <= 4.0 else 1.0

    @staticmethod
    def _make_optimizer(module: torch.nn.Module, hps, lr: float | None = None) -> torch.optim.AdamW:
        """AdamW with the fused CUDA kernel when available.

        The fused implementation is the same update rule executed in a single
        kernel launch (vertically fused), and it natively supports AMP
        gradient scaling (checks inf/unscale inside the kernel). Falls back to
        the default implementation anywhere else.
        """
        kwargs = dict(betas=hps.train["betas"], eps=hps.train["eps"])
        if lr is None:
            lr = hps.train["learning_rate"]
        if next(module.parameters()).is_cuda:
            try:
                return torch.optim.AdamW(
                    module.parameters(), lr,
                    fused=True, **kwargs,
                )
            except (TypeError, RuntimeError):
                logger.info("fused AdamW unavailable — default implementation")
        return torch.optim.AdamW(
            module.parameters(), lr, **kwargs
        )

    @staticmethod
    def _unwrap(module) -> torch.nn.Module:
        """Underlying module of a torch.compile / _CompileGuard wrapper
        (weights are shared)."""
        if isinstance(module, Trainer._CompileGuard):
            return module._raw
        return getattr(module, "_orig_mod", module)

    @staticmethod
    def _is_compile_kernel_error(exc: BaseException) -> bool:
        """Runtime errors raised from an Inductor-compiled graph.

        dynamo's suppress_errors only guards TRACE-time failures; a kernel
        that compiles fine but crashes at runtime (e.g. torch 2.7's
        assert_size_stride layout assert on the f0 NSF path) propagates and
        kills the run. Recognised by the inductor/dynamo frames in the
        traceback — anything raised from there is a compiled-kernel defect.
        """
        import traceback

        tb = traceback.extract_tb(exc.__traceback__)
        return any(
            "torchinductor" in (frame.filename or "")
            or "_inductor" in (frame.filename or "")
            or "torch/_dynamo" in (frame.filename or "")
            for frame in reversed(tb)
        )

    class _CompileGuard:
        """Call-time safety net around a torch.compile'd network.

        Deliberately NOT an nn.Module (registering the networks as submodules
        would prefix checkpoint keys); a transparent proxy instead. The first
        failed call with a compiled-kernel defect permanently swaps that
        network to eager — training continues, a clear warning names the
        offending net, and later calls bypass the broken graph for free.
        Everything except __call__ (train/eval/parameters/...) delegates to
        the raw module, whose weights are shared with the compiled graph.
        """

        def __init__(self, compiled, raw, label: str):
            self._compiled = compiled
            self._raw = raw
            self._label = label
            self._fell_back = False

        def __call__(self, *args, **kwargs):
            if self._fell_back:
                return self._raw(*args, **kwargs)
            try:
                return self._compiled(*args, **kwargs)
            except Exception as exc:
                if not Trainer._is_compile_kernel_error(exc):
                    raise
                self._fell_back = True
                logger.warning(
                    "torch.compile kernel failed on %s (%s: %s) — permanently "
                    "falling back to eager execution for this network; "
                    "training continues unaffected",
                    self._label, type(exc).__name__, exc,
                )
                return self._raw(*args, **kwargs)

        def __getattr__(self, name):
            # _compiled/_raw/_label/_fell_back live in __dict__; everything
            # else (train, eval, parameters, state_dict, ...) hits the raw
            # module — the guard is invisible to the rest of the trainer.
            return getattr(self.__dict__["_raw"], name)

    def _maybe_compile(self, net_g, net_d):
        """Optional torch.compile for both networks.

        Default "0" (disabled): Set RVC_TORCH_COMPILE=1 to force on.
        Audio training involves dynamic sequence lengths per bucket, which causes
        PyTorch Inductor / Triton to recompile graphs repeatedly, freezing training
        and pinning the CPU on SymPy/LLVM codegen. Eager mode on CUDA uses
        cuDNN pre-compiled kernels directly with 0 compilation latency.
        Checkpoints keep byte-identical keys because state dicts are always
        read from the unwrapped module (see _unwrap / _checkpoint).
        """
        choice = os.environ.get("RVC_TORCH_COMPILE", "0").strip().lower()
        if choice not in ("1", "on", "true", "yes"):
            logger.info("torch.compile disabled (eager execution mode active for maximum audio training throughput)")
            return net_g, net_d
        try:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
            # bucketing yields a handful of distinct shape sets; allow one
            # compiled graph per shape instead of reverting to eager at the
            # default recompile limit (8).
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit, 32
            )
            compiled_g = torch.compile(net_g)
            compiled_d = torch.compile(net_d)
            guarded_g = self._CompileGuard(compiled_g, net_g, "generator")
            guarded_d = self._CompileGuard(compiled_d, net_d, "discriminator")
            logger.info("torch.compile enabled for G and D "
                        "(trace + runtime eager fallback)")
            return guarded_g, guarded_d
        except Exception:
            logger.warning("torch.compile unavailable — continuing in eager mode",
                           exc_info=True)
            return net_g, net_d

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
                 dataset_signature: dict | None) -> tuple[bool, dict]:
        """Explicit > full resume > legacy resume > pretrained start.
        Returns (resumed?, extras) — extras carries checkpoint-embedded EMA /
        D2 (MSD) states so the caller can wire the quality levers without
        re-loading the payloads."""
        from rvc_platform.training.checkpointing import read_state

        extras: dict = {}
        if self.config.clean_start:
            self.config.resume = False
            self.config.resume_from = None
            logger.info("clean start requested; ignoring all existing checkpoints")
            return False, extras

        state = read_state(self.checkpoint_dir / "state.json")

        explicit = self._resolve_explicit()
        if explicit is not None:
            # policy 1: an explicit checkpoint wins over auto-discovery
            meta = checkpointing.load_full(explicit, net_g, optim_g, sched_g, scaler)
            d_path = (self.checkpoint_dir / explicit.name.replace("G_", "D_")
                      if explicit.name.startswith("G_") else None)
            if d_path is not None and d_path.is_file():
                checkpointing.load_full(d_path, net_d, optim_d, sched_d)
                self._restore_d2(d_path.with_name(d_path.name.replace("D_", "D2_")))
            self.epoch = max(meta["epoch"], 1)
            self.global_step = meta.get("global_step") or (self.epoch - 1) * self.steps_per_epoch
            extras["ema"] = meta.get("ema")
            logger.info("resumed (explicit) from %s (epoch=%s step=%s)",
                        explicit, self.epoch, self.global_step)
            self.emit("training.resumed", checkpoint=str(explicit), epoch=self.epoch,
                      global_step=self.global_step, explicit=True)
            return True, extras

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
                self._restore_d2(d_path.with_name(d_path.name.replace("D_", "D2_")))
            self.epoch = max(meta["epoch"], 1)
            self.global_step = meta.get("global_step") or (self.epoch - 1) * self.steps_per_epoch
            extras["ema"] = meta.get("ema")
            logger.info("resumed from %s (epoch=%s step=%s)", chosen, self.epoch, self.global_step)
            self.emit("training.resumed", checkpoint=str(chosen), epoch=self.epoch,
                      global_step=self.global_step)
            return True, extras

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
            return True, extras

        self._load_pretrained(net_g, net_d, optim_d)
        return False, extras

    def _restore_d2(self, d2_path: Path | None) -> None:
        """Restore the optional multi-scale discriminator (D2) from its own
        D2_<step>.pth; missing files (legacy runs) simply warm-start D2."""
        if self._msd is None:
            return
        if d2_path is None or not Path(d2_path).is_file():
            logger.info("D2 (multi-scale discriminator) warm-started fresh "
                        "(no D2 checkpoint)")
            return
        try:
            payload = torch.load(Path(d2_path), map_location="cpu",
                                 weights_only=False)
            self._msd.load_state_dict(payload["model"])
            if self._optim_d2 is not None and payload.get("optimizer"):
                self._optim_d2.load_state_dict(payload["optimizer"])
            if (self._sched_d2 is not None and payload.get("scheduler")
                    and isinstance(self._sched_d2,
                                   torch.optim.lr_scheduler.LRScheduler)):
                self._sched_d2.load_state_dict(payload["scheduler"])
            logger.info("D2 (multi-scale discriminator) state restored from %s",
                        d2_path)
        except Exception:
            logger.warning("D2 restore failed — continuing with fresh D2",
                           exc_info=True)

    # ----------------------------------------------------------- best tracking
    def _setup_best_tracking(self) -> None:
        """Derive the validation split (pinned on disk), build the validator
        and restore the persisted best bar. Must run before _build() so the
        training loader sees filelist_train.txt."""
        if not getattr(self.config, "save_best", True):
            return
        val_split = float(getattr(self.config, "val_split", 0.0) or 0.0)
        split = None
        validator = None
        if val_split > 0:
            training_files = Path(self.hps.data["training_files"])
            split = best_model.derive_validation_split(training_files, val_split)
            if split is not None and split.n_val > 0:
                # train on the held-out-free manifest from now on
                self.hps.data["training_files"] = str(split.train_path)
                try:
                    validator = best_model.ValidationEvaluator(
                        self.hps, self.device, self.is_half, split.val_path,
                    )
                except Exception:
                    logger.exception("validation evaluator unavailable; "
                                     "falling back to train-mel scoring")
                    validator = None
        # selection metric, strongest available:
        #   composite     (val_mel + spk_weight*(1-val_spk))   — speechbrain present
        #   composite_f0  (val_mel + f0_weight*val_f0_rmse/100) — no speechbrain
        #   val_mel                                              — no validator
        # val_mel ALONE tracks spectral smoothness and is deaf to speaker
        # identity — the v1.3.2 defect: best models sounded far from the
        # target speaker. The F0 composite keeps selection honest without
        # the optional dependency (both are "lower is better").
        if validator is not None and validator.has_speaker:
            metric = "composite"
        elif validator is not None:
            metric = "composite_f0"
            # only trustworthy while the evaluator actually produces it
            # (f0 eval needs the rmvpe asset; checked in _after_epoch)
        else:
            metric = "val_mel" if validator is not None else "train_mel"
        self._tracker = best_model.BestModelTracker(
            self.checkpoint_dir / best_model.BEST_STATE_NAME,
            metric=metric, mode="min",
        )
        had_best = self._tracker.load()
        if had_best and self._tracker.best and self._tracker.best.metric != metric:
            # scores of different metrics are not comparable — a resumed run
            # whose metric changed (e.g. val_mel -> composite_f0 after this
            # release) starts the best bar fresh instead of freezing on a
            # lower-scale legacy score forever
            old_metric = self._tracker.best.metric
            self._tracker.reset()
            logger.info("best-model metric changed (%s -> %s); best bar reset",
                        old_metric, metric)
        if split is not None and split.rederived and had_best:
            # scores across different data splits are not comparable
            self._tracker.reset()
            logger.info("validation split changed; best-model bar reset")
        self._validator = validator
        logger.info("best-model tracking enabled: metric=%s (validator=%s, "
                    "speaker-sim=%s, f0-eval=%s)",
                    metric, validator is not None,
                    bool(validator and validator.has_speaker),
                    bool(validator and validator.f0_fn is not None))

    # ------------------------------------------------------------- EMA helpers
    def _eval_weights(self, net_g, fn):
        """Run ``fn(module)`` against the EMA weights when enabled.

        EMA weights are swapped into the (unwrapped) module for the duration
        of the call and the live training weights are restored right after —
        the training loop itself never sees the shadow copy."""
        raw = self._unwrap(net_g)
        if self._ema is None:
            return fn(raw)
        backup = background_io.snapshot_to_cpu(raw.state_dict())
        self._ema.apply_to(raw)
        try:
            return fn(raw)
        finally:
            raw.load_state_dict(backup)

    def _export_snapshot(self, net_g) -> dict:
        """CPU state dict for inference exports: EMA weights when enabled."""
        raw = self._unwrap(net_g)
        if self._ema is None:
            return background_io.snapshot_to_cpu(raw.state_dict())
        return background_io.snapshot_to_cpu(self._ema.export_state_dict(raw))

    def _after_epoch(self, net_g, epoch: int) -> None:
        """Score the epoch, update the best bar, export on improvement.

        With EMA enabled the epoch is scored and exported against the EMA
        weights (the BigVGAN recipe — the averaged generator is the model you
        ship), while training itself continues on the raw weights."""
        if self._tracker is None:
            return
        scores: dict = {}
        if self._validator is not None:
            try:
                scores = self._eval_weights(net_g, self._validator.evaluate)
            except Exception:
                logger.exception("validation failed; using train-mel fallback")
                scores = {}
        if not scores:
            steps = max(1, self.global_step - self._epoch_start_step)
            if self._epoch_mel_acc is None:
                return
            c_mel = float(self.hps.train.get("c_mel", 1.0) or 1.0)
            scores = {"train_mel": float(self._epoch_mel_acc) / steps / c_mel}
        # metric degradation guard: composite_f0 needs the rmvpe asset; when
        # the evaluator could not produce it, fall back to val_mel for THIS
        # epoch so a missing optional metric never freezes the best bar
        if (self._tracker.metric == "composite_f0"
                and "composite_f0" not in scores and "val_mel" in scores):
            scores["composite_f0"] = scores["val_mel"]
        # GAN health monitor: a steadily CLIMBING val_kl with flat val_mel is
        # the signature of discriminator overwinning — the generator drifts
        # away from the data manifold (audio sounds "far from the speaker")
        # while the mel proxy still improves. Surface it once it is clear.
        kl = scores.get("val_kl")
        if kl is not None:
            hist = self._val_kl_history
            hist.append(float(kl))
            if len(hist) == 10:
                first, last = hist[0], hist[-1]
                if last > first * 1.5 and last > first:
                    logger.warning(
                        "GAN health: val_kl climbed %.2f -> %.2f over the last "
                        "10 epochs while val_mel stayed flat — the discriminator "
                        "is winning; consider --d-lr-scale 0.5 (slower D), a "
                        "lower --lr, or fewer epochs (early stop is armed)",
                        first, last,
                    )
                self._val_kl_history.clear()
        info = self._tracker.update(scores, epoch, self.global_step)
        best = self._tracker.best.to_dict() if self._tracker.best else None
        self.emit("training.validation", epoch=epoch,
                  metric=self._tracker.metric,
                  scores={k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in scores.items()},
                  best=best)
        if info is not None:
            logger.info("new best model: epoch=%s %s=%.6f",
                        epoch, info.metric, info.score)
            self._export_best(net_g, info)

    def _export_best(self, net_g, info: best_model.BestModelInfo) -> None:
        """Snapshot G and queue the best-model export + best.json on the
        background I/O worker; the on_done chain hands the landed file to the
        HF publisher (via best_export_cb) only after it exists on disk.

        The export carries the EMA weights when enabled."""
        snapshot = self._export_snapshot(net_g)
        out_path = self.exports_dir / (
            f"{self.config.experiment_name}_best_e{info.epoch}_s{info.global_step}.pth"
        )
        metadata = {
            **info.to_dict(),
            "experiment": self.config.experiment_name,
            "sample_rate": self.hps.sample_rate,
            "version": self.hps.version,
            "f0": bool(int(self.hps.if_f0)),
            "export": out_path.name,
        }
        best_json = self.checkpoint_dir / best_model.BEST_STATE_NAME

        def _on_done(_export: Path) -> None:
            self.emit("training.best_model", path=str(out_path),
                      epoch=info.epoch, global_step=info.global_step,
                      metric=info.metric, score=info.score)
            if self._best_export_cb is not None:
                try:
                    self._best_export_cb(str(out_path), metadata)
                except Exception:
                    logger.exception("best-export callback failed")

        self._io.submit(
            f"best-export-e{info.epoch}",
            best_model.write_best_artifacts,
            snapshot, self.hps, metadata, out_path, best_json,
            on_done=_on_done,
        )

    # ------------------------------------------------------------------ run
    def run(self, dataset_signature: dict | None = None) -> bool:
        """Train to configured epoch count. Returns completed (not stopped)."""
        if self.state == JobState.RUNNING:
            raise TrainingError("trainer already running")
        self.dataset_signature = dataset_signature
        torch.manual_seed(self.hps.train.get("seed", 1234))
        if self.device.startswith("cuda"):
            # cuDNN autotuner: the bucket sampler yields a small, repeating
            # set of shapes, so benchmarking cost is paid once per shape.
            # TF32 matmuls ("high") engage on Ampere+ (L4/A100); a no-op on
            # T4. Neither changes the training math beyond float rounding,
            # and the fp16 autocast paths are untouched.
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:  # pragma: no cover — very old torch
                pass
        self._setup_best_tracking()
        (dataset, loader, net_g, net_d,
         optim_g, optim_d) = self._build()
        self.steps_per_epoch = len(loader)
        sched_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=self.hps.train["lr_decay"]
        )
        sched_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=self.hps.train["lr_decay"]
        )
        if self._optim_d2 is not None:
            self._sched_d2 = torch.optim.lr_scheduler.ExponentialLR(
                self._optim_d2, gamma=self.hps.train["lr_decay"],
            )
        # Phase-2: warmup (+optional cosine tail) factor schedule layered on
        # the exponential decay — the factor multiplies the base lr per step.
        warmup_epochs = self._warmup_epochs()
        cosine_frac = self._cosine_frac()
        if warmup_epochs > 0 or cosine_frac > 0:
            from rvc_platform.training.schedules import WarmupCosineScheduler

            mk = lambda opt: WarmupCosineScheduler(  # noqa: E731
                opt, steps_per_epoch=self.steps_per_epoch,
                warmup_epochs=warmup_epochs, total_epochs=self.hps.total_epoch,
                cosine_frac=cosine_frac,
            )
            self._factor_g = mk(optim_g)
            self._factor_d = mk(optim_d)
            if self._optim_d2 is not None:
                self._factor_d2 = mk(self._optim_d2)
            logger.info("warmup/cosine factor schedule enabled: warmup=%s "
                        "epochs, cosine_frac=%s", warmup_epochs, cosine_frac)
        # Phase-2: online waveform augmentation (None when disabled)
        self._augmenter = build_augmenter(self.hps, self.device)
        scaler = torch.amp.GradScaler("cuda", enabled=self.is_half)
        self.state = JobState.RUNNING
        started = time.time()
        self.emit("training.started", steps_per_epoch=self.steps_per_epoch,
                  device=self.device, precision=str(self.dtype))

        _, extras = self._restore(net_g, net_d, optim_g, optim_d, sched_g, sched_d, scaler,
                                  dataset_signature)
        self._init_ema(net_g, extras.get("ema"))
        net_g, net_d = self._maybe_compile(net_g, net_d)
        start_epoch = self.epoch + 1
        flush_timeout = float(os.environ.get("RVC_IO_FLUSH_TIMEOUT", "900"))
        completed: bool
        try:
            if start_epoch > self.hps.total_epoch:
                logger.info("already past total_epoch (%s > %s) — nothing to do",
                            start_epoch, self.hps.total_epoch)
                completed = True
            else:
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
                        self._checkpoint(net_g, net_d, optim_g, optim_d, sched_g,
                                         sched_d, scaler, self.epoch,
                                         final=False, emergency=True)
                    except Exception:
                        logger.exception("emergency crash-save also failed")
                    # the crash-save must land on disk before we report failure
                    self._io.wait_idle(60.0)
                    self.emit("training.failed", error=str(exc),
                              emergency_checkpoint=bool(self.global_step))
                    logger.exception("training failed")
                    raise
            self._finalize(net_g, self.epoch, completed=completed)
        finally:
            # drain every queued write (checkpoints/mirrors/state/exports)
            # before the run ends — nothing is left to a daemon-thread race
            self._io.wait_idle(flush_timeout)
            self._io.close(timeout=60.0)
        self.state = JobState.COMPLETED if completed else JobState.STOPPED
        self.emit("training.finished", completed=completed,
                  epoch=self.epoch, seconds=round(time.time() - started, 1))
        return completed

    def _init_ema(self, net_g, saved_state: dict | None) -> None:
        """Create the generator EMA after weights are restored/initialised;
        resume continues from the checkpointed shadow (bit-exact)."""
        decay = self._ema_decay(self.hps)
        if decay <= 0.0:
            self._ema = None
            return
        from rvc_platform.training import ema as ema_mod

        self._ema = ema_mod.GeneratorEMA(net_g, decay=decay)
        if saved_state:
            try:
                device = next(net_g.parameters()).device
                self._ema.load_state_dict(saved_state, device=device)
            except Exception:
                logger.warning("EMA restore failed — re-initialised from weights",
                               exc_info=True)

    def _warmup_epochs(self) -> int:
        """Linear warmup epochs for fine-tune stability (0 = off)."""
        env = os.environ.get("RVC_WARMUP_EPOCHS")
        if env is not None and env.strip():
            try:
                return max(0, int(env))
            except ValueError:
                pass
        try:
            return max(0, int(self.hps.train.get("warmup_epochs", 1) or 0))
        except (TypeError, ValueError):
            return 0

    def _cosine_frac(self) -> float:
        """Cosine-tail target lr fraction (0 = pure exponential decay)."""
        env = os.environ.get("RVC_COSINE_FRAC")
        if env is not None and env.strip():
            try:
                return min(1.0, max(0.0, float(env)))
            except ValueError:
                pass
        try:
            return min(1.0, max(0.0, float(self.hps.train.get("cosine_frac", 0.0) or 0.0)))
        except (TypeError, ValueError):
            return 0.0

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
            self._epoch_start_step = self.global_step
            self._epoch_mel_acc = None
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

                # Phase-2: online augmentation of the REAL waveform the
                # discriminators score (spec target untouched — the mel/STFT
                # reconstruction losses always compare against the pristine
                # ground truth, so the generator is never asked to reproduce
                # noise; only D's "real" distribution is regularised).
                if self._augmenter is not None:
                    wave = self._augmenter.augment(wave)

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
                    if self._msd is not None:
                        # HiFi-GAN MSD scales 2/4 — same loss functions consume
                        # the concatenated output lists (zip iteration)
                        y2_r, y2_g, _, _ = self._msd(wave_sliced, y_hat.detach())
                        y_d_hat_r = y_d_hat_r + y2_r
                        y_d_hat_g = y_d_hat_g + y2_g
                    with torch.autocast("cuda" if self.is_half else "cpu", enabled=False):
                        loss_disc, losses_disc_r, losses_disc_g = L.discriminator_loss(
                            y_d_hat_r, y_d_hat_g
                        )
                optim_d.zero_grad(set_to_none=True)
                if self._optim_d2 is not None:
                    self._optim_d2.zero_grad(set_to_none=True)
                scaler.scale(loss_disc).backward()
                scaler.unscale_(optim_d)
                # upstream "clips" with value None — a no-op whose per-parameter
                # .item() norm bookkeeping forces a GPU->CPU sync every step;
                # skipped with identical numerics (no clamping happens anyway).
                scaler.step(optim_d)
                if self._optim_d2 is not None:
                    scaler.unscale_(self._optim_d2)
                    scaler.step(self._optim_d2)

                with torch.autocast("cuda" if self.is_half else "cpu",
                                    enabled=self.is_half):
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave_sliced, y_hat)
                    if self._msd is not None:
                        y2_r, y2_g, f2_r, f2_g = self._msd(wave_sliced, y_hat)
                        y_d_hat_r = y_d_hat_r + y2_r
                        y_d_hat_g = y_d_hat_g + y2_g
                        fmap_r = fmap_r + f2_r
                        fmap_g = fmap_g + f2_g
                    with torch.autocast("cuda" if self.is_half else "cpu", enabled=False):
                        loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * hps.train["c_mel"]
                        loss_kl = L.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train["c_kl"]
                        loss_fm = L.feature_loss(fmap_r, fmap_g)
                        loss_gen, losses_gen = L.generator_loss(y_d_hat_g)
                        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
                        # multi-resolution STFT auxiliary loss (UnivNet/BigVGAN
                        # recipe; c_stft=0 reproduces the upstream objective)
                        c_stft = float(hps.train.get("c_stft", 0.0) or 0.0)
                        loss_stft = None
                        if c_stft > 0.0:
                            loss_stft = stft_loss.mr_stft_loss(
                                wave_sliced, y_hat) * c_stft
                            loss_gen_all = loss_gen_all + loss_stft
                if self._tracker is not None and self._validator is None:
                    # GPU-side accumulation for the train_mel fallback metric;
                    # a single host sync happens at the end of the epoch
                    detached = loss_mel.detach()
                    self._epoch_mel_acc = (detached if self._epoch_mel_acc is None
                                           else self._epoch_mel_acc + detached)
                optim_g.zero_grad(set_to_none=True)
                scale_before = scaler.get_scale()
                scaler.scale(loss_gen_all).backward()
                scaler.unscale_(optim_g)
                # optional global-norm gradient clipping on G (BigVGAN §3.4:
                # "clipping the global norm of the gradient to 10^3 ...
                # prevents the early training collapse of the generator";
                # their v2 uses 500 and recommends ~100 for small-batch
                # starts). 0.0 = disabled = exact upstream behaviour.
                clip_g = float(hps.train.get("clip_grad_norm", 0.0) or 0.0)
                if clip_g > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        net_g.parameters(), clip_g)
                scaler.step(optim_g)
                scaler.update()
                if self._ema is not None:
                    # a reduced scale means the G step was skipped (inf grads);
                    # only fold real updates into the shadow
                    stepped = (not self.is_half) or scaler.get_scale() >= scale_before
                    if stepped:
                        self._ema.update(self._unwrap(net_g))

                self.global_step += 1
                # Phase-2: warmup/cosine factor schedule steps per training
                # step (per-step granularity, unlike the per-epoch
                # ExponentialLR). Stepped AFTER the optimizer so the first
                # step of the run uses the warmup factor of step 0.
                if self._factor_g is not None:
                    self._factor_g.step()
                if self._factor_d is not None:
                    self._factor_d.step()
                if self._factor_d2 is not None:
                    self._factor_d2.step()
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
                    if self._factor_g is not None:
                        self.last_losses["lr_factor"] = round(
                            float(self._factor_g._factor(self._factor_g.last_epoch)), 6)
                    if loss_stft is not None:
                        self.last_losses["stft"] = round(float(loss_stft), 4)
                    if self._ema is not None:
                        self.last_losses["ema_decay"] = round(
                            self._ema._current_decay(), 6)
                    self.emit("training.progress", **self.progress_payload(),
                              epoch_fraction=epoch / max(1, hps.total_epoch),
                              epoch_seconds=round(time.time() - epoch_start, 1),
                              batch_idx=batch_idx)

            sched_g.step()
            sched_d.step()
            if self._sched_d2 is not None:
                self._sched_d2.step()
            # score the epoch (held-out validation or train-mel fallback) and
            # export/publish the best-so-far model when the bar is beaten
            self._after_epoch(net_g, epoch)
            # early stop: patience epochs without a new best -> cooperative stop
            # (the researched answer to RVC "overtraining" degradation)
            if self._early_stop_due(epoch):
                self._early_stopped = True
                self.emit("training.early_stop", epoch=epoch,
                          patience=self._early_stop_patience(),
                          best=(self._tracker.best.to_dict()
                                if self._tracker is not None and self._tracker.best else None))
                self._checkpoint(net_g, net_d, optim_g, optim_d, sched_g, sched_d,
                                 scaler, epoch, final=True, early_stop=True)
                return False
            if epoch % hps.save_every_epoch == 0:
                self._checkpoint(net_g, net_d, optim_g, optim_d, sched_g, sched_d,
                                 scaler, epoch, final=False)
            self.emit("training.epoch_done", epoch=epoch,
                      epoch_seconds=round(time.time() - epoch_start, 1))
        # final full checkpoint regardless of save_every_epoch alignment
        self._checkpoint(net_g, net_d, optim_g, optim_d, sched_g, sched_d,
                         scaler, self.epoch, final=True)
        return True

    def _early_stop_patience(self) -> int:
        """Epochs without a new best before a cooperative stop (0 = off)."""
        try:
            return max(0, int(self.hps.train.get("early_stop_patience", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _early_stop_due(self, epoch: int) -> bool:
        patience = self._early_stop_patience()
        if patience <= 0 or self._tracker is None or self._tracker.best is None:
            return False
        return (epoch - int(self._tracker.best.epoch)) >= patience

    # ------------------------------------------------------------------ checkpointing
    def _checkpoint(self, net_g, net_d, optim_g, optim_d, sched_g, sched_d,
                    scaler, epoch: int, final: bool, emergency: bool = False,
                    early_stop: bool = False) -> dict:
        """Queue a full-state checkpoint on the background I/O worker.

        The only work on the training thread is building CPU snapshots of the
        G/D state (build_full_payload); serialization, mirror refreshes and
        the state.json update happen off-thread, strictly FIFO."""
        step_name = self.global_step if not final else "final"
        g_path = self.checkpoint_dir / f"G_{step_name}.pth"
        d_path = self.checkpoint_dir / f"D_{step_name}.pth"
        net_g_raw = self._unwrap(net_g)   # compiled wrappers share raw weights
        net_d_raw = self._unwrap(net_d)
        payload_g = checkpointing.build_full_payload(
            net_g_raw, optim_g, sched_g, scaler, epoch, self.global_step)
        payload_d = checkpointing.build_full_payload(
            net_d_raw, optim_d, sched_d, None, epoch, self.global_step)
        if self._ema is not None:
            # EMA rides inside the G payload — resume is bit-exact and legacy
            # readers simply see (and ignore) one extra key
            payload_g["ema"] = self._ema.state_dict()
        payload_d2 = None
        d2_path = None
        if self._msd is not None and self._optim_d2 is not None:
            d2_path = self.checkpoint_dir / f"D2_{step_name}.pth"
            payload_d2 = checkpointing.build_full_payload(
                self._msd, self._optim_d2, self._sched_d2, None, epoch,
                self.global_step,
            )
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "saved_at": time.time(),
            "experiment": self.config.experiment_name,
            "dataset_signature": self.dataset_signature,
            "last_losses": self.last_losses,
            "run_config": self.config.model_dump(mode="json"),
            "early_stopped": bool(early_stop or self._early_stopped),
            "best": (self._tracker.best.to_dict()
                     if self._tracker is not None and self._tracker.best is not None
                     else None),
        }
        mirror = int(self.hps.if_latest) == 1
        self._io.submit(
            f"checkpoint-e{epoch}{'+emergency' if emergency else ''}",
            checkpointing.write_checkpoint_files,
            payload_g, g_path, payload_d, d_path,
            state, self.checkpoint_dir / "state.json", mirror,
            payload_d2, d2_path,
        )
        if self.hps.save_every_weights == "1" or final:
            export_path = self.exports_dir / (
                f"{self.config.experiment_name}_e{epoch}_s{self.global_step}.pth"
            )

            def _on_export(export: Path) -> None:
                # fire only after the file exists on disk — the worker layer
                # reacts to this event by queueing the HF upload
                self.emit("training.exported", path=str(export), epoch=epoch)

            self._io.submit(
                f"export-e{epoch}",
                engine_ckpt.export_inference_checkpoint,
                self._export_snapshot(net_g), self.hps, epoch, self.hps.version,
                export_path, self.hps.sample_rate, self.hps.if_f0,
                on_done=_on_export,
            )
        logger.info("checkpoint queued: epoch=%s step=%s emergency=%s -> %s",
                    epoch, self.global_step, emergency, g_path)
        return state

    def _finalize(self, net_g, epoch: int, completed: bool) -> None:
        # the loop already queued a full checkpoint at stop/completion;
        # here we only export the final inference model when completed.
        if completed:
            export_path = self.exports_dir / f"{self.config.experiment_name}.pth"

            def _on_export(export: Path) -> None:
                self.emit("training.final_model", path=str(export))

            self._io.submit(
                "final-export",
                engine_ckpt.export_inference_checkpoint,
                self._export_snapshot(net_g), self.hps, epoch, self.hps.version,
                export_path, self.hps.sample_rate, self.hps.if_f0,
                on_done=_on_export,
            )

    # ------------------------------------------------------------------ control
    def request_stop(self) -> None:
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()


def build_generator(hps, device: str) -> torch.nn.Module:
    """Standalone generator builder (the trainer's G path) for tooling such as
    ``rvc-platform train evaluate``. In eval mode the SynthesizerTrn forward
    runs its inference path (prior-only, no ``enc_q``), so both full training
    checkpoints and portable inference exports can be evaluated."""
    from rvc_platform.engine.rvc.models import models as rvc_models

    if int(hps.if_f0) == 1:
        cls = (rvc_models.SynthesizerTrnMs768NSFsid if hps.version == "v2"
               else rvc_models.SynthesizerTrnMs256NSFsid)
        return cls(
            hps.data["filter_length"] // 2 + 1,
            hps.train["segment_size"] // hps.data["hop_length"],
            **hps.model, is_half=False, sr=hps.sample_rate,
        ).to(device)
    cls = (rvc_models.SynthesizerTrnMs768NSFsid_nono if hps.version == "v2"
           else rvc_models.SynthesizerTrnMs256NSFsid_nono)
    return cls(
        hps.data["filter_length"] // 2 + 1,
        hps.train["segment_size"] // hps.data["hop_length"],
        **hps.model, is_half=False,
    ).to(device)
