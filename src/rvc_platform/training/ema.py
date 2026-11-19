"""Exponential moving average (EMA) of the generator weights.

Research-backed quality lever for GAN vocoder training: the *shadow* copy of
the generator's parameters tracks a discounted average of every optimizer
step, and the averaged weights are the ones validated, exported and shipped.
References (see UPGRADE_REPORT_AR.md):
  * "The Unusual Effectiveness of Averaging in GAN Training" (Yazici et al.,
    DeepMind — ICLR blog track): EMA solutions beat last-iterate GAN weights.
  * BigVGAN (Lee et al., ICLR 2023) — the generator is validated/exported
    from its EMA weights; the same recipe is used across modern GAN vocoders.
  * "Exponential Moving Average of Weights in Deep Learning" (Polyak/Ruppert
    averaging, modern treatment) — EMA improves generalisation robustness.

Implementation notes:
  * shadow tensors live on the SAME device as the live parameters (update is
    two fused element-wise ops per parameter, cheap on GPU; on CPU it is the
    same math);
  * an optional warmup ramp ``min(decay, (1+t)/(10+t))`` keeps early epochs
    from being dominated by the random/pretrained init (standard diffusion-
    training trick, also DiffAug/cd-EMA);
  * the EMA state is checkpointed inside the G payload ("ema") so resume is
    bit-exact; legacy checkpoints without "ema" re-initialise from weights;
  * buffers (non-trainable state) are NOT averaged — they are copied live at
    export time, so the exported state dict is always complete.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class GeneratorEMA:
    """Maintains ``shadow = decay * shadow + (1 - decay) * param`` per step."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999,
                 warmup: bool = True) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"ema decay must be in (0,1), got {decay!r}")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.step = 0
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        logger.info("generator EMA enabled: decay=%s warmup=%s params=%d",
                    self.decay, self.warmup, len(self.shadow))

    # ------------------------------------------------------------------ update
    def _current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Fold the current parameters into the shadow (call after optimizer.step)."""
        decay = self._current_decay()
        params = dict(model.named_parameters())
        for name, param in params.items():
            shadow = self.shadow.get(name)
            if shadow is None:                      # new parameters appeared
                self.shadow[name] = param.detach().clone()
                continue
            if shadow.dtype != param.dtype or shadow.device != param.device:
                shadow = shadow.to(param.device, param.dtype)
                self.shadow[name] = shadow
            shadow.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        self.step += 1

    # ------------------------------------------------------------------ export
    @torch.no_grad()
    def export_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Full ``model.state_dict()``-compatible dict: EMA parameters +
        live buffers (any non-parameter entries are copied from the live model
        so the result always loads with ``strict=True``)."""
        live = model.state_dict()
        merged: dict[str, torch.Tensor] = {}
        for name, tensor in self.shadow.items():
            merged[name] = tensor.detach().clone()
        for key, value in live.items():
            if key not in merged:
                merged[key] = value.detach().clone() if torch.is_tensor(value) else value
        return merged

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        """Swap the EMA weights into ``model`` (buffers stay live)."""
        model.load_state_dict(self.export_state_dict(model))

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "step": self.step,
            "shadow": {
                name: tensor.detach().cpu() for name, tensor in self.shadow.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any], device: Any | None = None) -> None:
        shadow_in = state.get("shadow") or {}
        if not shadow_in:
            logger.warning("EMA state has no shadow tensors; re-initialising")
            return
        self.decay = float(state.get("decay", self.decay))
        self.warmup = bool(state.get("warmup", self.warmup))
        self.step = int(state.get("step", 0))
        self.shadow = {
            name: (tensor.to(device=device) if device is not None else tensor)
            for name, tensor in shadow_in.items()
        }
        logger.info("EMA state restored: decay=%s step=%s params=%d",
                    self.decay, self.step, len(self.shadow))
