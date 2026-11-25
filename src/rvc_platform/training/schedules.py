"""Warmup + cosine-annealing LR schedule for fine-tuning GAN vocoders.

Why (research): the upstream trainer applies AdamW lr=1e-4 with a pure
ExponentialLR decay (0.999875/step) from the very first step. For
fine-tuning from a pretrained base (the standard RVC path), the literature
and community practice favour:

  * a short warmup ramp — fine-tuning starts from pretrained weights; a
    full-size lr applied instantly can shock the generator/discriminator
    balance already learned by the base (fine-tuning guides recommend
    5-10% of total steps for warmup);
  * a cosine decay to near-zero at the end — the standard schedule for
    stable GAN fine-tuning (StyleGAN-3 uses a fixed then decaying schedule;
    BigVGAN fine-tune recipes rely on decayed tails to polish detail);
    the exponential tail upstream decays to ~87% of lr after 200 epochs
    (0.999875^steps), i.e. it never actually anneals — the final epochs run
    at nearly full lr, exactly when the model is most delicate and
    overtraining damage happens.

Implementation contract (Colab-safe, resume-exact):
  * the schedule composes with the existing ExponentialLR by *multiplying*
    step-wise factors only when ``warmup_epochs`` or ``cosine_final_lr_frac``
    are set; with the defaults (warmup_epochs=1 — one epoch of linear ramp
    to lr, cosine fraction 0 = disabled) the behaviour is the upstream
    exponential decay with a single-epoch linear ramp (a conservative,
    strictly-stabilising deviation);
  * resume-exact: it is rebuilt from (epoch, steps_per_epoch) at restore
    time and stepped to the recorded global_step — the same deterministic
    walk the checkpointed schedulers use;
  * pure-torch; no external dependency.
"""
from __future__ import annotations

import math

import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Factor schedule: warmup ramp -> 1.0 -> optional cosine tail (+ optional exp decay).

    When ``epoch_gamma < 1.0`` (e.g. 0.999875), the per-epoch exponential decay
    is composed analytically inside this single scheduler so two LRSchedulers
    never fight over ``optimizer.param_groups[0]['lr']``.
    """

    def __init__(self, optimizer, steps_per_epoch: int, warmup_epochs: int = 1,
                 total_epochs: int = 0, cosine_frac: float = 0.0,
                 epoch_gamma: float = 1.0, last_epoch: int = -1) -> None:
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.total_epochs = max(0, int(total_epochs))
        self.cosine_frac = float(cosine_frac)
        self.epoch_gamma = float(epoch_gamma)
        super().__init__(optimizer, last_epoch)

    def _factor(self, step: int) -> float:
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        if self.warmup_epochs > 0 and step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if self.cosine_frac > 0 and self.total_epochs > 0:
            total_steps = self.total_epochs * self.steps_per_epoch
            progress = min(1.0, max(0.0, (step - warmup_steps)
                                    / max(1, total_steps - warmup_steps)))
            # cosine from 1.0 down to cosine_frac
            return self.cosine_frac + (1.0 - self.cosine_frac) * 0.5 * (
                1.0 + math.cos(math.pi * progress))
        return 1.0

    def get_lr(self) -> list[float]:
        step = self.last_epoch  # stepped once per .step() call == per train step
        factor = self._factor(step)
        if 0.0 < self.epoch_gamma < 1.0:
            completed_epochs = max(0, step) // self.steps_per_epoch
            factor *= (self.epoch_gamma ** completed_epochs)
        return [base_lr * factor for base_lr in self.base_lrs]

    def sync_to_step(self, global_step: int) -> None:
        """Fast-forward scheduler state on checkpoint resume so warmup/cosine do not restart."""
        target = max(0, int(global_step))
        self.last_epoch = target
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group["lr"] = lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
