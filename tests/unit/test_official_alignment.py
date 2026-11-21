"""Regression tests for v1.4.0 official-source alignment fixes.

Each test pins one defect found by auditing our code against OFFICIAL
primary sources (RVC repo/docs, BigVGAN paper/repo, speechbrain model
cards, huggingface_hub docs):

1. ECAPA model id: "spkrec-ecapa-voxcelebx" does NOT exist on the Hub
   (verified: HTTP 401). The correct id is "spkrec-ecapa-voxceleb" —
   with the wrong id, speaker-similarity silently disabled itself EVEN
   WITH speechbrain installed.
2. index_rate default: official RVC WebUI default is 0.75 (retrieval
   blending toward the training-set timbre, FAQ Q11); our 0.0 default
   disabled the mechanism entirely — a direct cause of "the generated
   voice sounds far from the target speaker".
3. d_lr_scale default: every official vocoder/VC trainer (HiFi-GAN,
   VITS, BigVGAN, so-vits-svc, RVC) uses identical G/D LR. Default is
   now 1.0 (upstream parity).
4. clip_grad_norm lever: BigVGAN §3.4 documents global-norm clipping
   (10^3; v2 config 500) as preventing early generator collapse — now
   wired through run_config -> hps -> trainer.
5. HF token warning: the official docs recommend fine-grained tokens
   for production; the warning must point users to the right settings
   URL and token choice.
"""
from __future__ import annotations

import math

import pytest


# --------------------------------------------------------------------------
# 1) ECAPA model id exists on the Hub (verified against the HF API)
# --------------------------------------------------------------------------
class TestEcapaModelId:
    def test_model_id_is_the_official_one(self):
        from rvc_platform.training import eval_metrics

        assert eval_metrics._SPEAKER_MODEL_HF == \
            "speechbrain/spkrec-ecapa-voxceleb"

    def test_wrong_legacy_id_is_gone(self):
        from rvc_platform.training import eval_metrics

        # the old id never existed on the Hub (401) — must not resurface
        assert "voxcelebx" not in eval_metrics._SPEAKER_MODEL_HF

    def test_resample_16k_passthrough_and_conversion(self):
        """ECAPA requires 16 kHz input (model card) — our wrapper resamples."""
        import torch

        from rvc_platform.training.eval_metrics import resample_16k

        wav = torch.zeros(1, 16000)  # 1 second of silence at 16 kHz
        out = resample_16k(wav, 16_000)
        assert out.shape[-1] == 16000
        # 40 kHz training rate -> exactly 16 kHz output
        wav40 = torch.zeros(1, 40000)
        out40 = resample_16k(wav40, 40_000)
        assert out40.shape[-1] == 16000


# --------------------------------------------------------------------------
# 2) index_rate default = official 0.75
# --------------------------------------------------------------------------
class TestIndexRateDefault:
    def test_conversion_settings_default_is_official(self):
        from rvc_platform.core.entities import ConversionSettings

        s = ConversionSettings()
        assert s.index_rate == 0.75, (
            "official RVC default is 0.75 (webui.py); 0.0 disables the "
            "retrieval-blend that pulls the timbre toward the training set"
        )
        s.validate()  # 0.75 must also pass validation

    def test_config_schema_default_matches(self):
        from rvc_platform.infrastructure.config.schema import ConversionDefaults

        assert ConversionDefaults().index_rate == 0.75

    def test_build_settings_uses_configured_default(self):
        """The service-level default flows from config, not a hard-coded 0."""
        from rvc_platform.application.conversion_service import ConversionService
        from rvc_platform.infrastructure.config.schema import ConversionDefaults


        class _Stub:  # minimal engine/audio stand-in: build_settings only
            # touches self.defaults / engine_cfg
            pass

        svc = ConversionService.__new__(ConversionService)
        svc.defaults = ConversionDefaults()
        svc.engine_cfg = None
        settings = svc.build_settings({})
        assert settings.index_rate == 0.75


# --------------------------------------------------------------------------
# 3) d_lr_scale default = 1.0 (upstream parity)
# --------------------------------------------------------------------------
class TestDLrScaleDefault:
    def test_default_is_upstream_parity(self):
        from rvc_platform.training.run_config import TrainingRunConfig

        cfg = TrainingRunConfig(experiment_name="x", dataset_path="/tmp/x",
                                sample_rate="40k")
        assert cfg.d_lr_scale == 1.0

    def test_lands_in_hparams(self):
        from rvc_platform.training.run_config import TrainingRunConfig

        cfg = TrainingRunConfig(experiment_name="x", dataset_path="/tmp/x",
                                sample_rate="40k")
        hps = cfg.to_hparams()
        assert hps.train["d_lr_scale"] == 1.0

    def test_clip_grad_norm_lever_round_trips(self):
        from rvc_platform.training.run_config import TrainingRunConfig

        cfg = TrainingRunConfig(experiment_name="x", dataset_path="/tmp/x",
                                sample_rate="40k", clip_grad_norm=500.0)
        assert cfg.to_hparams().train["clip_grad_norm"] == 500.0
        # default off = exact upstream behaviour
        cfg0 = TrainingRunConfig(experiment_name="x", dataset_path="/tmp/x",
                                 sample_rate="40k")
        assert cfg0.to_hparams().train["clip_grad_norm"] == 0.0


# --------------------------------------------------------------------------
# 4) G-side gradient clipping honours the BigVGAN lever
# --------------------------------------------------------------------------
class TestGradClippingWiring:
    def test_trainer_reads_clip_from_hps(self):
        """The clipping branch keys off hps.train['clip_grad_norm']."""
        import inspect

        from rvc_platform.training import trainer as trainer_mod

        src = inspect.getsource(trainer_mod.Trainer)
        assert "clip_grad_norm" in src, "trainer must consume the lever"
        assert "clip_grad_norm_" in src, (
            "must call torch.nn.utils.clip_grad_norm_ (BigVGAN §3.4)"
        )

    def test_clip_math_is_global_norm(self):
        """Sanity: clipping to a norm above current grads is a no-op,
        below current norm rescales — the BigVGAN semantics."""
        import torch

        lin = torch.nn.Linear(10, 10, bias=False)
        torch.nn.init.ones_(lin.weight)
        (lin(torch.ones(1, 10)).sum() * 100).backward()
        # grad norm = 100 * sqrt(100) = 1000
        norm_before = torch.norm(
            torch.stack([p.grad.norm() for p in lin.parameters()]))
        torch.nn.utils.clip_grad_norm_(lin.parameters(), 10000.0)
        norm_after = torch.norm(
            torch.stack([p.grad.norm() for p in lin.parameters()]))
        assert math.isclose(norm_before.item(), norm_after.item())
        torch.nn.utils.clip_grad_norm_(lin.parameters(), 100.0)
        norm_clipped = torch.norm(
            torch.stack([p.grad.norm() for p in lin.parameters()]))
        assert norm_clipped.item() <= 100.0 + 1e-3


# --------------------------------------------------------------------------
# 5) HF token warning points to the official settings page + fine-grained
# --------------------------------------------------------------------------
class TestHfTokenWarningCopy:
    def test_warning_mentions_fine_grained_and_settings_url(self, tmp_path):
        import logging

        from rvc_platform.training.huggingface import HfBackgroundPublisher

        pub = HfBackgroundPublisher("x/y")
        records: list[logging.LogRecord] = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Handler()
        logger = logging.getLogger("rvc_platform.training.huggingface")
        logger.addHandler(handler)
        try:
            ok = pub._ensure_api()  # no token in env -> must fail loudly ONCE
        finally:
            logger.removeHandler(handler)

        assert ok is False
        assert pub._token_warned is True
        text = records[-1].getMessage() if records else ""
        assert "huggingface.co/settings/tokens" in text
        assert "fine-grained" in text
        # second call must NOT re-warn (once, loudly)
        records.clear()
        pub._ensure_api()
        assert not records, "token warning must fire exactly once"
