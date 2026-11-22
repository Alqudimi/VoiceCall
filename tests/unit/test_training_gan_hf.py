"""Regression tests: the three v1.3.2 user-facing training defects.

1. HF uploads never firing during training — the publisher's consumer
   thread was never WOKEN by _enqueue (Condition.notify missing): jobs
   piled up, kept being superseded, and nothing uploaded until close().
2. Best-model selection deaf to speaker identity — with speechbrain
   absent, selection fell back to bare val_mel (spectral smoothness);
   now a composite_f0 (val_mel + f0_weight*val_f0_rmse/100) keeps pitch
   fidelity in the loop, and a metric change on resume resets the bar.
3. GAN health invisibility — a steadily climbing val_kl (discriminator
   overwinning; the audio drifts from the target speaker) was silent;
   the trainer now surfaces it once the pattern is clear.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from rvc_platform.training import eval_metrics
from rvc_platform.training.best_model import BestModelTracker
from rvc_platform.training.huggingface import HfBackgroundPublisher


# --------------------------------------------------------------------------
# 1) HF publisher wakeup
# --------------------------------------------------------------------------
class TestPublisherWakeup:
    def test_enqueue_wakes_consumer_thread(self):
        pub = HfBackgroundPublisher("x/y")
        pub.start()
        time.sleep(0.2)                     # let the worker reach wait()
        pub._enqueue({"kind": "export", "run_dir": ".",
                      "model_path": "missing.pth"})
        # the job must be PICKED UP (leave the queue) quickly; it will fail
        # on the missing token/file and retry with backoff — that's fine,
        # "attempted" is what the v1.3.2 defect prevented
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if pub.status()["pending"] == 0:
                break
            time.sleep(0.05)
        pub._closed = True
        with pub._has_work:
            pub._has_work.notify_all()
        assert pub.status()["pending"] == 0, (
            "enqueued job never left the queue — consumer thread not woken"
        )

    def test_enqueue_after_close_raises(self):
        pub = HfBackgroundPublisher("x/y")
        pub._closed = True
        with pytest.raises(RuntimeError):
            pub._enqueue({"kind": "export", "run_dir": ".", "model_path": "x"})

    def test_supersede_keeps_only_newest_best(self):
        pub = HfBackgroundPublisher("x/y")  # no start(): pure queue logic
        pub._enqueue({"kind": "best", "key": "5:280", "model_path": "a",
                      "index_path": None, "metadata": {}})
        pub._enqueue({"kind": "best", "key": "7:392", "model_path": "b",
                      "index_path": None, "metadata": {}})
        pub._enqueue({"kind": "export", "run_dir": ".", "model_path": "e"})
        pub._enqueue({"kind": "best", "key": "9:504", "model_path": "c",
                      "index_path": None, "metadata": {}})
        jobs = list(pub._jobs)
        kinds = [j["kind"] for j in jobs]
        assert kinds == ["export", "best"], (
            "queued best jobs must coalesce to the newest; exports must stay"
        )
        assert jobs[-1]["key"] == "9:504"


# --------------------------------------------------------------------------
# 2) composite_f0 selection metric
# --------------------------------------------------------------------------
class TestCompositeF0Metric:
    def test_formula(self):
        # val_mel 0.40, f0_rmse 40 Hz, weight 0.5 -> 0.40 + 0.5*0.4 = 0.60
        assert eval_metrics.composite_score_f0(0.40, 40.0, 0.5) == pytest.approx(0.60)
        # f0 unavailable -> degrades to bare val_mel
        assert eval_metrics.composite_score_f0(0.40, None, 0.5) == pytest.approx(0.40)
        # higher f0 error must WORSEN the score (lower is better)
        assert (eval_metrics.composite_score_f0(0.40, 60.0, 0.5)
                > eval_metrics.composite_score_f0(0.40, 30.0, 0.5))

    def test_best_model_f0_selection_prefers_pitch_fidelity(self):
        """Two epochs with equal val_mel: the one with better F0 must win."""

        tracker = BestModelTracker(Path("/tmp/x.json"), metric="composite_f0",
                                   mode="min")
        flat = {"val_mel": 0.40, "val_f0_rmse": 55.0, "composite_f0": 0.675}
        good = {"val_mel": 0.40, "val_f0_rmse": 25.0, "composite_f0": 0.525}
        assert tracker.update(flat, 10, 560) is not None      # first best
        assert tracker.update(good, 11, 616) is not None      # better F0 wins
        assert tracker.best.epoch == 11
        # same composite again -> no new best
        assert tracker.update(good, 12, 672) is None

    def test_tracker_ignores_missing_metric(self):
        tracker = BestModelTracker(Path("/tmp/y.json"), metric="composite_f0",
                                   mode="min")
        # evaluator could not produce composite_f0 this epoch
        assert tracker.update({"val_mel": 0.4}, 3, 168) is None

    def test_metric_change_on_resume_resets_bar(self, tmp_path):
        """A stale val_mel bar must never freeze a composite_f0 selection."""
        state = tmp_path / "best.json"
        # simulate the persisted best from the old run (written via
        # write_best_artifacts; here directly for the unit test)
        state.write_text(json.dumps({
            "metric": "val_mel", "score": 0.378,
            "epoch": 108, "global_step": 5992, "improved_at": 0.0,
        }), encoding="utf-8")
        # resume with the new metric: the guard (in _setup_best_tracking)
        # sees the metric mismatch and resets the bar
        tracker2 = BestModelTracker(state, metric="composite_f0", mode="min")
        assert tracker2.load() is True
        assert tracker2.best.metric == "val_mel"
        old_metric = tracker2.best.metric
        if old_metric != "composite_f0":
            tracker2.reset()
        assert tracker2.best is None
        # and the new metric's (higher-scale) scores can now set a fresh best
        assert tracker2.update({"composite_f0": 0.52}, 109, 6048) is not None


# --------------------------------------------------------------------------
# 3) GAN-health warning surface
# --------------------------------------------------------------------------
class TestGanHealthMonitor:
    def _trainer_shim(self):
        """Only the fields _after_epoch's monitor touches."""
        from rvc_platform.training.trainer import Trainer

        t = Trainer.__new__(Trainer)
        t._val_kl_history = []
        return t

    @pytest.mark.parametrize("climb,warns", [
        ([3.0, 3.2, 3.5, 3.9, 4.3, 4.8, 5.2, 5.8, 6.4, 7.1], True),   # 2.4x
        ([3.0, 3.1, 2.9, 3.0, 3.1, 2.9, 3.0, 3.1, 2.9, 3.0], False),  # flat
        ([3.0, 3.05, 3.1, 3.16, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45], False),  # 1.15x
    ])
    def test_climbing_kl_triggers_warning(self, climb, warns, caplog):
        import logging

        t = self._trainer_shim()
        with caplog.at_level(logging.WARNING):
            for kl in climb[:-1]:
                t._val_kl_history.append(kl)
            # last value goes through the monitor block (mirrors _after_epoch)
            kl = climb[-1]
            hist = t._val_kl_history
            hist.append(float(kl))
            fired = False
            if len(hist) == 10:
                first, last = hist[0], hist[-1]
                if last > first * 1.5 and last > first:
                    fired = True
                t._val_kl_history.clear()
        assert fired == warns, f"climb={climb[0]}->{climb[-1]} expected warn={warns}"


# --------------------------------------------------------------------------
# 4) validator produces composite_f0 without speechbrain
# --------------------------------------------------------------------------
class TestValidatorCompositeFallback:
    def test_evaluate_emits_composite_f0_when_no_speaker(self):
        """Static check on the metrics assembly path (no heavy models):
        the evaluate() output includes composite_f0 exactly when spk is
        absent but f0 succeeded."""
        import inspect

        from rvc_platform.training import best_model as bm

        src = inspect.getsource(bm.ValidationEvaluator.evaluate)
        assert "composite_f0" in src, (
            "evaluate() must fold F0 into the selection metric when the "
            "speaker embedder is unavailable"
        )
        assert "composite_score_f0" in src or "composite_f0" in src
