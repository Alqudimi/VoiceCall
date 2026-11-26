"""ULTIMATE AI VOICE OPTIMIZER — intelligent RVC v2 parameter tuning.

A multi-stage optimization subsystem that:

1. builds a *reference speaker profile* from clean samples of the voice the
   model was trained on (speaker embeddings + F0 statistics + timbre
   statistics);
2. runs real RVC v2 conversions over a search space of the engine's actual
   parameters (f0 method, index rate, filter radius, RMS mix rate, protect,
   optional pitch policy);
3. scores every candidate output with a *multi-metric evaluation engine*
   (speaker identity similarity via independent speaker-verification models,
   timbre similarity, pitch consistency, naturalness, clarity, artifact
   detection, prosody preservation, audio quality);
4. drives the search with Bayesian-style optimization (Optuna TPE / NSGA-II
   when installed, with a built-in adaptive coarse-to-fine fallback);
5. validates the best candidates on held-out audio segments, then exposes
   honest results, A/B listening and one-click "apply best settings".

Design rules inherited from the platform:

* every heavy dependency is OPTIONAL and guarded — the optimizer degrades to
  DSP-only scoring (clearly flagged as low-reliability) instead of crashing;
* no metric is ever presented as an absolute "identity match percentage";
  all scores are documented relative measures within this search space;
* the engine's own loaded model is reused across trials (no reloads).
"""
from rvc_platform.optimizer.analysis import AudioAnalyzer, AnalysisResult
from rvc_platform.optimizer.embeddings import (
    SpeakerEmbedderManager,
    SpeakerEmbeddingStatus,
)
from rvc_platform.optimizer.profile import (
    ReferenceProfile,
    ReferenceProfileStore,
)
from rvc_platform.optimizer.scoring import (
    EvaluationEngine,
    MetricWeights,
    VoiceEvaluation,
)
from rvc_platform.optimizer.search import (
    BuiltInOptimizer,
    OptimizationSpace,
    OptunaOptimizer,
    TrialRecord,
    available_f0_methods,
    build_backend,
)
from rvc_platform.optimizer.service import OptimizerService

__all__ = [
    "AudioAnalyzer",
    "AnalysisResult",
    "SpeakerEmbedderManager",
    "SpeakerEmbeddingStatus",
    "ReferenceProfile",
    "ReferenceProfileStore",
    "EvaluationEngine",
    "MetricWeights",
    "VoiceEvaluation",
    "OptimizationSpace",
    "TrialRecord",
    "available_f0_methods",
    "build_backend",
    "OptunaOptimizer",
    "BuiltInOptimizer",
    "OptimizerService",
]
