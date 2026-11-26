"""Automatic optimization diagnostics — root cause analysis (Phase 02 W15).

When results disappoint, "try more trials" is often the wrong advice. This
module separates FIVE problem classes and diagnoses each from measured
evidence:

1. **inference parameters** — the settings the search controls;
2. **reference data** — the profile samples defining "the target voice";
3. **input audio** — the utterance being converted;
4. **model training quality** — limits no inference setting can fix
   (checked via honest proxies: index availability/size, round-trip
   ceiling when available, version compatibility, output-vs-reference
   mismatch patterns);
5. **evaluation models** — speaker embedders / analyzer limitations.

Every diagnosis carries: the problem, the EVIDENCE (the measured numbers
behind it), likely causes ranked with likelihood, a confidence value
(never "confirmed" from correlation alone), a technical recommendation
and — when one exists — a concrete action the ENGINE can try (e.g.
"fine-search index_rate in 0.7-0.9; trials there scored +0.06 composite").

Hard limits enforced here:

* the module never modifies model weights, never triggers training, never
  deletes or rewrites PTH/INDEX files;
* "model quality" conclusions are clearly marked PROXY-BASED (statistical
  association), never asserted as verified training defects;
* every recommendation names its evidence; no hand-waving.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["root_cause_analysis", "model_quality_report"]

_MAX_INPUT_MB = 200


def _round(x: Any, nd: int = 3) -> Any:
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


def _num(x: Any) -> float | None:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# model quality (proxy-based, honest)
# --------------------------------------------------------------------------- #
def model_quality_report(
    model_info: dict[str, Any] | None,
    profile: dict[str, Any] | None,
    best_eval: dict[str, Any] | None,
    roundtrip: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Proxy-based model training-quality checks.

    What CAN be measured from the outside (honest proxies only):

    * INDEX file present + its size (retrieval capacity proxy);
    * model file size (capacity proxy);
    * round-trip ceiling: converting a *reference sample of the target
      speaker's own voice* through the model — if even that scores low on
      speaker similarity, no inference parameter can reach the target
      (the model itself is the limiter);
    * version/compat flags the engine exposes;
    * persistent output-vs-reference formant/F0 mismatch.

    What CANNOT be measured here is explicitly listed as unavailable
    (training set noise level, training diversity, actual training
    settings) — the report never pretends otherwise.
    """
    info = model_info or {}
    profile_dict = profile or {}
    checks: list[dict[str, Any]] = []
    limits: list[str] = []

    index_path = info.get("index_path")
    checks.append({
        "check": "index_file",
        "value": (str(Path(str(index_path)).name) if index_path
                  else None),
        "status": ("ok" if index_path else
                   "warn: no index file — index_rate does nothing; "
                   "timbre matching relies on the generator alone"),
    })
    if index_path:
        try:
            size_mb = Path(str(index_path)).stat().st_size / (1024 * 1024)
            checks.append({
                "check": "index_size_mb",
                "value": round(size_mb, 2),
                "status": ("ok" if size_mb >= 5 else
                           "warn: small index — limited retrieval coverage "
                           "(short/small training set proxy)"),
            })
        except OSError:
            pass
    try:
        model_path = Path(str(info.get("path", "")))
        if model_path.is_file():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            checks.append({
                "check": "model_size_mb",
                "value": round(size_mb, 2),
                "status": ("ok" if size_mb >= 20 else
                           "warn: unusually small model file"),
            })
    except OSError:
        pass
    # round-trip ceiling — the strongest available model-quality evidence
    if roundtrip is not None:
        sim = _num(roundtrip.get("speaker_similarity"))
        if sim is not None:
            checks.append({
                "check": "roundtrip_speaker_similarity",
                "value": round(sim, 4),
                "status": ("ok" if sim >= 0.6 else
                           f"LIMIT: converting the target speaker's OWN "
                           f"reference audio through this model scores "
                           f"{sim:.2f} on speaker similarity — the model "
                           "itself caps how close any output can get "
                           "(proxy-based conclusion)"),
            })
    else:
        limits.append("round-trip probe not run (optional; add it to "
                      "measure the model's own ceiling)")
    # persistent formant mismatch: output formants consistently far from
    # the reference profile across trials -> possible model/training mismatch
    if best_eval is not None:
        errs = _num((best_eval.get("detail") or {}).get(
            "formant_relative_errors_mean"))
        if errs is not None and errs > 0.25:
            checks.append({
                "check": "formant_mismatch",
                "value": round(errs, 3),
                "status": "warn: output formants sit far from the "
                          "reference profile across trials — consistent "
                          "with a model trained on different "
                          "acoustics/phone (proxy-based)",
            })
    limits.append("training-set noise level, diversity and training "
                 "settings are NOT observable from inference — no claims "
                 "are made about them")
    return {
        "checks": checks,
        "unavailable": limits,
        "disclaimer": "model-quality conclusions are proxy-based "
                      "inference from measurable evidence, not verified "
                      "training diagnostics",
    }


# --------------------------------------------------------------------------- #
# root-cause analysis over the whole session
# --------------------------------------------------------------------------- #
def root_cause_analysis(
    *,
    trials: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    best: dict[str, Any] | None,
    validation: list[dict[str, Any]],
    profile_reliability: dict[str, Any],
    input_analyses: list[dict[str, Any]],
    model_info: dict[str, Any] | None,
    roundtrip: dict[str, Any] | None = None,
    insights: dict[str, Any] | None = None,
    artifact_recurrence: dict[str, Any] | None = None,
    embedder_status: dict[str, Any] | None = None,
    robustness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evidence-based diagnosis, separated by problem class."""
    diagnoses: list[dict[str, Any]] = []

    ok_trials = [t for t in (trials or [])
                 if not t.get("error") and t.get("composite") is not None]

    # ---------- class 1: inference parameters ----------
    if best is not None and baseline is not None:
        best_c = _num(best.get("composite"))
        base_c = _num((baseline.get("evaluation") or {}).get("composite"))
        if best_c is not None and base_c is not None:
            delta = best_c - base_c
            if delta < 0.02:
                diagnoses.append({
                    "problem": "search found almost no improvement over the "
                               "baseline settings",
                    "class": "inference_parameters",
                    "evidence": [f"baseline composite {_round(base_c, 4)}",
                                 f"best composite {_round(best_c, 4)} "
                                 f"(delta {delta:+.4f})",
                                 f"{len(ok_trials)} completed trials"],
                    "likely_causes": [
                        {"cause": "the engine's parameter space genuinely "
                                  "cannot move this input/model pair",
                         "likelihood": "high"},
                        {"cause": "input or model limitations dominate the "
                                  "score (see the other classes)",
                         "likelihood": "medium"},
                    ],
                    "confidence": 0.6,
                    "recommendation": "check the reference-data and "
                                     "model-quality classes before spending "
                                     "more trials; more search will not fix "
                                     "a data problem",
                    "engine_action": None,
                })
            else:
                # recommend refinement around the best bin of the most
                # influential parameter
                action = None
                if insights and insights.get("status") == "ok":
                    recs = insights.get("recommendations") or []
                    for r in recs:
                        if "highest composite observed in" in str(r):
                            action = ("fine-search around the observed best "
                                      "region: " + str(r).split(";")[0])
                            break
                diagnoses.append({
                    "problem": None,      # informational entry
                    "class": "inference_parameters",
                    "summary": f"search improved composite by "
                               f"{delta:+.4f} over baseline",
                    "evidence": [f"baseline {_round(base_c, 4)} -> best "
                                 f"{_round(best_c, 4)} over "
                                 f"{len(ok_trials)} trials"],
                    "confidence": 0.9,
                    "recommendation": action or "keep the found settings; "
                                     "optionally raise the trial budget "
                                     "for fine-tuning",
                    "engine_action": action,
                })
    # validation overfit signal
    if validation:
        drops = [_num(v.get("drop")) for v in validation
                 if _num(v.get("drop")) is not None]
        drops = [d for d in drops if d is not None]
        if drops and max(drops) > 0.15:
            diagnoses.append({
                "problem": "top candidates dropped sharply on held-out "
                           "audio — scores were partly overfitted to the "
                           "search audio",
                "class": "inference_parameters",
                "evidence": [f"max validation drop "
                             f"{max(drops) * 100:.0f}% across "
                             f"{len(drops)} validated candidates"],
                "likely_causes": [
                    {"cause": "search audio too short/one-sided for the "
                              "parameter landscape",
                     "likelihood": "high"},
                ],
                "confidence": 0.7,
                "recommendation": "trust the validated numbers; add a "
                                 "second input take so the search sees "
                                 "more variety",
                "engine_action": None,
            })
    # artifact-driven parameter advice
    if artifact_recurrence:
        links = artifact_recurrence.get("parameter_links") or {}
        for artifact, ls in list(links.items())[:3]:
            for link in ls[:1]:
                diagnoses.append({
                    "problem": f"artifact '{artifact}' recurs across trials",
                    "class": "inference_parameters",
                    "evidence": [f"{artifact} severity correlates with "
                                 f"{link['parameter']} "
                                 f"(rho={link['spearman_rho']}, "
                                 f"n={link['n']})"],
                    "likely_causes": [
                        {"cause": f"{link['parameter']} interacts with this "
                                  f"artifact",
                         "likelihood": "medium (statistical association, "
                                       "not causation)"},
                    ],
                    "confidence": 0.5,
                    "recommendation": f"when re-tuning, watch {link['parameter']}"
                                     f" against {artifact} severity",
                    "engine_action": f"bias the search away from "
                                     f"{link['direction']} of "
                                     f"{link['parameter']}",
                })
    # index missing -> index_rate is a dead knob
    if model_info is not None and not model_info.get("index_path"):
        diagnoses.append({
            "problem": "index_rate cannot affect the conversion at all",
            "class": "inference_parameters",
            "evidence": ["model has no .index file registered"],
            "likely_causes": [{"cause": "index was never built/shared for "
                                        "this model",
                               "likelihood": "certain"}],
            "confidence": 1.0,
            "recommendation": "build or attach a feature index for this "
                             "model to unlock timbre retrieval; until then "
                             "ignore index_rate results",
            "engine_action": "exclude index_rate from the search space",
        })

    # ---------- class 2: reference data ----------
    rel = profile_reliability or {}
    if rel.get("level") in ("low", "warn") or (rel.get("notes") or []):
        ev: list[str] = []
        ev += [f"reliability level: {rel.get('level')}",
               f"samples: {rel.get('n_samples')}, speech "
               f"{rel.get('total_duration_s')}s"]
        if rel.get("self_consistency"):
            worst = min((v for v in rel["self_consistency"].values()
                         if v is not None), default=None)
            if worst is not None:
                ev.append(f"worst intra-profile embedding similarity "
                          f"{_round(worst)}")
        causes = []
        if (rel.get("n_samples") or 0) < 2:
            causes.append({"cause": "too few reference samples",
                           "likelihood": "high"})
        if (rel.get("total_duration_s") or 0) < 6:
            causes.append({"cause": "not enough reference speech",
                           "likelihood": "high"})
        if rel.get("outliers"):
            causes.append({"cause": "an outlier sample (different speaker "
                                    "or corrupted recording)",
                           "likelihood": "medium"})
        sq = rel.get("sample_quality") or []
        poor = [q for q in sq if q.get("grade") == "poor"]
        if poor:
            causes.append({"cause": f"{len(poor)} low-quality reference "
                                    f"sample(s)",
                           "likelihood": "high"})
        diagnoses.append({
            "problem": "reference profile is a weak identity anchor",
            "class": "reference_data",
            "evidence": ev,
            "likely_causes": causes or
                             [{"cause": "see notes",
                               "likelihood": "unknown"}],
            "confidence": 0.75,
            "recommendation": "add 2-5 clean, varied samples of the target "
                             "speaker (>= 3 s each, >= 10 s total; no "
                             "music/background voices) and rebuild the "
                             "profile",
            "engine_action": None,
        })
    # calibration availability
    emb_counts = (rel.get("embedding_counts") or {})
    if any((v or 0) < 2 for v in emb_counts.values()):
        diagnoses.append({
            "problem": "speaker-similarity calibration unavailable",
            "class": "reference_data",
            "evidence": [f"embedding counts per backend: {emb_counts} "
                         "(calibration needs >= 2 samples per backend)"],
            "likely_causes": [{"cause": "single reference sample per "
                                        "backend",
                               "likelihood": "certain"}],
            "confidence": 0.9,
            "recommendation": "add reference samples so each embedder has "
                             ">= 2; cross-model scores are then calibrated "
                             "before ensembling",
            "engine_action": None,
        })

    # ---------- class 3: input audio ----------
    for ia in (input_analyses or [])[:4]:
        loud = ia.get("loudness") or {}
        warn_items: list[str] = []
        ev = [f"input '{ia.get('source', '?')}': "
              f"{ia.get('duration_s')}s"]
        snr = _num(loud.get("snr_proxy_db"))
        clip = _num(loud.get("clipping_ratio"))
        voiced = _num((ia.get("vad") or {}).get("voiced_ratio"))
        if snr is not None:
            ev.append(f"SNR proxy {snr:.1f} dB")
            if snr < 12:
                warn_items.append("noisy input")
        if clip is not None and clip > 0.001:
            warn_items.append("clipped input")
            ev.append(f"clipping ratio {clip:.3%}")
        if voiced is not None:
            ev.append(f"voiced ratio {voiced:.0%}")
            if voiced < 0.25:
                warn_items.append("barely any speech content")
        dur = _num(ia.get("duration_s"))
        if dur is not None and dur < 3.0:
            warn_items.append("input under 3 s — scores unstable")
            ev.append(f"duration {dur:.1f}s")
        if warn_items:
            diagnoses.append({
                "problem": "input audio quality limits conversion scoring: "
                           + ", ".join(warn_items),
                "class": "input_audio",
                "evidence": ev,
                "likely_causes": [{"cause": w, "likelihood": "high"}
                                  for w in warn_items],
                "confidence": 0.8,
                "recommendation": "supply a clean 5-15 s input take "
                                 "(low noise, no clipping, continuous "
                                 "speech) and re-run the optimization",
                "engine_action": None,
            })

    # ---------- class 4: model quality ----------
    mq = model_quality_report(model_info, None,
                              (best or {}).get("detail")
                              if isinstance(best, dict) else None,
                              roundtrip)
    limit_checks = [c for c in mq["checks"]
                    if str(c.get("status", "")).startswith("LIMIT")]
    for c in limit_checks:
        diagnoses.append({
            "problem": "the model itself caps achievable similarity",
            "class": "model_quality",
            "evidence": [f"{c['check']}: {c['status']}"],
            "likely_causes": [
                {"cause": "training data/speaker mismatch or insufficient "
                          "training (proxy-based)",
                 "likelihood": "medium"},
            ],
            "confidence": 0.6,
            "recommendation": "no inference setting can cross this "
                             "ceiling; consider improving the training "
                             "data (more/cleaner/more-varied samples of "
                             "the target speaker) and re-training — a "
                             "separate decision, never automatic",
            "engine_action": None,
        })

    # ---------- class 5: evaluation models ----------
    es = embedder_status or {}
    if es.get("reliability") == "low" or not (
            es.get("wavlm") or es.get("ecapa")):
        diagnoses.append({
            "problem": "no neural speaker-verification model available — "
                       "identity scoring used the DSP voiceprint fallback",
            "class": "evaluation_models",
            "evidence": [f"wavlm={bool(es.get('wavlm'))}, "
                         f"ecapa={bool(es.get('ecapa'))}",
                         str((es.get("messages") or {}))[:300]],
            "likely_causes": [
                {"cause": "model download unavailable at startup "
                          "(offline / firewall)", "likelihood": "high"},
            ],
            "confidence": 0.9,
            "recommendation": "install/allow the quality extra "
                             "(speechbrain) and network access to "
                             "huggingface, then restart — identity scores "
                             "will rise from LOW to MEDIUM/HIGH "
                             "reliability",
            "engine_action": None,
        })

    # ---------- robustness ----------
    if robustness and robustness.get("std") is not None:
        mean = _num(robustness.get("mean"))
        std = _num(robustness.get("std"))
        if mean is not None and std is not None and std > 0.08:
            diagnoses.append({
                "problem": "best settings perform unevenly across input "
                           "types",
                "class": "inference_parameters",
                "evidence": [f"probe mean {mean:.3f} +/- {std:.3f}; "
                             f"worst type '{robustness.get('worst_type')}'",
                             "types: " + str(list((robustness.get("per_type")
                                                   or {}).keys()))],
                "likely_causes": [
                    {"cause": "settings tuned to one speaking style "
                              "(single-take search audio)",
                     "likelihood": "medium"},
                ],
                "confidence": 0.55,
                "recommendation": "add input variety (quiet + loud takes) "
                                 "and prefer the stability-aware winner",
                "engine_action": "rank by stability (mean - 0.5xstd) "
                                 "instead of the single best composite",
            })

    return {
        "n_diagnoses": len(diagnoses),
        "diagnoses": diagnoses,
        "problem_classes": {
            "inference_parameters": [d for d in diagnoses
                                     if d.get("class") ==
                                     "inference_parameters"],
            "reference_data": [d for d in diagnoses
                               if d.get("class") == "reference_data"],
            "input_audio": [d for d in diagnoses
                            if d.get("class") == "input_audio"],
            "model_quality": [d for d in diagnoses
                              if d.get("class") == "model_quality"],
            "evaluation_models": [d for d in diagnoses
                                  if d.get("class") == "evaluation_models"],
        },
        "model_quality": mq,
        "disclaimer": "diagnoses are evidence-ranked hypotheses with "
                      "confidence values — correlation-based entries are "
                      "never asserted as proven causes",
    }
