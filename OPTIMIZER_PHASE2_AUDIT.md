# PHASE 02 — Deep Audit of the Phase 01 Optimizer (internal report)

Scope: `src/rvc_platform/optimizer/` (7 modules, ~3.5k lines), the API routes,
the web panel and the test suite, audited against the Phase 02 requirements.
All 37 Phase 01 unit tests and 16 integration tests pass at audit time
(`PYTHONPATH=src pytest tests/unit/test_optimizer_*.py tests/integration/test_optimizer_*.py`).

## A. What Phase 01 does correctly (kept, not rebuilt)

1. Real search loop over **only engine-supported parameters**
   (`f0_method`, `index_rate`, `filter_radius` {0,3} with honest kernel-3
   semantics, `rms_mix_rate`, `protect`, optional `pitch_shift` policy).
2. Per-model reference profiles with filesystem isolation; multi-profile.
3. Two independent neural speaker embedders (WavLM x-vector + ECAPA via
   SpeechBrain), reported **per backend** with agreement spread.
4. Weighted multi-axis scoring with hard quality gates; Pareto front;
   validation on held-out audio; relative-score honesty disclaimers.
5. Crash-safe session persistence, resume, pause, cancel; audio pruning.
6. Additive API + UI (Arabic RTL panel) — original endpoints untouched.

## B. Weaknesses found (each drives a Phase 02 fix)

| # | Finding | Severity | Fix |
|---|---------|----------|-----|
| W1 | Cosines from different SV models averaged into one ensemble directly — no per-backend calibration despite different score distributions | high | `calibration.py`: per-backend cohort z-margin built from intra-speaker similarity distribution; ensemble only after calibration |
| W2 | Embeddings computed over **full utterance including silence**; single embedding per file, no stability estimate | high | analysis v2: speech-only (VAD) embedding + windowed embedding stability |
| W3 | `embeddings.py` docstring promises a "DSP fallback voiceprint, always available" — **it does not exist** (functional claim with no implementation) | high | implement a real MFCC-statistics DSP voiceprint fallback (make the claim true) |
| W4 | Reference samples folded with **equal weight**, no per-sample quality assessment, no outlier exclusion, no unified centroid | high | profile v2: per-sample quality reports (SNR/clipping/silence/noise), quality-weighted centroid + outlier flagging, individual embeddings kept |
| W5 | `scoring._prosody` docstring claims "loudness envelope correlation" — **not implemented** (VAD ratio/pause count/coverage/F0-histogram only) | high | implement true frame-envelope correlation + time-warped F0 contour correlation + speech-rate proxy |
| W6 | Artifact detection: only 6 coarse proxies, no **location**, no recurrence tracking, no parameter linkage | high | `quality.py`: 15 detectors with severity + time locations + cross-trial recurrence + parameter correlation |
| W7 | **No parameter relationship discovery at all** (no sensitivity/importance/interactions/instability regions) | high | `interactions.py`: binned marginal effects, sensitivity ranges, pairwise interaction terms, importance ranking, unstable-region detection — all derived from real trial data |
| W8 | `_phase_for` labels are cosmetic: BuiltIn search behavior does **not** change across phases (fixed probabilities, not staged budgets); quick-mode "phases" don't exist in behavior | high | search v2: explicit staged budgets (A explore → B TPE-guided → C fine-grain), built-in multi-objective support |
| W9 | Session seed = `hash(session_id)` — Python str hash is **not reproducible across processes** (PYTHONHASHSEED) | high | stable CRC-32 seed |
| W10 | `OptunaOptimizer` multi-objective = [composite, clarity] — both derived from the same weighted composite, not true objectives | medium | objectives = (speaker similarity, naturalness); single-objective stays composite×gate |
| W11 | Final selection = one composite winner only; no IDENTITY / NATURALNESS / CLARITY / BALANCED alternatives | high | best-by-strategy selection from validated candidates, all evidence shown |
| W12 | Validation split = interleaved segments of the **same utterance** (adjacent-part split); no multi-file support; no final-test set | high | robustness v2: file-level role assignment when multiple inputs, honest degradation notes, segment-type robustness probes (short/long/quiet/loud) |
| W13 | No linguistic-content axis: duration fidelity + HF retention mixed into "clarity"; no speech-rate/structure agreement | medium | new `content_preservation` axis (duration, VAD structure, speech-rate ratio, HF consonant retention) |
| W14 | No experiment memory: trials live inside one session; profile history recorded but never used; no model/index/input hashes; no cross-session warm start or dedup; no invalidation | high | `memory.py`: append-only experiment log keyed by hashes + versions, warm start, conversion cache, invalidation |
| W15 | No root-cause analysis; no separation of inference-param vs reference-data vs input-audio vs model-quality vs evaluation-model problems | high | `diagnostics.py`: evidence-based rule engine over the 5 classes, confidence per diagnosis, recommended engine action |
| W16 | No human-in-the-loop: A/B player exists but is not blind, captures no preference, learns nothing | medium | `preferences.py`: blind A/B pairs, preference store, preference-adjusted recommendation kept strictly separate from objective scores |
| W17 | `_analyze_f0` jitter/octave loops in pure Python (O(frames)); `_analyze_hnr` fallback re-extracts F0 (double extraction per analysis) | medium | vectorise; pass the F0 contour through |
| W18 | No benchmark protocol for engine versions; improvements not measured | medium | `scripts/optimizer_benchmark.py` + Phase 02 report with measured before/after |
| W19 | UI: single progress chart; no parameter-sensitivity/correlation/pareto visuals; no top-N comparison with audio; no diagnostics or preference panels | medium | UI v2 panels (all driven by real session data) |
| W20 | Vibrato/sibilance/breath/discontinuity measures absent from analysis despite being needed by the 15 detectors | medium | analysis v2 feature set |

## C. Non-issues verified during the audit (no action)

* VRAM/RAM/CPU sampling is already per-trial and honest (CPU-only sandbox
  reports CPU+RAM; CUDA path guarded).
* Audio pruning (top-5) prevents disk blow-up on long sessions.
* Model/index isolation per profile directory is enforced; `audio_path`
  sanitises traversal (`..`/leading `/`).
* The engine RLock serialises optimizer conversions against UI conversions.
* Gate-failing candidates can never win selection (mathematically excluded).
* filter_radius {0,3} semantics match the engine's real kernel-3 behaviour.
