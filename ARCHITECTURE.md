# ARCHITECTURE — Iraqi Voice Studio

> Engineering record of the system's structure and of *why* every significant
> technical decision was taken. Companion documents: `README.md` (usage),
> `API.md` (interface), `TRAINING.md` / `REALTIME.md` (workflows).

## 1. System overview

Iraqi Voice Studio is a **voice-to-voice (Audio-to-Audio) conversion platform**
built around the RVC v2 architecture, specialised for Iraqi-Arabic speech via
dataset guidance, with a full training loop (resumable checkpoints), a
low-latency streaming engine, a web control surface, and cross-platform
automation.

```
                       ┌──────────────────────────────────────────────────┐
                       │                  Web SPA (RTL)                   │
                       │  dashboard · models · convert · realtime ·       │
                       │  training · settings      (vanilla ES modules)   │
                       └───────────────┬──────────────────────────────────┘
                                       │ HTTP + WebSocket (/api/...)
                       ┌───────────────┴──────────────────────────────────┐
                       │        API Layer — FastAPI (composition root)    │
                       │  auth · routes(system/models/convert/training/   │
                       │  realtime-ws/tunnel) · pydantic schemas          │
                       └───────────────┬──────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼───────────────────────────────┐
        │                              │                                │
┌───────┴────────┐          ┌───────────┴──────────┐        ┌────────────┴──────────┐
│   Application  │          │     Real-Time        │        │      Infrastructure   │
│  services      │          │  streaming sessions  │        │  config · logging ·   │
│ (models/conv/  │          │  (WS -> thread ->    │        │  audio DSP · repos ·  │
│  training/     │          │   RealTimeRVC)       │        │  processes · security │
│  tunnel/system)│          └───────────┬──────────┘        └────────────┬──────────┘
└───────┬────────┘                      │                                │
        │                    ┌───────────┴──────────┐                     │
        │                    │   RVC v2 ENGINE      │─────────────────────┘
        ├───────────────────>│  inference · realtime│
        │                    │  checkpoint · index  │
        │                    │  hubert · rmvpe/fcpe │  (vendored + adapted
        │                    │  models (VITS/NSF)   │   upstream numerics)
        │                    └───────────┬──────────┘
┌───────┴────────────────────────────────┴─────────────────────────────────┐
│                        TRAINING PIPELINE                                  │
│  check -> preprocess -> F0 -> HuBERT feats -> FAISS index -> filelist    │
│  -> GAN trainer (resumable full-state checkpoints) -> export             │
└───────────────────────────────────────────────────────────────────────────┘
```

## 2. Layering rules (Clean Architecture)

| Layer | Contains | Depends on | Never contains |
|---|---|---|---|
| `core` | entities, `Protocol` interfaces, errors, event bus | stdlib only | torch, IO, frameworks |
| `application` | use-case services (model mgmt, conversion, training control, tunnel, realtime sessions, downloads) | core + engine/infra via interfaces | route handlers, CLI details |
| `engine/rvc` | the RVC v2 model engine (vendored nets + our adapters) | torch, core | HTTP, CLI, YAML |
| `training` | data pipeline + trainer + checkpointing | engine, infra | HTTP |
| `infrastructure` | config, logging, audio DSP, persistence, process mgmt, security | core | business logic |
| `api` / `cli` / `web` | delivery mechanisms | application services | domain rules |

The **composition root is a single function** (`api/app.py::create_app` /
CLI wiring). Swapping an implementation (e.g. a future non-RVC engine)
replaces one `Protocol` implementation and one line at the root — verified by
the runtime-checkable `Protocol`s in `core/interfaces.py`.

## 3. Key decisions & their evidence

### 3.1 RVC v2 as the engine — with vendored numerics
The synthesiser (`SynthesizerTrnMs768NSFsid` + NSF-HiFiGAN + flow),
discriminators, RMVPE and training numerics are **vendored from the official
repository with documented adaptations** (`engine/rvc/ATTRIBUTION.md`,
re-vendor via `scripts/dev_vendor_rvc.py`). Rationale: these components are
*checkpoint-format-critical* — every tensor name is part of the RVC weight
format used by thousands of existing models and by the pretrained bases.
Re-implementing them invites silent incompatibilities; vendoring preserves
byte-level compatibility (verified by
`tests/integration/test_engine.py::test_checkpoint_roundtrip_matches_rvc_format`).

### 3.2 F0 extraction — RMVPE default, registry-switchable
Researched options: `rmvpe` (InterSpeech 2023; best robustness on noisy
speech, GPU-fast), `fcpe` (fast context model), `pm` (Praat; no weights),
`crepe` (requires legacy torchcrepe pinned to old librosa — rejected),
`harvest`/`dio` (slower, more gross-pitch errors on Arabic — rejected).
The registry (`engine/rvc/f0.py`) makes the method a configuration choice;
the UI exposes exactly the three implemented methods — no phantom settings.

### 3.3 Sample rates
32k / 40k / 48k templates ship with the package
(`training/templates/*.json`, project `configs/training/` overrides).
40k is the default (widest ecosystem of pretrained bases). v1/v2 share the
same JSON model section — the version flag selects the 256/768-dim encoder
class, matching the upstream main branch behaviour.

### 3.4 Training — single-process by design
One GPU (or CPU) per run, no DDP. Evidence: the target deployment for
training is Colab (single T4/L4/A100) or a workstation; DDP adds failure
modes (gloo ports, rank-0 IO races) that cannot be tested here, violating
the project rule "don't claim success for untested functionality". The trade
-off and the upgrade path (upstream's spawn-based DDP) are documented.

### 3.5 Resumable training — full state + legacy compatibility
Checkpoints persist **model + optimizer + scheduler + scaler + epoch +
global_step + RNG** + `state.json` (dataset signature, run config). Resume
order: (1) own full-state, (2) vanilla-RVC `G_/D_` legacy checkpoints,
(3) pretrained bases. A dataset-signature mismatch hard-fails with an
actionable message (guard covered by test). Periodic exports
(`exports/<exp>_e<epoch>_s<step>.pth`) are standard RVC inference
checkpoints usable by any RVC tool.

### 3.6 Real-time pipeline
Faithful port of the upstream streaming design: 10 ms frame grid; sliding
input window (lookback + crossfade + search + block); 1024-frame pitch
cache; `skip_head`/`return_length` slicing so only new audio is emitted;
SOLA output crossfade; RMS envelope mixing; input noise gate. Additions:
per-block latency instrumentation (min/avg/p95/max), mid-stream safe
parameter updates, decoupled transport (WebSocket) instead of sound-device
callbacks, and an honest `realtime-benchmark` command that measures the
**real** engine with synthetic blocks (no mic needed).

Measured on this build machine (CPU-only, torch 2.7.1+cpu, no GPU):
avg 519 ms / p95 560 ms per 160 ms block (re-measured: vCPU×2, fp32, real v2/40k model; min 464 / max 866) — **above the real-time budget**,
as physics dictates for a 36M-param synthesiser + HuBERT + RMVPE on CPU.
Upstream reports 170 ms end-to-end on a suitable CUDA GPU (90 ms with ASIO
drivers on Windows). The platform surfaces both numbers honestly instead of
pretending; run `make realtime MODEL=<name>` on your hardware for truth.

### 3.7 Python 3.12 + uv, CPU-default torch
The upstream main branch targets Python 3.12 (fairseq-free, transformers
HuBERT) — we pin `>=3.12,<3.13`. `uv` manages the venv/lockfile.
**CPU wheels install by default** so every environment boots (Colab, CPU
laptops, CI); GPU users run  `make env-gpu CUDA=cu128` which installs the
matching CUDA wheels into the project venv (available variants for the locked
torch 2.7.x: cu118/cu126/cu128 only). Because plain `uv run` re-syncs the
venv from `uv.lock`, GPU wheels persist only with `UV_NO_SYNC=1` (wired
automatically by the Colab installer). No global pip installs, no
cross-project pollution; all caches/artifacts under the project root with
per-directory env redirections.

### 3.8 Web stack — FastAPI + static SPA, no Node build
The dashboard is vanilla ES modules (RTL Arabic) served by FastAPI. No npm
build step: the app must run on Colab/Windows/offline boxes where a Node
toolchain is unreasonable. Real-time audio travels over WebSocket (Int16 PCM
frames), capture via `AudioWorklet` at 16 kHz, playback scheduled on an
`AudioContext` at the model rate with a 2-block jitter cushion.

### 3.9 Security model
Uploaded files are untrusted: extension + magic-byte + size validation,
path-traversal-proof name sanitisation, decode timeouts, 40 MB audio /
300 MB model caps. Auth is off on localhost by default and **auto-enforced
(bearer token) whenever a Cloudflare tunnel is opened**; the token is
auto-generated into `logs/auth.token` (never committed). Quick tunnels are
ephemeral URLs by definition (new URL per start) — documented, not hidden.

### 3.10 Iraqi-dialect quality strategy
RVC converts *timbre*, it does not translate dialect; Iraqi character comes
from the training data. The platform's contribution is the guidance layer:
dataset checker (duration, corruption, duplicates, silence ratio,
≥10-minute recommendation), preprocessing tuned exactly to upstream
constants, RMVPE F0 (robust on the noisy recordings typical of Iraqi
datasets), `protect` to keep voiceless consonants crisp (Arabic emphatics),
index-rate retrieval to reduce timbre leakage, and per-recording knobs
(pitch, filter radius, RMS mixing) documented in `TRAINING.md`.

## 4. Data & directory contract

```
iraqi-voice-platform/
├─ src/rvc_platform/            # the Python package (see layer table)
├─ configs/                     # default.yaml + training templates (+ user.yaml)
├─ scripts/{colab,ubuntu,windows}/  # executable installers
├─ tests/{unit,integration}/    # pytest suite (see §6)
├─ docs (README, API, TRAINING, REALTIME, ...)
├─ assets/    # downloaded: hubert_base/ · rmvpe/ · pretrained_v2/   (~1.3 GB)
├─ models/    # voice models registry (.pth + optional .index + registry.json)
├─ data/      # raw · uploads · conversions
└─ runs/<exp> # 0_gt_wavs · 1_16k_wavs · 2a_f0 · 2b-f0nsf · 3_feature768 ·
              # checkpoints/ · exports/ · filelist.txt · state/progress/logs
```

Everything lives under one root; `RVC_PLATFORM_ROOT` and per-directory env
redirects move heavy artifacts to Drive/other disks without code changes.

## 5. Process & failure model

* Trainer runs as a **subprocess** (`training.worker`): stdout JSONL events,
  atomic `progress.json`, SIGTERM = cooperative stop (checkpoint first,
  exit 0). A crashed trainer never takes the API down.
* `ProcessManager` supervises named processes (trainer, cloudflared) with
  process-group SIGTERM→SIGKILL escalation and log capture.
* Tunnel start is refused unless the local server answers; URL scraped from
  the log with retries; auth is force-enabled while exposed.
* All cross-layer errors are typed (`core/errors.py`) — no silent excepts.

## 6. Verification posture (what tests actually prove)

`tests/` run on CPU with real numerics; the **only** mock is the HuBERT
encoder in two integration tests (374 MB asset) — synthesiser, RMVPE-free
F0 path (`pm`), FAISS, losses, trainer and API are all real.

* unit: config layering, DSP numerics, path/env isolation, checkpoint I/O
  round-trips, latency percentiles, auth, event bus;
* integration: checkpoint format round-trip vs RVC format, file conversion
  end-to-end (incl. FAISS blending), real-time block pipeline, **full GAN
  epoch + true resume + legacy resume + dataset-signature guard**, dataset
  preprocessing/slicing, API CRUD/auth/upload rejection, WebSocket session
  with binary streaming and stats.

Additionally verified live in this environment (real HuBERT + RMVPE):
file conversion of 4 s audio (7.9 s on CPU), realtime benchmark
(min/avg/p95 = 464/519/560 ms per 160 ms block on CPU, real v2/40k weights — honest verdict),
API+SPA serving, model load via API, clean shutdown.

## 7. ULTIMATE AI VOICE OPTIMIZER (optimizer/ package + web tab)

A per-model intelligent RVC v2 parameter tuner added as a first-class
subsystem. Design decisions and their evidence:

* **Reference speaker profile per model** (`optimizer/profile.py`) —
  neural speaker-verification embeddings (WavLM x-vector
  `microsoft/wavlm-base-plus-sv` — a CORE dependency via `transformers`,
  no new install; ECAPA-TDNN via the existing SpeechBrain integration,
  optional) + F0 statistics + MFCC/band/formant timbre statistics + HNR/
  jitter/ shimmer anchors. Multiple profiles per model; per-model isolation
  enforced by the store layout (`data/optimizer/reference/<model>/…`).
  Research basis: cosine similarity between speaker-verification embeddings
  is the standard VC identity metric (YourTTS, RVC overtraining detectors);
  WavLM-based similarity tracks *perceptual* similarity well (VoxSim, 2024)
  — hence a two-model ensemble with an **agreement check** (large spread
  degrades the reliability verdict instead of hiding inside the mean).
* **Analysis at one canonical 16 kHz rate** (`optimizer/analysis.py`) —
  every axis compares like with like regardless of model output rate. F0 via
  the engine's own registry (RMVPE default, Praat fallback); VAD is
  dependency-free adaptive energy thresholding with hangover.
* **Nine-axis evaluation with quality gates** (`optimizer/scoring.py`) —
  speaker similarity (per-backend + **per-backend calibrated** ensemble —
  see `optimizer/calibration.py` — never a hidden average), timbre, pitch
  consistency, naturalness (distance to the *speaker's own*
  HNR/jitter/shimmer/vibrato-irregularity anchors, not absolutes), clarity
  (vs input), artifacts (the real 15-detector bank in `optimizer/quality.py`
  — see below), prosody preservation (**true** frame-envelope correlation +
  time-warped F0 contour correlation + speech-rate agreement vs input),
  content preservation (duration, voiced structure, speech-rate ratio,
  consonant-band retention vs input — the linguistic-content axis), audio
  quality. Composite = weighted mean (speaker identity leads by design,
  weights user-adjustable); candidates failing the artifact/clarity/prosody/
  content gates are mathematically excluded from selection — **a distorted
  voice can never win on similarity alone**. Pareto front over
  (speaker, naturalness, clarity, artifact) exposes trade-offs, and
  best-by-strategy (identity/naturalness/clarity/balanced) exposes four
  explicit winners instead of one hidden weighting.
* **Per-backend speaker-score calibration** (`optimizer/calibration.py`) —
  raw cosines from different SV models live on different scales, so each
  backend gets a same-speaker cohort (intra-speaker pairwise cosines from
  the profile, persisted in profile.json) and raw scores are converted to
  margins + calibrated 0..1 values before any ensembling; backends without
  a cohort are reported uncalibrated and the ensemble is withheld rather
  than averaging incomparable scales.
* **Real artifact intelligence** (`optimizer/quality.py`) — 15 detectors
  (metallic, robotic, buzzing, crackling, resonance, pitch stability,
  vibrato, sibilance, THD, spectral holes, clipping, noise, breathing,
  broken consonants, unnatural transitions), each with severity, time
  locations, evidence and confidence; cross-trial recurrence analysis links
  artifact severity to parameters (Spearman, always labelled statistical
  association, never causation).
* **Parameter relationship discovery** (`optimizer/interactions.py`) —
  binned marginal effects, sensitivity (anchored to the session's composite
  spread), one-way ANOVA importance (eta²), pairwise interactions (2-D cell
  deviation from the additive main-effects model; median-split cells with a
  minimum cell size — sparse data degrades to an honest "insufficient
  evidence" instead of inventing interactions), unstable-region detection
  and evidence-backed recommendations — all derived from the session's real
  trials only.
* **Root-cause diagnostics** (`optimizer/diagnostics.py`) — separates five
  problem classes (inference parameters / reference data / input audio /
  model training quality via honest proxies incl. the optional round-trip
  ceiling / evaluation models); every diagnosis carries evidence, ranked
  causes, confidence, a recommendation and — when one exists — an engine
  action grounded in the session's own measurements.
* **Experiment memory** (`optimizer/memory.py`) — append-only log keyed by
  model/index/input hashes + optimizer version: warm start (prior best
  gate-passing parameters seed stage A), cross-session duplicate prevention
  and a conversion cache that replays the stored analysis + raw
  similarities through the *current* calibration/weights without
  re-converting (version mismatches are excluded, never mixed).
* **Human-in-the-loop** (`optimizer/preferences.py`) — server-backed blind
  A/B pairs (labels randomised, parameters hidden until the choice, one
  choice per pair) + preference learning kept strictly separate: a bounded
  (≤ 5%) boost affects only the combined *ranking* view; objective axes are
  never modified.
* **Generalisation protocol** (`optimizer/robustness.py`) — file-level
  search/validation/test role assignment when multiple inputs are supplied
  (segment-level fallback with an explicit note for a single take); probe-
  type robustness evaluation (short/long/quiet/loud/voiced-heavy/…) with
  stability = mean − 0.5×std; the final test runs on a take that never
  participated in search or validation.
* **Search backends behind one interface** (`optimizer/search.py`) —
  Optuna TPE (multivariate, constant liar) or NSGA-II with objectives
  (speaker similarity, naturalness) — two REAL axes, not two views of one
  composite — when `optuna` is installed (quality extra); otherwise a
  dependency-free **staged** backend whose behaviour genuinely changes per
  stage: A) warm-started golden-ratio low-discrepancy exploration (30% of
  budget), B) TPE-style top-quantile guidance with categorical Thompson
  sampling + adaptive KDE bandwidth (45%), C) fine-grained Gaussian
  refinement around the top-3 incumbents (25%); multi-objective mode keeps
  a Pareto archive and samples around random non-dominated members.
  Deterministic CRC-32 seeds (Phase 01 used process-salted `hash()`).
  Quantised key cache guarantees no duplicate trial is ever executed. The
  space is honest: `filter_radius` searches {0, 3} because the engine's
  median filter is a fixed kernel-3 active only when radius > 2.
* **Resumable sessions** (`optimizer/session.py`) — state.json is written
  atomically after every trial; restart-safe resume replays the trial log
  into the sampler. Model-swap guard: if the engine's loaded model changes
  mid-run, remaining trials fail loudly instead of scoring the wrong voice.
* **Anti-overfitting** — with multiple inputs, whole FILES are assigned
  to search/validation/test roles (single takes degrade to segment-level
  splits with an explicit note); the top-k candidates are re-validated on
  held-out probes of distinct types (short/long/quiet/loud) and the
  *validated + stability-aware* score decides the winner; validation drops
  > 15% surface a reliability note. Short inputs (< 3.5 s) honestly skip
  validation (noted), and so do runs where no candidate passed the gates.
* **Honesty rules** — every report carries reliability notes (profile
  quality, embedder availability, embedder-set mismatches) and a disclaimer
  that all scores are relative search metrics, not identity-match
  probabilities.
* **One engine, one lock** — trials reuse the already-loaded inference
  engine (no model reloads); the engine's RLock serialises against UI
  conversions; a single background thread + per-trial resource sampling
  (CPU/RAM via psutil, VRAM via torch.cuda) powers the live dashboard.

Verified live in this environment (real HuBERT + synthetic v2/48k model +
real WavLM embedder): profile build over 2 samples ("good" reliability),
full quick session through the browser UI (baseline + trials + validation +
best), live dashboard polling, results/chart/A-B rendering, blind mode,
APPLY (settings carried into `/api/convert/settings` → one-click fill in
the convert tab) and JSON report export — zero console errors.
