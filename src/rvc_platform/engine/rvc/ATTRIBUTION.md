# Vendored RVC components — provenance & adaptation log

## Source

Upstream repository: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
Upstream licence: MIT (see `LICENSE` in the upstream repository). The MIT
licence and copyright notice are preserved in every vendored file header.

Cloned reference commit: `main` branch, Python 3.12 target (the branch used
for this project's research snapshot; re-vendor via
`scripts/dev_vendor_rvc.py` to upgrade — the script is idempotent).

## Why vendoring (engineering rationale)

The synthesiser (`models.py`), its sub-modules (`modules.py`,
`attentions.py`, `commons.py`, `transforms.py`), the RMVPE pitch model and
the training numerics (`losses.py`, `mel_processing.py`, `data_utils.py`)
are **checkpoint-format-critical**: every tensor name, layer shape and
numeric constant is part of the RVC v1/v2 weight format. Re-implementing
them from scratch invites silent incompatibilities with the thousands of
existing RVC checkpoints (including all pretrained bases and every model
trained by the RVC ecosystem). We therefore integrate the reference
implementation with *minimal, documented* adaptations instead of
re-inventing it, per the project's own rule: "do not re-invent existing RVC
components without a genuine engineering need."

## Adapted files

| Our file | Upstream file | Adaptations |
| --- | --- | --- |
| `engine/rvc/models/*.py` | `infer/module/*.py` | imports only |
| `engine/rvc/hubert.py` | `infer/hubert.py` | asset path via `infrastructure.paths` |
| `engine/rvc/rmvpe.py` | `infer/rmvpe.py` | device policy via `utils.device` |
| `engine/rvc/fcpe.py` | `infer/fcpe.py` | none (import rewrites) |
| `engine/rvc/cuda_graph.py` | `tools/cuda_graph.py` | none (import rewrites) |
| `training/losses.py` | `train/losses.py` | imports only |
| `training/mel_processing.py` | `train/mel_processing.py` | imports only |
| `training/dataset.py` | `train/data_utils.py` | imports only |
| `infrastructure/audio/slicer.py` | `train/dataset/slicer2.py` | none (vendored verbatim; the upstream file itself credits librosa/DDSP-SVC) |
| `infrastructure/audio/_ffmpeg_load.py` | `infer/audio.py` | reference copy for the ffmpeg decode path (our `io.py` re-implements the contract) |

Additional mechanical refactor: explicit ``super(ClassName, self).__init__()``
forms were converted to zero-argument ``super().__init__()`` (semantically
identical) so the vendored classes can be subclassed — needed by the test
suite's discriminator shrink and by future model variants.

**No algorithmic or structural changes** were made to model numerics. The
parts we *did* write ourselves (inference pipeline, realtime session,
trainer shell, checkpointing, index builder, F0 registry) are thin, tested
adaptation layers around these vendored numerics; their behaviour is
validated against upstream's own driver code (`infer/vc/pipeline.py`,
`infer/rtrvc.py`, `realtime_gui.py`, `train/train.py`).

## Third-party credits inside vendored code

* NSF-HiFiGAN vocoder — from DiffSinger (MIT), embedded in upstream `models.py`.
* SOLA crossfade — from DDSP-SVC, embedded in upstream `realtime_gui.py`
  (used by our `engine/rvc/realtime.py`).
* `slicer.py` — derived from librosa (ISC) via upstream.
* RMVPE — Wei et al., InterSpeech 2023 (upstream ships weights;
  we download them from the official HuggingFace release).
* HuBERT/ContentVec — Facebook fairseq research licence, converted
  checkpoint released by the RVC author (lj1995/VoiceConversionWebUI).
