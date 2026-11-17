#!/usr/bin/env python3
"""Vendor adaptation script.

Copies battle-tested neural-network modules from the OFFICIAL RVC repository
(RVC-Project/Retrieval-based-Voice-Conversion-WebUI, MIT licence, main branch
targeting Python 3.12) into ``src/rvc_platform/engine`` and rewrites their
internal imports to our package layout.

Engineering rationale (see docs/ARCHITECTURE.md §"RVC engine integration"):
the synthesiser / discriminator / NSF-HiFiGAN architectures and the RMVPE
model are checkpoint-format-critical components. Re-writing them from scratch
would risk silent weight-loading incompatibilities with every existing RVC
v2 checkpoint, so we vendor the reference implementation with minimal,
documented adaptations instead of re-inventing it.

This script is idempotent; run it only when upgrading the vendored snapshot.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / ".devref" / "rvc-official"
PKG = Path(__file__).resolve().parents[1] / "src" / "rvc_platform"

HEADER = '''# --------------------------------------------------------------------------- #
# Adapted from the official RVC repository:
#   https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
# (c) RVC-Project contributors — MIT licence.
# Adaptations performed by Iraqi Voice Studio (documented in
# docs/ARCHITECTURE.md and engine/rvc/ATTRIBUTION.md):
#   * imports rewritten to the ``rvc_platform`` package layout
#   * i18n / gradio / webui couplings removed
#   * asset paths routed through rvc_platform.infrastructure.paths
#   * no algorithmic changes to numerics or model structure
# --------------------------------------------------------------------------- #
'''

# target -> source
VENDOR_MAP = {
    # neural network modules (numerically identical to upstream)
    "engine/rvc/models/models.py": "infer/module/models.py",
    "engine/rvc/models/modules.py": "infer/module/modules.py",
    "engine/rvc/models/attentions.py": "infer/module/attentions.py",
    "engine/rvc/models/commons.py": "infer/module/commons.py",
    "engine/rvc/models/transforms.py": "infer/module/transforms.py",
    # content encoder
    "engine/rvc/hubert.py": "infer/hubert.py",
    # F0 extractors
    "engine/rvc/rmvpe.py": "infer/rmvpe.py",
    "engine/rvc/fcpe.py": "infer/fcpe.py",
    # optional CUDA-graph fast path
    "engine/rvc/cuda_graph.py": "tools/cuda_graph.py",
    # training numerics
    "training/losses.py": "train/losses.py",
    "training/mel_processing.py": "train/mel_processing.py",
    "training/dataset.py": "train/data_utils.py",
    # audio slicer used by preprocessing
    "infrastructure/audio/slicer.py": "train/dataset/slicer2.py",
    # any-format audio loader (ffmpeg based)
    "infrastructure/audio/_ffmpeg_load.py": "infer/audio.py",
}

REWRITES = [
    ("from infer.module import attentions, commons, modules",
     "from rvc_platform.engine.rvc.models import attentions, commons, modules"),
    ("from infer.module import commons, modules",
     "from rvc_platform.engine.rvc.models import commons, modules"),
    ("from infer.module import commons",
     "from rvc_platform.engine.rvc.models import commons"),
    ("from infer.module.commons import get_padding, init_weights",
     "from rvc_platform.engine.rvc.models.commons import get_padding, init_weights"),
    ("from infer.module.modules import LayerNorm",
     "from rvc_platform.engine.rvc.models.modules import LayerNorm"),
    ("from infer.module.transforms import piecewise_rational_quadratic_transform",
     "from rvc_platform.engine.rvc.models.transforms import piecewise_rational_quadratic_transform"),
    ("from tools.cuda_graph import cuda_graph_enabled, run_cuda_graph",
     "from rvc_platform.engine.rvc.cuda_graph import cuda_graph_enabled, run_cuda_graph"),
    ("from tools.cuda_graph import run_cuda_graph",
     "from rvc_platform.engine.rvc.cuda_graph import run_cuda_graph"),
    ("from train.mel_processing import spectrogram_torch",
     "from rvc_platform.training.mel_processing import spectrogram_torch"),
    ("from train.utils import load_filepaths_and_text, load_wav_to_torch",
     "from rvc_platform.training.utils import load_filepaths_and_text, load_wav_to_torch"),
    ("from infer.hubert import extract_hubert_features, load_hubert_model",
     "from rvc_platform.engine.rvc.hubert import extract_hubert_features, load_hubert_model"),
    ("from infer.rmvpe import RMVPE",
     "from rvc_platform.engine.rvc.rmvpe import RMVPE"),
    ("from infer.fcpe import FCPEInfer",
     "from rvc_platform.engine.rvc.fcpe import FCPEInfer"),
    ("from infer.module import models",
     "from rvc_platform.engine.rvc import models"),
    # i18n couplings in upstream -> neutralised
    ("from i18n.i18n import I18nAuto", "I18nAuto = None  # removed: upstream i18n coupling"),
]


def main() -> None:
    assert REPO.is_dir(), f"reference repo missing: {REPO}"
    for rel_target, rel_src in VENDOR_MAP.items():
        src = REPO / rel_src
        dst = PKG / rel_target
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8")
        for old, new in REWRITES:
            text = text.replace(old, new)
        # drop upstream module-level logger side effects (keep logging import)
        text = re.sub(r"^i18n = I18nAuto\(\)\n", "", text, flags=re.M)
        # relative path assumptions
        text = text.replace(
            "PROJECT_ROOT = Path(__file__).resolve().parent.parent",
            "PROJECT_ROOT = Path(__file__).resolve().parents[3]",
        )
        dst.write_text(HEADER + text, encoding="utf-8")
        print(f"vendored {rel_src} -> {rel_target}")


if __name__ == "__main__":
    main()
