"""RVC v2 checkpoint loading / exporting (format-critical code).

The inference checkpoint format is defined by the upstream project and must
remain byte-compatible so every existing RVC v1/v2 model loads:

    torch.save(OrderedDict(
        weight   = {k: v.half() for k in state_dict if "enc_q" not in k},
        config   = [18 positional hyper-parameters ...],
        info     = "<epoch>epoch",
        sr       = 40000,
        f0       = 1,
        version  = "v2",
    ))
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import torch

from rvc_platform.core.errors import CheckpointError, ModelNotFoundError
from rvc_platform.engine.rvc.models import models as rvc_models

logger = logging.getLogger(__name__)

SUPPORTED_VERSIONS = ("v1", "v2")


def build_generator(cpt: dict, is_half: bool):
    """Instantiate the matching SynthesizerTrn* from a checkpoint dict."""
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = int(cpt.get("f0", 1))
    version = str(cpt.get("version", "v1"))
    if version not in SUPPORTED_VERSIONS:
        raise CheckpointError(f"unsupported model version: {version}")
    if version == "v1":
        cls = (rvc_models.SynthesizerTrnMs256NSFsid if if_f0 == 1
               else rvc_models.SynthesizerTrnMs256NSFsid_nono)
    else:
        cls = (rvc_models.SynthesizerTrnMs768NSFsid if if_f0 == 1
               else rvc_models.SynthesizerTrnMs768NSFsid_nono)
    net_g = cls(*cpt["config"], is_half=False)
    del net_g.enc_q                       # posterior encoder unused at inference
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    if is_half:
        net_g = net_g.half()
    net_g.eval()
    return net_g


def load_checkpoint(path: Path, device: str, is_half: bool) -> tuple[torch.nn.Module, dict]:
    """Load an RVC inference checkpoint; returns (eval-mode generator, cpt dict)."""
    path = Path(path)
    if not path.is_file():
        raise ModelNotFoundError(f"checkpoint not found: {path}")
    try:
        cpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"unreadable checkpoint {path.name}: {exc}") from exc
    if not isinstance(cpt, dict) or "weight" not in cpt or "config" not in cpt:
        raise CheckpointError(
            f"{path.name} is not an RVC inference checkpoint "
            "(missing 'weight'/'config' keys)"
        )
    net_g = build_generator(cpt, is_half=is_half)
    net_g.to(device)
    if hasattr(net_g, "remove_weight_norm"):
        net_g.remove_weight_norm()
    return net_g, cpt


def inspect_checkpoint(path: Path) -> dict:
    """Read metadata without instantiating the network."""
    path = Path(path)
    if not path.is_file():
        raise ModelNotFoundError(f"checkpoint not found: {path}")
    try:
        cpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"unreadable checkpoint {path.name}: {exc}") from exc
    if not isinstance(cpt, dict):
        raise CheckpointError(f"{path.name}: not an RVC checkpoint")
    return {
        "sr": cpt.get("sr"),
        "f0": cpt.get("f0"),
        "version": cpt.get("version"),
        "info": cpt.get("info", ""),
        "has_index_ref": "index" in cpt,
        "speakers": [s.get("name") for s in cpt.get("speaker_info", [])] or None,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }


def normalize_speaker_info(speaker_info) -> list:
    """Upstream train/process_ckpt.py: dedupe + validate id/name pairs
    (ids 0..109, non-empty names), sorted by id."""
    result = []
    seen = set()
    for item in speaker_info or []:
        try:
            speaker_id = int(item["id"])
            speaker_name = str(item["name"])
        except (KeyError, TypeError, ValueError):
            continue
        if speaker_id < 0 or speaker_id > 109 or not speaker_name or speaker_id in seen:
            continue
        seen.add(speaker_id)
        result.append({"id": speaker_id, "name": speaker_name})
    return sorted(result, key=lambda item: item["id"])


def export_inference_checkpoint(
    state_dict: dict,
    hps,
    epoch: int,
    version: str,
    out_path: Path,
    sample_rate: int,
    if_f0: int,
) -> Path:
    """Export a training generator state to the portable RVC inference format.

    (This is the upstream ``savee`` logic, lifted into our architecture and
    writing to an explicit destination instead of ``assets/weights``.)
    """
    opt: "OrderedDict[str, object]" = OrderedDict()
    opt["weight"] = {}
    for key, value in state_dict.items():
        if "enc_q" in key:
            continue                      # posterior encoder never ships
        opt["weight"][key] = value.half()
    opt["config"] = [
        hps.data.filter_length // 2 + 1,
        32,
        hps.model.inter_channels,
        hps.model.hidden_channels,
        hps.model.filter_channels,
        hps.model.n_heads,
        hps.model.n_layers,
        hps.model.kernel_size,
        hps.model.p_dropout,
        hps.model.resblock,
        hps.model.resblock_kernel_sizes,
        hps.model.resblock_dilation_sizes,
        hps.model.upsample_rates,
        hps.model.upsample_initial_channel,
        hps.model.upsample_kernel_sizes,
        hps.model.spk_embed_dim,
        hps.model.gin_channels,
        hps.data.sampling_rate,
    ]
    opt["info"] = "%sepoch" % epoch
    opt["sr"] = sample_rate
    opt["f0"] = if_f0
    opt["version"] = version
    speaker_info = normalize_speaker_info(getattr(hps, "speaker_info", []))
    if speaker_info:
        opt["speaker_info"] = speaker_info
    f0_stats = getattr(hps, "f0_stats", None)
    if isinstance(f0_stats, dict) and f0_stats.get("median_hz"):
        opt["f0_stats"] = dict(f0_stats)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(opt, out_path)
    logger.info("exported inference checkpoint: %s (epoch %s)", out_path, epoch)
    return out_path
