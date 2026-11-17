"""RVC engine integration tests (CPU, no pretrained assets).

These validate the checkpoint round-trip format, the full file-conversion
pipeline mechanics and the real-time block pipeline mechanics using a REAL
randomly-initialised v2/40k synthesiser, a small REAL FAISS index and the
REAL Praat "pm" F0 extractor. The only mocked component is the HuBERT
encoder (374 MB download) — replaced by a deterministic feature stand-in;
documented in tests/conftest.py.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from rvc_platform.core.entities import ConversionSettings
from rvc_platform.engine.rvc import checkpoint as ckpt_mod
from rvc_platform.engine.rvc.index import FeatureIndex, build_index
from rvc_platform.engine.rvc.models import models as rvc_models


def test_checkpoint_roundtrip_matches_rvc_format(synthetic_model_file, rvc_hparams_40k):
    net_g, cpt = ckpt_mod.load_checkpoint(synthetic_model_file, "cpu", False)
    assert isinstance(net_g, rvc_models.SynthesizerTrnMs768NSFsid)
    assert cpt["sr"] == 40_000 and cpt["f0"] == 1 and cpt["version"] == "v2"
    assert len(cpt["config"]) == 18
    # weights are fp16 in the file, model runs fp32 on CPU
    assert next(net_g.parameters()).dtype == torch.float32
    info = ckpt_mod.inspect_checkpoint(synthetic_model_file)
    assert info["sr"] == 40_000 and info["version"] == "v2"


def test_generator_infer_shapes():
    net_g = rvc_models.SynthesizerTrnMs768NSFsid(
        1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11],
        [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512,
        [16, 16, 4, 4], 109, 256, is_half=False, sr=40_000,
    )
    net_g.eval()
    phone = torch.randn(1, 40, 768)
    lengths = torch.tensor([40])
    pitch = torch.randint(1, 255, (1, 40))
    pitchf = torch.rand(1, 40) * 200 + 80
    sid = torch.tensor([0])
    with torch.no_grad():
        audio = net_g.infer(phone, lengths, pitch, pitchf, sid)[0]
    assert audio.shape[-1] == 40 * 400  # 10 ms frames -> 40 kHz samples


def test_index_build_and_blend(tiny_feature_dir, tmp_path):
    added = build_index(tiny_feature_dir, tmp_path, "testrun", "v2")
    assert added.exists()
    index = FeatureIndex(added)
    assert index.size > 0 and index.dim == 768
    query = np.random.default_rng(2).standard_normal((5, 768)).astype(np.float32)
    blended = index.blend(query, index_rate=0.5)
    assert blended.shape == query.shape
    # at rate 0 the identity is returned untouched
    np.testing.assert_allclose(index.blend(query, 0.0), query)


def test_file_conversion_end_to_end(
    project_env, speech_wav_file, synthetic_model_file, mock_hubert
):
    from rvc_platform.engine.rvc.inference import RVCInferenceEngine
    from rvc_platform.infrastructure.audio.pipeline import AudioProcessingEngine
    from rvc_platform.utils.device import DeviceProfile

    profile = DeviceProfile("cpu", False, None, 0.0, 0.0)
    engine = RVCInferenceEngine(profile, AudioProcessingEngine())
    info = engine.load_model(synthetic_model_file, None)
    assert engine.status == "ready"
    assert info.sample_rate == 40_000

    # build a real small index for the loaded model and reload with it
    feats_dir = project_env / "3_feature768"
    feats_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4)
    np.save(feats_dir / "a.npy", rng.standard_normal((30, 768)).astype(np.float32))
    added = build_index(feats_dir, project_env / "run", "conv", "v2")
    engine.load_model(synthetic_model_file, added)

    settings = ConversionSettings(
        pitch_shift=2, f0_method="pm", index_rate=0.3,
        filter_radius=3, rms_mix_rate=0.25, protect=0.33, denoise=False,
    )
    out_path = project_env / "converted.wav"
    result = engine.convert_file(speech_wav_file, out_path, settings)
    assert result.output_path.exists()
    import soundfile as sf

    data, sr = sf.read(str(result.output_path))
    assert sr == 40_000
    assert len(data) >= 16_000 * 3.2  # ~4s input minus silence margins
    assert float(np.max(np.abs(data))) > 0.01  # non-silent output
    assert result.realtime_factor > 0  # measurable speed reported
    engine.unload()
    assert engine.status == "unavailable"


def test_realtime_block_pipeline(project_env, synthetic_model_file, mock_hubert):
    from rvc_platform.engine.rvc.realtime import RealTimeRVC
    from rvc_platform.utils.device import DeviceProfile

    profile = DeviceProfile("cpu", False, None, 0.0, 0.0)
    engine = RealTimeRVC(profile, block_ms=160, crossfade_ms=8, extra_ms=250)
    engine.f0_method = "pm"  # CPU F0 extractor, no weights needed
    engine.load_model(synthetic_model_file, None)
    assert engine.status == "ready"

    rng = np.random.default_rng(7)
    chunk = engine.block_frame_16k
    out_expected = engine.block_units * (engine.tgt_sr // 100)
    outs = []
    for i in range(12):
        t = np.arange(chunk) / 16_000 + i * 0.16
        wave = 0.2 * np.sin(2 * np.pi * 160 * t) + 0.01 * rng.standard_normal(chunk)
        out = engine.process_block(wave.astype(np.float32))
        assert out.shape[0] == out_expected, "each block must be block-aligned"
        outs.append(out)
    joined = np.concatenate(outs)
    assert float(np.max(np.abs(joined))) > 0.01  # real audio flowing
    stats = engine.latency()
    assert stats.count == 12
    assert stats.min_ms > 0 and stats.avg_ms >= stats.min_ms
    # mid-stream setting changes must not crash and take effect
    engine.update_settings(pitch_shift=5, index_rate=0.0)
    engine.process_block(np.zeros(chunk, dtype=np.float32) + 1e-4)
