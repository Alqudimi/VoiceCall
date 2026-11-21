"""Shared fixtures: isolated project root per test, config reset, synthetic assets."""
from __future__ import annotations


import numpy as np
import pytest
import torch


@pytest.fixture()
def project_env(tmp_path, monkeypatch):
    """Redirect the whole platform into a temp directory."""
    monkeypatch.setenv("RVC_PLATFORM_ROOT", str(tmp_path))
    from rvc_platform.infrastructure.paths import reset_paths

    reset_paths()
    from rvc_platform.infrastructure.config import loader as cfg_loader

    cfg_loader.reset_config()
    yield tmp_path
    reset_paths()
    cfg_loader.reset_config()


@pytest.fixture()
def sample_rate_16k():
    return 16_000


def speech_like(sr: int, seconds: float, f0: float = 160.0, seed: int = 3) -> np.ndarray:
    """Synthetic speech-like signal: harmonics + amplitude mod + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    sig = 0.4 * np.sin(2 * np.pi * f0 * t)
    sig += 0.2 * np.sin(2 * np.pi * 2 * f0 * t)
    sig += 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
    env = 0.5 * (1 + np.sin(2 * np.pi * 2.5 * t)) ** 2
    sig = sig * env
    sig += 0.005 * rng.standard_normal(sig.shape)
    # silence paddings to exercise trimming/slicing
    sig[: int(sr * 0.3)] = 1e-6
    sig[-int(sr * 0.3) :] = 1e-6
    return sig.astype(np.float32)


@pytest.fixture()
def speech_wav_file(project_env, tmp_path):
    import soundfile as sf

    path = tmp_path / "speech.wav"
    sf.write(str(path), speech_like(16_000, 4.0), 16_000, subtype="PCM_16")
    return path


SYNTH_CONFIG_40K = [
    1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256, 48_000,
]


@pytest.fixture(scope="session")
def rvc_hparams_40k():  # official v2/48k pairing
    from rvc_platform.training.run_config import load_template
    from rvc_platform.training.utils import HParams

    template = load_template("48k", "v2")
    return HParams.from_template(
        {"train": dict(template["train"]), "data": dict(template["data"]),
         "model": dict(template["model"])},
        version="v2", sample_rate=48_000, if_f0=1, exp_name="test",
        model_dir="", total_epoch=1, save_every_epoch=1,
        save_every_weights="1", if_latest=1, if_cache_data_in_gpu=False,
        pretrainG="", pretrainD="", gpus="0",
    )


@pytest.fixture(scope="session")
def synthetic_model_file(tmp_path_factory, rvc_hparams_40k):
    """Real RVC-format checkpoint with random weights (v2, f0, 48k)."""
    from rvc_platform.engine.rvc.checkpoint import export_inference_checkpoint
    from rvc_platform.engine.rvc.models import models

    root = tmp_path_factory.mktemp("models")
    net_g = models.SynthesizerTrnMs768NSFsid(
        *SYNTH_CONFIG_40K[:17], is_half=False, sr=48_000
    )
    out = root / "synthetic_v2_48k.pth"
    export_inference_checkpoint(net_g.state_dict(), rvc_hparams_40k, 3, "v2", out, 48_000, 1)
    return out


def fake_hubert_features(model, source, version, padding_mask=None):
    """Deterministic stand-in for the 374 MB HuBERT encoder (768-dim, 20 ms hop)."""
    n = source.shape[-1]
    t = max(1, n // 320)
    feats = torch.zeros(1, t, 768, dtype=source.dtype, device=source.device)
    feats[0, :, 0] = torch.linspace(0, 1, t, device=source.device).to(source.dtype)
    return feats


class _FakeHubert:
    """Object stand-in; only used as an opaque handle by the engines."""


@pytest.fixture()
def mock_hubert(project_env, monkeypatch):
    """Patch the imported symbols in the engine namespaces + create the asset
    marker so the engines' asset-existence guard passes without the 374 MB
    download. Only HuBERT is mocked — synthesiser/F0/DSP remain fully real.
    """
    hubert_dir = project_env / "assets" / "hubert_base"
    hubert_dir.mkdir(parents=True, exist_ok=True)
    (hubert_dir / "config.json").write_text("{}", encoding="utf-8")
    (hubert_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    import rvc_platform.engine.rvc.inference as inf
    import rvc_platform.engine.rvc.realtime as rt

    monkeypatch.setattr(inf, "extract_hubert_features", fake_hubert_features)
    monkeypatch.setattr(rt, "extract_hubert_features", fake_hubert_features)
    monkeypatch.setattr(inf, "load_hubert_model", lambda device, is_half=False: _FakeHubert())
    monkeypatch.setattr(rt, "load_hubert_model", lambda device, is_half=False: _FakeHubert())
    return fake_hubert_features


@pytest.fixture()
def tiny_feature_dir(tmp_path):
    import numpy as np

    d = tmp_path / "3_feature768"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    for i in range(3):
        np.save(d / f"{i}.npy", rng.standard_normal((40, 768)).astype(np.float32))
    return d
