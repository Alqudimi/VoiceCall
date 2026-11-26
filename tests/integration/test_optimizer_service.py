"""Optimizer integration tests — the real loop on the real engine.

Uses the project's own test stand-ins exactly like `test_engine.py`:
a REAL randomly-initialised v2/48k RVC synthesiser, a REAL small FAISS
index, the REAL Praat ``pm`` F0 extractor; only HuBERT is mocked
(deterministic stand-in, see tests/conftest.py) and the neural speaker
embedders are replaced by a deterministic stand-in (no 378 MB download
in CI). Everything else — analysis, scoring, search, session
persistence, results, apply — is the real optimizer stack.

Runtime discipline: one module-scoped optimization session is shared by
the "post-session" assertions (full-loop checks, apply, report, UI link)
so the whole suite performs ~3 sessions of real CPU conversions.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rvc_platform.optimizer.analysis import SR_ANALYSIS
from tests.conftest import speech_like


class _FakeEmbedderManager:
    """Deterministic neural-embedder stand-in (documented stand-in, like
    conftest's mock_hubert): embeddings are a smooth function of the signal
    spectrum, so real conversions produce different-but-comparable scores
    and identical inputs produce identical embeddings."""

    def __init__(self, device="cpu"):
        self.device = device
        self.status = type("S", (), {"wavlm": True, "ecapa": True, "dsp": False,
                                      "messages": {}, "neural_count": 2,
                                      "reliability": "high"})()

    def embeddings(self, wav16k):
        x = np.asarray(wav16k, dtype=np.float32)
        if x.size == 0:
            return {"wavlm": np.zeros(32, dtype=np.float32),
                    "ecapa": np.zeros(32, dtype=np.float32)}
        spec = np.abs(np.fft.rfft(x * 0.5))[:2048]
        e = np.zeros(32, dtype=np.float32)
        e[0] = float(spec.mean())
        e[1] = float(spec.std())
        for j in range(2, 32):
            e[j] = float(spec[(j - 2) * 70: (j - 1) * 70].mean())
        n = np.linalg.norm(e)
        e = e / n if n > 0 else e
        e2 = np.tanh(e) * 3.0
        e2 /= max(1e-9, np.linalg.norm(e2))
        return {"wavlm": e, "ecapa": e2}

    def cosine(self, a, b):
        d = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / d) if d > 0 else 0.0

    def similarity_report(self, wav16k, reference_embeddings):
        sims = {}
        cand = self.embeddings(wav16k)
        for name, refs in reference_embeddings.items():
            if name in cand and refs:
                sims[name] = float(np.mean(
                    [self.cosine(cand[name], np.asarray(r)) for r in refs]))
        neural = [sims[k] for k in ("wavlm", "ecapa") if k in sims]
        return {
            "similarities": sims,
            "ensemble": float(np.mean(neural)) if neural else None,
            "agreement_spread": (max(neural) - min(neural))
            if len(neural) >= 2 else 0.0,
            "reliability": "high",
        }


@pytest.fixture()
def optimizer_stack(project_env, mock_hubert, tmp_path):
    """Wired engine + models service + optimizer service, all real."""
    from rvc_platform.application.model_service import ModelService
    from rvc_platform.engine.rvc.inference import RVCInferenceEngine
    from rvc_platform.infrastructure.audio.pipeline import AudioProcessingEngine
    from rvc_platform.infrastructure.config.schema import (
        ConversionDefaults, EngineConfig, OptimizerConfig,
    )
    from rvc_platform.infrastructure.config import loader as cfg_loader
    from rvc_platform.infrastructure.persistence.model_repository import ModelRepository
    from rvc_platform.optimizer.analysis import AudioAnalyzer
    from rvc_platform.optimizer.service import OptimizerService
    from rvc_platform.utils.device import DeviceProfile

    profile = DeviceProfile("cpu", False, None, 0.0, 0.0)
    engine = RVCInferenceEngine(profile, AudioProcessingEngine())
    repo = ModelRepository(project_env / "models")
    models = ModelService(repo, engine)
    cfg_loader.reset_config()
    cfg = type("C", (), {
        "conversion": ConversionDefaults(),
        "engine": EngineConfig(f0_method="pm"),
        "optimizer": OptimizerConfig(),
    })()
    svc = OptimizerService(engine, models, cfg)
    svc._embedder = _FakeEmbedderManager("cpu")
    svc._analyzer = AudioAnalyzer(svc._embedder, device="cpu")
    return {"engine": engine, "models": models, "svc": svc, "cfg": cfg}


def _register_model_with_index(optimizer_stack, synthetic_model_file,
                               project_env):
    from rvc_platform.engine.rvc.index import build_index

    models = optimizer_stack["models"]
    models.register_upload(synthetic_model_file, "opt_model.pth")
    feats = project_env / "3_feature768"
    feats.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    np.save(feats / "a.npy", rng.standard_normal((30, 768)).astype(np.float32))
    added = build_index(feats, project_env / "run", "opt_model", "v2")
    models.repo.register_index(added, "opt_model")
    return models.get("opt_model")


def _make_ref_samples(tmp_path, n=2):
    paths = []
    for i in range(n):
        p = tmp_path / f"ref_{i}.wav"
        sf.write(str(p), speech_like(SR_ANALYSIS, 3.5, f0=150.0 + 6 * i,
                                     seed=20 + i),
                 SR_ANALYSIS, subtype="PCM_16")
        paths.append(p)
    return paths


def _make_input(tmp_path, seconds=7.0):
    """Speech with pauses so the analyzer can split search/validation."""
    wav = np.zeros(int(SR_ANALYSIS * seconds), dtype=np.float32)
    bursts = [
        (0.1, speech_like(SR_ANALYSIS, 2.2, f0=182.0, seed=40)),
        (3.0, speech_like(SR_ANALYSIS, 2.2, f0=178.0, seed=41)),
        (5.6, speech_like(SR_ANALYSIS, 1.2, f0=186.0, seed=42)),
    ]
    for start, seg in bursts:
        lo = int(start * SR_ANALYSIS)
        if lo >= wav.size:
            break
        end = min(lo + seg.size, wav.size)
        wav[lo:end] = seg[: end - lo]
    p = tmp_path / "input.wav"
    sf.write(str(p), wav, SR_ANALYSIS, subtype="PCM_16")
    return p


def _wait_phase(svc, phases, timeout_s=420.0):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        st = svc.status()
        last = st
        if st.get("phase") in phases:
            return st
        if not st.get("running") and st.get("phase") in ("failed",):
            break
        time.sleep(0.4)
    return last


@pytest.fixture(scope="module")
def completed_session(request):
    """One REAL full optimization loop, shared by the post-session tests.

    Module-scoped on purpose: real CPU conversions are expensive; the
    loop's artifacts (results, audio, applied settings) serve several
    assertions, exactly like a user's single optimization run serves the
    dashboard, the A/B panel and the apply button.
    """
    import shutil

    root = Path("/tmp/opt_it_session")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    os_env = {"RVC_PLATFORM_ROOT": str(root)}
    old = os.environ.get("RVC_PLATFORM_ROOT")
    os.environ.update(os_env)

    from rvc_platform.infrastructure.paths import reset_paths

    reset_paths()
    stack = _build_stack(root)
    info = _register_model_with_index(stack, _synthetic_model(root), root)
    svc = stack["svc"]
    for p in _make_ref_samples(root, n=2):
        svc.store_reference_sample("opt_model", "default", p.name,
                                   p.read_bytes())
    svc.build_profile("opt_model", "default")
    input_path = _make_input(root, seconds=7.0)
    stored = svc.store_input_audio("input.wav", input_path.read_bytes())
    started = svc.start({
        "model_name": "opt_model", "input_path": stored["path"],
        "mode": "quick", "max_trials": 4, "strategy": "builtin",
        "f0_methods": ["pm"], "pitch_policy": "preserve",
    })
    st = _wait_phase(svc, ("completed", "failed", "stopped"))
    yield {"stack": stack, "svc": svc, "session_id": started["session_id"],
           "phase": st["phase"], "engine": stack["engine"],
           "models": stack["models"], "input_path": input_path, "root": root}
    if old is None:
        os.environ.pop("RVC_PLATFORM_ROOT", None)
    else:
        os.environ["RVC_PLATFORM_ROOT"] = old
    reset_paths()


def _build_stack(root):
    from rvc_platform.application.model_service import ModelService
    from rvc_platform.engine.rvc.inference import RVCInferenceEngine
    from rvc_platform.infrastructure.audio.pipeline import AudioProcessingEngine
    from rvc_platform.infrastructure.config.schema import (
        ConversionDefaults, EngineConfig, OptimizerConfig,
    )
    from rvc_platform.infrastructure.persistence.model_repository import ModelRepository
    from rvc_platform.optimizer.analysis import AudioAnalyzer
    from rvc_platform.optimizer.service import OptimizerService
    from rvc_platform.utils.device import DeviceProfile

    import rvc_platform.engine.rvc.inference as inf
    from tests.conftest import fake_hubert_features

    inf.extract_hubert_features = fake_hubert_features
    inf.load_hubert_model = lambda d, h=False: None

    profile = DeviceProfile("cpu", False, None, 0.0, 0.0)
    engine = RVCInferenceEngine(profile, AudioProcessingEngine())
    repo = ModelRepository(root / "models")
    models = ModelService(repo, engine)
    cfg = type("C", (), {
        "conversion": ConversionDefaults(),
        "engine": EngineConfig(f0_method="pm"),
        "optimizer": OptimizerConfig(),
    })()
    svc = OptimizerService(engine, models, cfg)
    svc._embedder = _FakeEmbedderManager("cpu")
    svc._analyzer = AudioAnalyzer(svc._embedder, device="cpu")
    return {"engine": engine, "models": models, "svc": svc, "cfg": cfg}


def _synthetic_model(root):
    from rvc_platform.engine.rvc.checkpoint import export_inference_checkpoint
    from rvc_platform.engine.rvc.models import models as rvc_models
    from rvc_platform.training.run_config import load_template
    from rvc_platform.training.utils import HParams
    from tests.conftest import SYNTH_CONFIG_40K

    hubert_dir = root / "assets" / "hubert_base"
    hubert_dir.mkdir(parents=True, exist_ok=True)
    (hubert_dir / "config.json").write_text("{}", encoding="utf-8")
    (hubert_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    tpl = load_template("48k", "v2")
    hps = HParams.from_template(
        {"train": dict(tpl["train"]), "data": dict(tpl["data"]),
         "model": dict(tpl["model"])},
        version="v2", sample_rate=48_000, if_f0=1, exp_name="t",
        model_dir="", total_epoch=1, save_every_epoch=1,
        save_every_weights="1", if_latest=1, if_cache_data_in_gpu=False,
        pretrainG="", pretrainD="", gpus="0")
    net_g = rvc_models.SynthesizerTrnMs768NSFsid(
        *SYNTH_CONFIG_40K[:17], is_half=False, sr=48_000)
    out = root / "opt_model.pth"
    export_inference_checkpoint(net_g.state_dict(), hps, 3, "v2", out,
                                48_000, 1)
    return out



# --------------------------------------------------------------------------- #
# fast tests (no session)
# --------------------------------------------------------------------------- #

def test_capabilities_report_honest_state(optimizer_stack):
    caps = optimizer_stack["svc"].capabilities()
    assert "f0_methods" in caps and "pm" in caps["f0_methods"]
    assert caps["embedders"]["reliability"] in ("low", "medium", "high")
    assert set(caps["modes"]) == {"quick", "advanced", "deep"}
    assert "speaker_similarity" in caps["weights"]


def test_reference_sample_workflow(optimizer_stack, tmp_path):
    svc = optimizer_stack["svc"]
    refs = _make_ref_samples(tmp_path)
    meta = svc.store_reference_sample(
        "opt_model", "default", "ref_0.wav", refs[0].read_bytes())
    assert meta["id"]
    assert len(svc.list_reference_samples("opt_model", "default")) == 1
    with pytest.raises(ValueError):
        svc.store_reference_sample("opt_model", "default", "x.exe", b"MZ")
    with pytest.raises(ValueError):
        svc.store_reference_sample("opt_model", "default", "junk.wav",
                                   b"not an audio file")
    assert svc.delete_reference_sample("opt_model", "default", meta["id"])
    assert svc.list_reference_samples("opt_model", "default") == []


def test_build_profile_persists_with_reliability(optimizer_stack, tmp_path):
    svc = optimizer_stack["svc"]
    for p in _make_ref_samples(tmp_path, n=2):
        svc.store_reference_sample("opt_model", "p1", p.name, p.read_bytes())
    built = svc.build_profile("opt_model", "p1")
    assert built["n_samples"] == 2
    assert built["reliability"]["n_samples"] == 2
    assert any(x["name"] == "p1" for x in svc.list_profiles("opt_model"))
    # per-model isolation
    assert svc.list_profiles("other_model") == []
    with pytest.raises(ValueError):
        svc.build_profile("other_model", "p1")


def test_start_validates_inputs(optimizer_stack, tmp_path):
    svc = optimizer_stack["svc"]
    with pytest.raises(ValueError):
        svc.start({"model_name": "", "input_path": "/nope"})
    input_path = _make_input(tmp_path, 5.0)
    stored = svc.store_input_audio("i.wav", input_path.read_bytes())
    with pytest.raises(Exception):
        svc.start({"model_name": "ghost_model", "input_path": stored["path"]})


def test_controls_without_session_report_gracefully(optimizer_stack):
    svc = optimizer_stack["svc"]
    assert svc.pause()["paused"] is False
    r = svc.cancel()
    assert r["cancelled"] is False


# --------------------------------------------------------------------------- #
# the real loop (module-scoped session)
# --------------------------------------------------------------------------- #

def test_full_optimization_loop(completed_session):
    svc = completed_session["svc"]
    sid = completed_session["session_id"]
    assert completed_session["phase"] == "completed"

    results = svc.results(sid)
    assert results["session_id"] == sid
    # baseline + 4 trials recorded, none errored
    assert results["baseline"] is not None
    assert len(results["trials"]) == 5
    assert all(t["error"] is None for t in results["trials"])
    assert results["best"] is not None
    assert results["best"]["params"]["f0_method"] in ("rmvpe", "fcpe", "pm")
    assert 0.0 <= results["best"]["params"]["index_rate"] <= 1.0
    assert 0.0 <= results["best"]["params"]["protect"] <= 0.5
    # the eight axes are all scored
    for key in ("speaker_similarity", "timbre_similarity", "pitch_consistency",
                "naturalness", "clarity", "artifact_score",
                "prosody_preservation", "audio_quality"):
        assert key in results["best"]["axes"]
    # chart carries evolution data
    assert results["chart"]["numbers"]
    assert len(results["chart"]["composite"]) == len(results["chart"]["numbers"])
    # validation happened on held-out audio (bursted input)
    assert isinstance(results["validation"], list)
    # pareto computed
    assert isinstance(results["pareto"], list)
    # A/B audio artifacts exist
    audio_dir = svc.store.audio_dir(sid)
    assert (audio_dir / "best.wav").is_file()
    assert (audio_dir / "input_segment.wav").is_file()
    # audio path resolution is guarded
    assert svc.audio_path(sid, "best").is_file()
    with pytest.raises((ValueError, FileNotFoundError)):
        svc.audio_path(sid, "../state")


def test_engine_still_works_after_optimization(completed_session):
    """No regression: the shared engine still converts normally after the
    optimization session used it."""
    from rvc_platform.core.entities import ConversionSettings

    engine = completed_session["engine"]
    out = completed_session["root"] / "post_opt.wav"
    res = engine.convert_file(
        completed_session["input_path"], out,
        ConversionSettings(f0_method="pm"))
    assert res.output_path.exists()
    assert svc_idle_after_completion(completed_session)


def svc_idle_after_completion(completed_session):
    st = completed_session["svc"].status()
    return st["running"] is False


def test_apply_best_settings_and_convert_ui_link(completed_session):
    svc = completed_session["svc"]
    sid = completed_session["session_id"]
    applied = svc.apply_best(sid)
    assert applied["model"] == "opt_model"
    s = applied["settings"]
    assert 0.0 <= s["index_rate"] <= 1.0
    assert 0.0 <= s["protect"] <= 0.5
    assert s["f0_method"] in ("rmvpe", "fcpe", "pm")
    # the convert-UI link: applied settings retrievable per model
    fetched = svc.applied_settings("opt_model")
    assert fetched["settings"] == s
    # NOT applied to other models
    assert svc.applied_settings("other") is None
    # re-apply keeps previous restorable
    applied2 = svc.apply_best(sid)
    assert applied2["previous"] == s


def test_report_export_and_disclaimer(completed_session):
    svc = completed_session["svc"]
    sid = completed_session["session_id"]
    report = svc.export_report(sid)
    assert report["disclaimer"]
    assert "RELATIVE" in report["disclaimer"]
    assert report["results"]["session_id"] == sid
    assert (svc.store.session_dir(sid) / "report.json").is_file()


def test_profile_records_optimization_history(completed_session):
    from rvc_platform.optimizer.profile import ReferenceProfileStore

    store = ReferenceProfileStore()
    prof = store.load("opt_model", "default")
    assert prof is not None
    assert any(h.get("session_id") == completed_session["session_id"]
               for h in prof.history)


# --------------------------------------------------------------------------- #
# resume after restart (own short session)
# --------------------------------------------------------------------------- #

def test_session_persistence_and_resume_after_restart(tmp_path):
    import shutil

    root = Path("/tmp/opt_it_resume")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    old = os.environ.get("RVC_PLATFORM_ROOT")
    os.environ["RVC_PLATFORM_ROOT"] = str(root)
    try:
        from rvc_platform.infrastructure.paths import reset_paths

        reset_paths()
        stack = _build_stack(root)
        _register_model_with_index(stack, _synthetic_model(root), root)
        svc = stack["svc"]
        for p in _make_ref_samples(root, n=2):
            svc.store_reference_sample("opt_model", "default", p.name,
                                       p.read_bytes())
        svc.build_profile("opt_model", "default")
        input_path = _make_input(root, 6.0)
        stored = svc.store_input_audio("input.wav", input_path.read_bytes())

        svc.start({"model_name": "opt_model", "input_path": stored["path"],
                   "mode": "quick", "max_trials": 3, "strategy": "builtin",
                   "f0_methods": ["pm"]})
        sid = svc.store.list_sessions()[0]["session_id"]

        # simulate a process restart: a NEW service over the same disk state
        from rvc_platform.optimizer.analysis import AudioAnalyzer
        from rvc_platform.optimizer.service import OptimizerService

        svc2 = OptimizerService(stack["engine"], stack["models"], stack["cfg"])
        # after a restart the same embedder environment is rebuilt (the
        # stand-in mirrors the rebuilt real backends: same dims as build)
        svc2._embedder = _FakeEmbedderManager("cpu")
        svc2._analyzer = AudioAnalyzer(svc2._embedder, device="cpu")
        state = svc2.store.load(sid)
        assert state is not None
        if state.phase != "completed":
            r = svc2.resume_saved()
            assert r["resumed"] is True and r["session_id"] == sid
            st = _wait_phase(svc2, ("completed", "failed", "stopped"))
            assert st["phase"] == "completed"
            results = svc2.results(sid)
            assert len(results["trials"]) >= 2  # baseline + trials
            assert results["best"] is not None
        else:
            assert svc2.resume_saved()["resumed"] is False
    finally:
        if old is None:
            os.environ.pop("RVC_PLATFORM_ROOT", None)
        else:
            os.environ["RVC_PLATFORM_ROOT"] = old
        from rvc_platform.infrastructure.paths import reset_paths as rp

        rp()


def test_engine_model_guard_fails_trials_on_model_swap(tmp_path):
    """If the engine's loaded model changes mid-search, trials must fail
    honestly instead of scoring another model's output as this model's."""
    import shutil

    root = Path("/tmp/opt_it_swap")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    old = os.environ.get("RVC_PLATFORM_ROOT")
    os.environ["RVC_PLATFORM_ROOT"] = str(root)
    try:
        from rvc_platform.infrastructure.paths import reset_paths

        reset_paths()
        stack = _build_stack(root)
        models = stack["models"]
        _register_model_with_index(stack, _synthetic_model(root), root)
        svc = stack["svc"]
        for p in _make_ref_samples(root, n=2):
            svc.store_reference_sample("opt_model", "default", p.name,
                                       p.read_bytes())
        svc.build_profile("opt_model", "default")
        input_path = _make_input(root, 5.5)
        stored = svc.store_input_audio("input.wav", input_path.read_bytes())

        # a second model to swap in mid-run
        second = root / "second_model.pth"
        shutil.copyfile(root / "opt_model.pth", second)
        models.register_upload(second, "second_model.pth")

        svc.start({"model_name": "opt_model", "input_path": stored["path"],
                   "mode": "quick", "max_trials": 3, "strategy": "builtin",
                   "f0_methods": ["pm"]})
        swapped = False
        for _ in range(400):
            st = svc.status()
            if st.get("running"):
                try:
                    models.load("second_model")
                    swapped = True
                    break
                except Exception:
                    pass
            time.sleep(0.05)
        assert swapped, "session did not start in time"
        st = _wait_phase(svc, ("completed", "failed", "stopped"))
        assert st["phase"] in ("failed", "completed")
        if st["phase"] == "completed":
            results = svc.results()
            errored = [t for t in results["trials"] if t["error"]]
            assert errored, "model swap must never be scored silently"
        # restore for cleanliness
        models.load("opt_model")
    finally:
        if old is None:
            os.environ.pop("RVC_PLATFORM_ROOT", None)
        else:
            os.environ["RVC_PLATFORM_ROOT"] = old
        from rvc_platform.infrastructure.paths import reset_paths as rp

        rp()
