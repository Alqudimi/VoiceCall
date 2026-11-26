"""Optimizer API integration tests — the real FastAPI service graph.

Boots the REAL app factory (create_app — the composition root where the
optimizer service is wired) and exercises the optimizer routes end to
end: capabilities, reference upload, profile build, input upload, start,
live status, pause/cancel semantics, results, audio retrieval, apply and
report export. The real optimization loop is intentionally small and runs
on the same synthetic v2/48k checkpoint + mocked HuBERT + pm F0 setup as
the other integration tests; the neural embedders are replaced by a
deterministic stand-in (no network downloads in CI).
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from rvc_platform.optimizer.analysis import SR_ANALYSIS
from tests.conftest import speech_like


class _FakeEmbedderManager:
    def __init__(self, device="cpu"):
        self.device = device
        self.status = type("S", (), {"wavlm": True, "ecapa": True, "dsp": False,
                                      "messages": {}, "neural_count": 2,
                                      "reliability": "high"})()

    def embeddings(self, wav16k):
        x = np.asarray(wav16k, dtype=np.float32)
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
            sims[name] = float(np.mean(
                [self.cosine(cand[name], np.asarray(r)) for r in refs]))
        neural = [sims[k] for k in ("wavlm", "ecapa") if k in sims]
        return {"similarities": sims,
                "ensemble": float(np.mean(neural)) if neural else None,
                "agreement_spread": 0.02,
                "reliability": "high",
                "notes": []}


@pytest.fixture()
def opt_client(project_env, mock_hubert, tmp_path):
    from rvc_platform.api.app import create_app
    from rvc_platform.optimizer.analysis import AudioAnalyzer

    app = create_app()
    # replace the neural speaker embedders with the deterministic stand-in
    # (documented in tests/conftest.py style — no network downloads in CI)
    with TestClient(app) as client:
        optimizer = app.state.services["optimizer"]
        optimizer._embedder = _FakeEmbedderManager("cpu")
        optimizer._analyzer = AudioAnalyzer(optimizer._embedder, device="cpu")
        yield client


def _ref_wav_bytes(f0=150.0, seed=21):
    import io

    buf = io.BytesIO()
    sf.write(buf, speech_like(SR_ANALYSIS, 3.5, f0=f0, seed=seed),
             SR_ANALYSIS, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _input_wav_bytes():
    import io

    wav = np.zeros(int(SR_ANALYSIS * 6.5), dtype=np.float32)
    for start, f0, seed in ((0.1, 182.0, 40), (3.0, 178.0, 41)):
        seg = speech_like(SR_ANALYSIS, 2.2, f0=f0, seed=seed)
        lo = int(start * SR_ANALYSIS)
        wav[lo:lo + seg.size] = seg[:wav.size - lo]
    buf = io.BytesIO()
    sf.write(buf, wav, SR_ANALYSIS, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_optimizer_capabilities_and_status(opt_client):
    resp = opt_client.get("/api/optimizer/capabilities")
    assert resp.status_code == 200
    caps = resp.json()
    assert "pm" in caps["f0_methods"]
    assert caps["modes"]["quick"]["max_trials"] >= 4
    assert "speaker_similarity" in caps["weights"]

    resp = opt_client.get("/api/optimizer/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["running"] is False
    assert body["phase"] == "idle"
    assert "embedders" in body


def test_optimizer_reference_upload_rejects_bad_files(opt_client):
    # bad suffix
    resp = opt_client.post(
        "/api/optimizer/reference-samples",
        files={"file": ("x.exe", b"MZ", "application/octet-stream")},
        data={"model_name": "opt_model", "profile_name": "default"},
    )
    assert resp.status_code == 400
    # not audio
    resp = opt_client.post(
        "/api/optimizer/reference-samples",
        files={"file": ("junk.wav", b"definitely not wav", "audio/wav")},
        data={"model_name": "opt_model", "profile_name": "default"},
    )
    assert resp.status_code in (400, 422)


def test_optimizer_requires_profile_before_start(opt_client, tmp_path):
    resp = opt_client.post(
        "/api/optimizer/input",
        files={"file": ("input.wav", _input_wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 200
    input_path = resp.json()["stored"]["path"]

    resp = opt_client.post(
        "/api/optimizer/start",
        data={"model_name": "ghost", "input_path": input_path},
    )
    assert resp.status_code == 404  # model not found


def test_optimizer_full_api_loop(opt_client, synthetic_model_file, mock_hubert):
    # 1) register + load the model via the existing models API
    with open(synthetic_model_file, "rb") as fh:
        resp = opt_client.post(
            "/api/models/upload",
            files={"file": ("opt_model.pth", fh, "application/octet-stream")},
        )
    assert resp.status_code == 200
    model_name = resp.json()["registered"]["name"]
    resp = opt_client.post(f"/api/models/{model_name}/load")
    assert resp.status_code == 200

    # 2) reference samples -> profile build
    for i, f0 in enumerate((150.0, 156.0)):
        resp = opt_client.post(
            "/api/optimizer/reference-samples",
            files={"file": (f"ref_{i}.wav", _ref_wav_bytes(f0), "audio/wav")},
            data={"model_name": model_name, "profile_name": "default"},
        )
        assert resp.status_code == 200
        assert resp.json()["stored"]["id"]
    resp = opt_client.post(
        f"/api/optimizer/analyze-reference?model_name={model_name}"
        f"&profile_name=default")
    assert resp.status_code == 200
    assert resp.json()["n_samples"] == 2

    # profiles listing + samples listing
    resp = opt_client.get(f"/api/optimizer/profiles?model_name={model_name}")
    assert resp.status_code == 200
    assert any(p["name"] == "default" for p in resp.json()["profiles"])
    resp = opt_client.get(
        f"/api/optimizer/reference-samples?model_name={model_name}"
        "&profile_name=default")
    assert len(resp.json()["samples"]) == 2

    # 3) input audio upload
    resp = opt_client.post(
        "/api/optimizer/input",
        files={"file": ("input.wav", _input_wav_bytes(), "audio/wav")},
    )
    input_path = resp.json()["stored"]["path"]

    # 4) start a small real session
    resp = opt_client.post(
        "/api/optimizer/start",
        data={"model_name": model_name, "input_path": input_path,
              "mode": "quick", "max_trials": "3", "strategy": "builtin",
              "pitch_policy": "preserve",
              "f0_methods_json": '["pm"]', "weights_json": "{}"},
    )
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    # 5) live status polling until completion
    import time

    deadline = time.time() + 420
    last = {}
    while time.time() < deadline:
        st = opt_client.get("/api/optimizer/status").json()
        last = st
        if st["phase"] in ("completed", "failed", "stopped"):
            break
        time.sleep(0.4)
    assert last["phase"] == "completed", f"phase={last.get('phase')}"

    # pause/cancel semantics after completion
    assert opt_client.post("/api/optimizer/pause").json()["paused"] is False

    # 6) results + chart + audio + report
    resp = opt_client.get(f"/api/optimizer/results?session_id={sid}")
    assert resp.status_code == 200
    results = resp.json()
    assert results["best"] is not None
    assert results["chart"]["numbers"]
    assert results["reliability_notes"] is not None

    resp = opt_client.get(
        f"/api/optimizer/results/audio?session_id={sid}&name=best")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/")
    # traversal guarded
    resp = opt_client.get(
        f"/api/optimizer/results/audio?session_id={sid}&name=../state")
    assert resp.status_code in (400, 404)

    resp = opt_client.get(f"/api/optimizer/report?session_id={sid}")
    assert resp.status_code == 200
    assert "disclaimer" in resp.json()

    # 7) apply -> convert-settings link
    resp = opt_client.post(f"/api/optimizer/apply?session_id={sid}")
    assert resp.status_code == 200
    applied = resp.json()
    assert applied["model"] == model_name
    resp = opt_client.get(f"/api/optimizer/models/{model_name}/applied")
    assert resp.status_code == 200
    assert resp.json()["applied"]["settings"] == applied["settings"]

    # 8) the conversion settings endpoint now carries the optimizer link
    resp = opt_client.get("/api/convert/settings")
    assert resp.status_code == 200
    linked = resp.json()["optimizer_applied"]
    assert linked and linked["settings"]["f0_method"] in ("rmvpe", "fcpe", "pm")

    # 9) old endpoints unaffected
    assert opt_client.get("/api/system").status_code == 200
    assert opt_client.get("/api/convert/history").status_code == 200
