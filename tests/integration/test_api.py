"""API integration tests via FastAPI TestClient (real service graph)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient



@pytest.fixture()
def client(project_env):
    from rvc_platform.api.app import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth_client(project_env, monkeypatch):
    monkeypatch.setenv("RVC_AUTH_ENABLED", "1")
    from rvc_platform.infrastructure.config.loader import reset_config

    reset_config()
    from rvc_platform.api.app import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_system(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200 and resp.json()["ok"] is True
    resp = client.get("/api/system")
    body = resp.json()
    assert body["app_version"] == "1.0.0"
    assert body["device"] == "cpu"
    assert body["engine"] in {"unavailable", "error", "ready"}
    assert "gpu" in body and "ram" in body and "training" in body


def test_models_crud_flow(client, project_env, synthetic_model_file):
    assert client.get("/api/models").json() == []
    # upload the real synthetic checkpoint through the API
    with open(synthetic_model_file, "rb") as fh:
        resp = client.post(
            "/api/models/upload",
            files={"file": ("synthetic_v2_40k.pth", fh, "application/octet-stream")},
        )
    assert resp.status_code == 200
    registered = resp.json()["registered"]
    assert registered["sample_rate"] == 40_000

    listing = client.get("/api/models").json()
    assert len(listing) == 1 and listing[0]["f0"] is True

    info = client.get(f"/api/models/{registered['name']}")
    assert info.json()["engine_loaded"] is False

    # engine load via API (CPU; mock-free: only checkpoint construction)
    resp = client.post(f"/api/models/{registered['name']}/load")
    assert resp.status_code == 200
    system = client.get("/api/system").json()
    assert system["engine"] == "ready"
    assert system["loaded_model"]["name"] == registered["name"]

    resp = client.post("/api/models/unload")
    assert resp.status_code == 200

    resp = client.delete(f"/api/models/{registered['name']}")
    assert resp.status_code == 200
    assert client.get("/api/models").json() == []


def test_model_upload_rejects_junk(client):
    resp = client.post(
        "/api/models/upload",
        files={"file": ("evil.pth", b"not-a-torch-checkpoint", "application/octet-stream")},
    )
    assert resp.status_code == 422


def test_model_delete_missing_returns_404(client):
    assert client.delete("/api/models/nope.pth").status_code == 404


def test_convert_settings_endpoint(client):
    body = client.get("/api/convert/settings").json()
    assert body["f0_methods"] == ["rmvpe", "fcpe", "pm"]
    assert body["engine"]["status"] == "unavailable"
    assert body["defaults"]["protect"] == pytest.approx(0.33)


def test_convert_requires_loaded_model(client, speech_wav_file):
    with open(speech_wav_file, "rb") as fh:
        resp = client.post(
            "/api/convert",
            files={"file": ("speech.wav", fh, "audio/wav")},
            data={"pitch_shift": "2", "f0_method": "pm"},
        )
    assert resp.status_code == 409  # honest failure: no model loaded


def test_convert_rejects_bad_upload(client):
    resp = client.post(
        "/api/convert",
        files={"file": ("song.exe", b"RIFF-notaudio", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_training_status_idle(client):
    body = client.get("/api/training/status").json()
    assert body["state"] == "idle"
    assert client.get("/api/training/runs").json() == []


def test_training_checkpoints_404(client):
    assert client.get("/api/training/runs/ghost/checkpoints").status_code == 404


def test_tunnel_status_offline(client):
    body = client.get("/api/tunnel/status").json()
    assert body["running"] is False and body["url"] is None


def test_auth_required_when_enabled(auth_client, project_env):
    assert auth_client.get("/api/system").status_code == 401
    token = (project_env / "logs" / "auth.token").read_text().strip().splitlines()[-1]
    headers = {"Authorization": f"Bearer {token}"}
    assert auth_client.get("/api/system", headers=headers).status_code == 200
    # health stays public; wrong token rejected
    assert auth_client.get("/api/health").status_code == 200
    assert auth_client.get("/api/system", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_assets_endpoint_reports_missing(project_env, client):
    body = client.get("/api/assets").json()
    assert "_summary" in body
    assert body["_summary"]["hubert_ready"] is False  # no assets in temp env


def test_logs_endpoint(auth_client, project_env):
    token = (project_env / "logs" / "auth.token").read_text().strip().splitlines()[-1]
    resp = auth_client.get("/api/logs?lines=10", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200 and "lines" in resp.json()
