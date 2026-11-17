"""WS auth (constant-time path) + unhandled-error envelope via the API app."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def auth_client(project_env, monkeypatch):
    monkeypatch.setenv("RVC_AUTH_ENABLED", "1")
    monkeypatch.setenv("RVC_AUTH_TOKEN", "secret-token-123")
    from rvc_platform.infrastructure.config.loader import reset_config

    reset_config()
    from rvc_platform.api.app import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_ws_rejects_wrong_token(auth_client):
    with pytest.raises(Exception):
        with auth_client.websocket_connect("/api/realtime/ws?token=wrong-token"):
            pass


def test_ws_rejects_missing_token(auth_client):
    with pytest.raises(Exception):
        with auth_client.websocket_connect("/api/realtime/ws"):
            pass


def test_ws_accepts_correct_token(auth_client):
    with auth_client.websocket_connect(
        "/api/realtime/ws?token=secret-token-123"
    ) as ws:
        ws.send_json({"type": "ping"})
        reply = ws.receive_json()
        assert reply["type"] == "pong"


def test_unhandled_route_returns_json_envelope(project_env):
    """A route that raises unexpectedly must still answer with JSON (500)."""
    from rvc_platform.infrastructure.config.loader import reset_config

    reset_config()  # drop any auth leftover from previous tests
    from rvc_platform.api.app import create_app
    from fastapi import APIRouter

    app = create_app()
    boom = APIRouter()

    @boom.get("/api/boom")
    def _boom():
        raise RuntimeError("kaboom")

    app.include_router(boom)
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/boom")
        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"].startswith("internal error")
        assert body["path"] == "/api/boom"
