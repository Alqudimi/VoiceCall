"""Real-time WebSocket session integration test (CPU, mock HuBERT only)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def rt_client(project_env, mock_hubert, synthetic_model_file):
    from rvc_platform.api.app import create_app
    from rvc_platform.infrastructure.paths import get_paths

    # register the synthetic f0 model so the session can load it by name
    models_dir = get_paths().models
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / synthetic_model_file.name).write_bytes(synthetic_model_file.read_bytes())

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client, synthetic_model_file.stem


def test_realtime_ws_full_session(rt_client):
    client, model_name = rt_client
    info = client.get("/api/realtime/info").json()
    assert info["block_ms"] == 160
    block_ms = info["block_ms"]

    with client.websocket_connect("/api/realtime/ws") as ws:
        ws.send_text(json.dumps({
            "type": "start",
            "model": model_name,
            "settings": {"f0_method": "pm", "pitch_shift": 0, "index_rate": 0.0},
        }))
        started = json.loads(ws.receive_text())
        assert started["type"] == "started"
        assert started["sample_rate"] == 48_000
        assert started["block_ms"] == block_ms

        # stream 8 synthetic blocks
        samples_per_block = block_ms * 16          # 16 kHz capture
        out_per_block = (block_ms // 10) * (started["sample_rate"] // 100)  # 10 ms grid
        rng = np.random.default_rng(3)
        for i in range(8):
            t = (np.arange(samples_per_block) + i * samples_per_block) / 16_000
            wave = 0.2 * np.sin(2 * np.pi * 180 * t) + 0.01 * rng.standard_normal(t.size)
            payload = (wave * 32767).astype(np.int16).tobytes()
            ws.send_bytes(payload)
            out = ws.receive_bytes()
            assert len(out) == out_per_block * 2, "int16 mono at model sr"

        # control frames
        ws.send_text(json.dumps({"type": "settings", "settings": {"pitch_shift": 3}}))
        reply = json.loads(ws.receive_text())
        assert reply["type"] == "settings_applied"
        ws.send_text(json.dumps({"type": "stats"}))
        stats = json.loads(ws.receive_text())
        assert stats["type"] == "stats"
        assert stats["blocks"] >= 8
        assert stats["server"]["count"] >= 8
        assert stats["server"]["avg_ms"] > 0

    # aggregate REST stats after the session closed
    rest = client.get("/api/realtime/stats").json()
    assert "aggregate" in rest and rest["sessions"] == []
