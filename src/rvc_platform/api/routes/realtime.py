"""Real-time streaming: REST info + the WebSocket endpoint."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets

import numpy as np
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from rvc_platform.core.errors import RealtimeError
from rvc_platform.engine.rvc.speaker_guard import build_speaker_profile


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/realtime", tags=["realtime"])


def _svc(request: Request):
    return request.app.state.services


@router.get("/info")
def info(request: Request):
    svc = _svc(request)
    loaded = svc["models"].loaded()
    cfg = svc["config"].realtime
    # the model the UI should preselect: the env-driven HuggingFace default
    # (e.g. eng7mi/Iraqi-v3) when present, otherwise the engine-loaded model.
    default_model = None
    recommended = None
    hf = svc.get("huggingface")
    if hf is not None:
        hf_status = hf.status()
        default_model = hf_status["default_model"]
        recommended = hf_status["recommended"]
    return {
        "available": loaded is not None and loaded.f0,
        "block_ms": cfg.block_ms,
        "crossfade_ms": cfg.crossfade_ms,
        "lookback_ms": cfg.lookback_ms,
        "silence_threshold_db": cfg.silence_threshold_db,
        "f0_method": cfg.f0_method,
        "loaded_model": loaded.to_dict() if loaded else None,
        "default_model": default_model,
        "recommended": recommended,
        "note": "protocol: TEXT control frames + BINARY int16 mono 16kHz in / tgt-sr out",
    }


class EnrollRequest(BaseModel):
    pcm_b64: str


@router.post("/enroll")
def enroll_speaker(req: EnrollRequest):
    try:
        raw = base64.b64decode(req.pcm_b64)
        wav = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        profile = build_speaker_profile(wav)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"تعذّر تحليل بصمة الصوت: {exc}") from exc
    return {
        "ok": True,
        "profile": profile.to_dict(),
        "f0_hz": round(profile.f0_hz, 1),
        "ref_db": round(profile.ref_db, 1),
    }


@router.get("/stats")
def stats(request: Request):
    svc = _svc(request)
    agg = svc["realtime"].aggregate_latency()
    sessions = [
        s.stats() for s in list(svc["realtime"].sessions.values())
    ]
    return {"aggregate": agg, "sessions": sessions}


@router.websocket("/ws")
async def realtime_ws(ws: WebSocket) -> None:
    svc = ws.app.state.services
    auth = svc["auth"]
    token_param = ws.query_params.get("token")
    # browsers cannot set Authorization headers on WebSocket: the token rides
    # the query string; comparison is constant-time either way.
    if auth.enabled and (not token_param
                         or not secrets.compare_digest(token_param, auth._token)):
        await ws.close(code=4401, reason="unauthorized")
        return
    await ws.accept()
    session = None
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if "text" in message and message["text"] is not None:
                try:
                    control = json.loads(message["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "bad json"})
                    continue
                kind = control.get("type")
                if kind == "start":
                    if session is not None:
                        svc["realtime"].close(session)
                    # optional per-session block size (60..500 ms) — lets the
                    # client trade latency against GPU headroom without a
                    # server restart; validated inside new_session().
                    requested_block_ms = control.get("block_ms")
                    try:
                        session = svc["realtime"].new_session(requested_block_ms)
                    except RealtimeError as exc:
                        session = None
                        await ws.send_json({"type": "error", "message": str(exc)})
                        continue
                    try:
                        hello = session.load(control.get("model"), control.get("settings") or {})
                        await ws.send_json({"type": "started", **hello})
                    except Exception as exc:
                        svc["realtime"].close(session)
                        session = None
                        await ws.send_json({"type": "error", "message": str(exc)})
                elif kind == "settings" and session is not None:
                    session.update(control.get("settings") or {})
                    await ws.send_json({"type": "settings_applied"})
                elif kind == "rtt" and session is not None:
                    # client-measured round-trip of a played block (ms) — feeds
                    # client_rtt_p95_ms in /stats (was never populated before).
                    try:
                        rtt_ms = float(control.get("rtt_ms"))
                    except (TypeError, ValueError):
                        rtt_ms = None
                    if rtt_ms is not None and rtt_ms > 0:
                        session.client_reported_rtt_ms.append(rtt_ms)
                        del session.client_reported_rtt_ms[:-20]
                elif kind == "ping":
                    await ws.send_json({
                        "type": "pong",
                        "blocks": session.blocks if session else 0,
                        "guard_status": session.engine.speaker_guard.last_reason if session else "silence",
                        "speaker_locked": (session.engine.speaker_guard.profile is not None) if session else False,
                    })
                elif kind == "stats":
                    await ws.send_json({"type": "stats", **(session.stats() if session else {})})
                else:
                    await ws.send_json({"type": "error", "message": f"unknown control {kind}"})
            elif "bytes" in message and message["bytes"] is not None:
                if session is None:
                    await ws.send_json({"type": "error", "message": "send {type:start} first"})
                    continue
                data = message["bytes"]
                # inference runs in a worker thread; the loop stays responsive
                out = await asyncio.to_thread(session.process, data)
                await ws.send_bytes(out)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - report then clean up
        logger.exception("realtime ws failed")
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if session is not None:
            svc["realtime"].close(session)
