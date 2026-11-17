"""Real-time streaming: REST info + the WebSocket endpoint."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from rvc_platform.core.errors import RealtimeError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/realtime", tags=["realtime"])


def _svc(request: Request):
    return request.app.state.services


@router.get("/info")
def info(request: Request):
    svc = _svc(request)
    loaded = svc["models"].loaded()
    cfg = svc["config"].realtime
    return {
        "available": loaded is not None and loaded.f0,
        "block_ms": cfg.block_ms,
        "crossfade_ms": cfg.crossfade_ms,
        "lookback_ms": cfg.lookback_ms,
        "silence_threshold_db": cfg.silence_threshold_db,
        "f0_method": cfg.f0_method,
        "loaded_model": loaded.to_dict() if loaded else None,
        "note": "protocol: TEXT control frames + BINARY int16 mono 16kHz in / tgt-sr out",
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
    if auth.enabled and (not token_param or token_param != auth._token):
        await ws.close(code=4401, reason="unauthorized")
        return
    await ws.accept()
    session = None
    loop = asyncio.get_running_loop()
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
                    session = svc["realtime"].new_session()
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
                elif kind == "ping":
                    await ws.send_json({"type": "pong", "blocks": session.blocks if session else 0})
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
