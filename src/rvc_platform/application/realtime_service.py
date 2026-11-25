"""Real-time conversion service: WebSocket session management.

Transport contract (documented in docs/REALTIME.md):

  client -> server TEXT   {"type":"start","model":"<name>","settings":{...}}
                            {"type":"settings","pitch_shift":..,"index_rate":..}
                            {"type":"ping"}
  client -> server BINARY  Int16 mono PCM @ 16 kHz, exactly block_ms chunks
  server -> client TEXT    {"type":"started","sample_rate":<tgt>,
                            "block_ms":<n>, "channels":1, "format":"int16"}
                            {"type":"stats","latency":{...},"blocks":n}
                            {"type":"error","message":".."}
  server -> client BINARY  Int16 mono PCM @ model sample rate (block-aligned)

Each connection owns a dedicated RealTimeRVC engine; a server-side
asyncio->thread bridge runs the torch inference off the event loop.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

import numpy as np

from rvc_platform.core.entities import LatencyStats
from rvc_platform.core.errors import RealtimeError
from rvc_platform.engine.rvc.realtime import RealTimeRVC
from rvc_platform.infrastructure.config.loader import get_config
from rvc_platform.infrastructure.persistence.model_repository import ModelRepository
from rvc_platform.utils.device import DeviceProfile

logger = logging.getLogger(__name__)


class RealtimeSession:
    def __init__(self, session_id: str, profile: DeviceProfile,
                 realtime_cfg, repo: ModelRepository) -> None:
        self.id = session_id
        self.cfg = realtime_cfg
        self.repo = repo
        self.engine = RealTimeRVC(
            profile,
            block_ms=realtime_cfg.block_ms,
            crossfade_ms=realtime_cfg.crossfade_ms,
            extra_ms=realtime_cfg.lookback_ms,
            silence_threshold_db=realtime_cfg.silence_threshold_db,
            f0_method=realtime_cfg.f0_method,   # config-driven server default
        )
        self.created_at = time.time()
        self.last_seen = time.time()
        self.blocks = 0
        self.client_reported_rtt_ms: list[float] = []
        self._transport_overhead_ms = 0.0

    def idle_seconds(self) -> float:
        return max(0.0, time.time() - self.last_seen)

    def load(self, model_name: str, settings: dict[str, Any]) -> dict:
        info = self.repo.get(model_name)
        if not info.f0:
            raise RealtimeError(
                "real-time conversion requires a pitch-guided (f0) model"
            )
        # apply f0 method before load so the warm-up pass uses it
        f0_method = (settings or {}).get("f0_method")
        if f0_method:
            self.engine.f0_method = str(f0_method)
        self.engine.load_model(info.path, info.index_path)
        self.engine.strict_isolation = True
        if settings:
            self.engine.update_settings(**self._filter_settings(settings))
        self.engine.warmup()
        return {
            "sample_rate": self.engine.tgt_sr,
            "block_ms": self.cfg.block_ms,
            "channels": 1,
            "format": "int16",
            "model": info.name,
            "latency_budget_ms": self.cfg.block_ms,
            "speaker_locked": self.engine.speaker_guard.profile is not None,
        }

    @staticmethod
    def _filter_settings(settings: dict[str, Any]) -> dict[str, Any]:
        allowed = {"pitch_shift", "index_rate", "f0_method", "rms_mix_rate",
                   "silence_threshold_db", "strict_isolation"}
        filtered = {k: v for k, v in settings.items() if k in allowed and v is not None}
        if "speaker_profile" in settings:
            filtered["speaker_profile"] = settings["speaker_profile"]
        return filtered

    def update(self, settings: dict[str, Any]) -> None:
        self.engine.update_settings(**self._filter_settings(settings))

    def process(self, pcm_int16: bytes) -> bytes:
        block = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        t0 = time.perf_counter()
        audio = self.engine.process_block(block)
        transport = (time.perf_counter() - t0) * 1000.0
        self._transport_overhead_ms = transport
        self.blocks += 1
        self.last_seen = time.time()
        out = np.clip(audio, -1.0, 1.0)
        return (out * 32767.0).astype(np.int16).tobytes()

    def stats(self) -> dict:
        engine_stats = self.engine.latency()
        # true wire latency = engine inference + block transport bridge
        return {
            "session_id": self.id,
            "blocks": self.blocks,
            "server": {
                "count": engine_stats.count,
                "min_ms": engine_stats.min_ms,
                "avg_ms": engine_stats.avg_ms,
                "p95_ms": engine_stats.p95_ms,
                "max_ms": engine_stats.max_ms,
            },
            "bridge_overhead_ms": round(self._transport_overhead_ms, 2),
            "block_ms": self.cfg.block_ms,
            "client_rtt_p95_ms": (
                sorted(self.client_reported_rtt_ms)[
                    min(len(self.client_reported_rtt_ms) - 1,
                        int(0.95 * (len(self.client_reported_rtt_ms) - 1)))
                ] if self.client_reported_rtt_ms else None
            ),
            "engine_status": self.engine.status,
            "guard_status": self.engine.speaker_guard.last_reason,
            "speaker_locked": self.engine.speaker_guard.profile is not None,
        }


class RealtimeService:
    def __init__(self, profile: DeviceProfile, repo: ModelRepository) -> None:
        self.profile = profile
        self.repo = repo
        self.cfg = get_config().realtime
        self.sessions: dict[str, RealtimeSession] = {}
        self._lock = threading.Lock()
        self._reaper_started = False

    def new_session(self, block_ms: int | None = None) -> RealtimeSession:
        # VRAM guard: every session holds its own engine + model copy — cap them
        realtime_cfg = self.cfg
        if block_ms is not None:
            if not (60 <= int(block_ms) <= 500):
                raise RealtimeError("block_ms must be between 60 and 500")
            # per-session geometry override (client picks latency vs. headroom)
            realtime_cfg = self.cfg.model_copy(update={"block_ms": int(block_ms)})
        with self._lock:
            self._drop_idle_locked()
            live = sum(1 for s in self.sessions.values())
            if live >= self.cfg.max_sessions:
                raise RealtimeError(
                    f"too many realtime sessions ({live}/{self.cfg.max_sessions}); "
                    "close one before opening another"
                )
        session = RealtimeSession(
            f"rt_{uuid.uuid4().hex[:8]}", self.profile, realtime_cfg, self.repo
        )
        with self._lock:
            self.sessions[session.id] = session
        return session

    def _drop_idle_locked(self) -> None:
        """Close sessions silent past keep_alive_s (abandoned mics, dead tabs)."""
        for sid in list(self.sessions):
            session = self.sessions.get(sid)
            if session is not None and session.idle_seconds() > self.cfg.keep_alive_s:
                logger.info("dropping idle realtime session %s (%.0fs silent)",
                            sid, session.idle_seconds())
                self.sessions.pop(sid, None)

    def close(self, session: RealtimeSession) -> None:
        with self._lock:
            self.sessions.pop(session.id, None)

    def active(self) -> int:
        with self._lock:
            return sum(1 for s in self.sessions.values() if s.blocks >= 0)

    def aggregate_latency(self) -> dict:
        with self._lock:
            sessions = list(self.sessions.values())
        if not sessions:
            return LatencyStats().to_dict() if hasattr(LatencyStats, "to_dict") else {}
        merged = [r for s in sessions for r in list(s.engine._records)]
        # LatencyStats is a frozen summary; rebuild via from_records
        stats = LatencyStats.from_records(merged)
        return {
            "count": stats.count, "min_ms": stats.min_ms, "avg_ms": stats.avg_ms,
            "p95_ms": stats.p95_ms, "max_ms": stats.max_ms,
            "sessions": len(sessions),
        }
