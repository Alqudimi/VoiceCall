"""Lightweight in-process pub/sub used between engine threads and services.

Training subprocesses publish JSON lines; services translate them into these
typed events for the API layer (SSE / WebSocket / polling endpoints).
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


@dataclass
class Event:
    topic: str            # "training.progress" | "training.log" | "conversion.progress" ...
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def json(self) -> str:
        return json.dumps({"topic": self.topic, "ts": self.ts, "payload": self.payload},
                          ensure_ascii=False, default=str)


class EventBus:
    """Thread-safe fan-out bus with a bounded history for late subscribers."""

    def __init__(self, history: int = 256) -> None:
        self._subs: dict[str, list[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()
        self._history: deque[Event] = deque(maxlen=history)

    def subscribe(self, topic: str, cb: Callable[[Event], None]) -> Callable[[], None]:
        with self._lock:
            self._subs.setdefault(topic, []).append(cb)
        def _unsub() -> None:
            with self._lock:
                try:
                    self._subs[topic].remove(cb)
                except (KeyError, ValueError):
                    pass
        return _unsub

    def publish(self, topic: str, payload: dict[str, Any]) -> Event:
        evt = Event(topic=topic, payload=payload)
        with self._lock:
            self._history.append(evt)
            handlers = list(self._subs.get(topic, ())) + list(self._subs.get("*", ()))
        for h in handlers:
            try:
                h(evt)
            except Exception:  # a faulty subscriber must not break the publisher
                pass
        return evt

    def recent(self, topic: str | None = None, limit: int = 32) -> list[Event]:
        with self._lock:
            items = list(self._history)
        if topic:
            items = [e for e in items if e.topic == topic]
        return items[-limit:]


_GLOBAL = EventBus()


def global_bus() -> EventBus:
    return _GLOBAL
