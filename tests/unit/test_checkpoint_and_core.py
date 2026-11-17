"""Checkpointing + entity + security + eventbus unit tests."""
from __future__ import annotations

import torch
from torch import nn

from rvc_platform.core.entities import ConversionSettings, LatencyRecord, LatencyStats
from rvc_platform.core.events import EventBus
from rvc_platform.infrastructure.security.auth import TokenAuth
from rvc_platform.training import checkpointing


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.bias = nn.Parameter(torch.zeros(2))


def test_full_checkpoint_roundtrip(tmp_path):
    model = Tiny()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    for _ in range(3):
        optim.step()
        sched.step()
    before_sched_lrs = [g["lr"] for g in optim.param_groups]
    path = tmp_path / "G_100.pth"
    checkpointing.save_full(path, model, optim, sched, None, epoch=7, global_step=100)

    model2 = Tiny()
    optim2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.ExponentialLR(optim2, gamma=0.9)
    meta = checkpointing.load_full(path, model2, optim2, sched2)
    assert meta["epoch"] == 7 and meta["global_step"] == 100
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
    assert [g["lr"] for g in optim2.param_groups] == before_sched_lrs


def test_legacy_epoch_probe(tmp_path):
    payload = {"model": {}, "optimizer": {}, "learning_rate": 1e-4, "iteration": 42}
    path = tmp_path / "G_2333333.pth"
    torch.save(payload, path)
    assert checkpointing.legacy_iteration_epoch(path) == 42


def test_latest_step_checkpoint_prefers_highest(tmp_path):
    for name in ("G_50.pth", "G_200.pth", "G_2333333.pth", "G_100.pth"):
        torch.save({"model": {}, "iteration": 1}, tmp_path / name)
    assert checkpointing.latest_step_checkpoint(tmp_path).name == "G_200.pth"


def test_state_json_roundtrip(tmp_path):
    state = {"epoch": 3, "global_step": 90, "dataset_signature": {"files": 5}}
    path = tmp_path / "state.json"
    checkpointing.write_state(state, path)
    assert checkpointing.read_state(path) == state


def test_latency_stats_percentiles():
    records = [
        LatencyRecord(captured_at=0, inference_ms=1, total_ms=float(i + 1))
        for i in range(20)
    ]
    stats = LatencyStats.from_records(records)
    assert stats.count == 20
    assert stats.min_ms == 1.0
    assert stats.max_ms == 20.0
    assert stats.avg_ms == 10.5
    assert stats.p95_ms >= 19  # 95th percentile near the top


def test_conversion_settings_validation():
    ConversionSettings().validate()
    for bad in (
        {"pitch_shift": 30},
        {"index_rate": 2.0},
        {"protect": 0.7},
        {"rms_mix_rate": -1},
        {"f0_method": "crepe"},
    ):
        try:
            ConversionSettings(**bad).validate()
            raise AssertionError(f"should have failed: {bad}")
        except ValueError:
            pass


def test_token_auth_compare():
    auth = TokenAuth("secret-token", enabled=True)
    assert auth.check_header("Bearer secret-token")
    assert not auth.check_header("Bearer wrong")
    assert not auth.check_header(None)
    disabled = TokenAuth("", enabled=False)
    assert disabled.check_header(None)


def test_eventbus_isolation_and_history():
    bus = EventBus(history=8)
    seen = []
    bus.subscribe("topic", lambda e: seen.append(e.payload["v"]))
    bus.subscribe("topic", lambda e: 1 / 0)  # faulty subscriber must not break publish
    bus.publish("topic", {"v": 1})
    bus.publish("other", {"v": 2})
    assert seen == [1]
    assert len(bus.recent("topic")) == 1
    for _ in range(10):
        bus.publish("topic", {"v": 0})
    assert len(bus.recent()) == 8  # bounded history
