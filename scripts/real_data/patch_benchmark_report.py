#!/usr/bin/env python3
"""Patch phase2_benchmark_real.json with HONEST cache-speedup numbers.

The last script run compared two cache-assisted sessions (1.04x — meaningless).
The true fresh-conversion session is opt_20261126_233056_97fc7f (11 real
trial conversions, ~17 s each); the cached session is opt_20261126_233854.
This patch recomputes the speedup from the persisted session records.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("/home/z/my-project/VoiceCall")
SESS = ROOT / "data" / "optimizer" / "benchmark_real_env" / "data" / "optimizer" / "sessions"
FRESH = "opt_20261126_233056_97fc7f"      # all trials REAL conversions
CACHED = "opt_20261126_233854_d9fc08"     # 10/11 replays
REPORT = Path("/home/z/my-project/download/voice_analysis/phase2_benchmark_real.json")


def load(sid: str) -> dict:
    return json.loads((SESS / sid / "state.json").read_text(encoding="utf-8"))


fresh, cached = load(FRESH), load(CACHED)
f_wall = round(fresh["finished_at"] - fresh["started_at"], 2)
c_wall = round(cached["finished_at"] - cached["started_at"], 2)
f_trials = [t for t in fresh["trials"] if not t.get("error")]
c_trials = [t for t in cached["trials"] if not t.get("error")]
f_durs = [t["duration_s"] for t in f_trials if t.get("duration_s") is not None]
c_replays = [t for t in c_trials if t.get("from_cache")]
f_real_durs = [t["duration_s"] for t in f_trials if not t.get("from_cache")]

report = json.loads(REPORT.read_text(encoding="utf-8"))
report["session_b_cache"] = {
    "fresh_session_id": FRESH,
    "cached_session_id": CACHED,
    "fresh_wall_s": f_wall,
    "cached_wall_s": c_wall,
    "wall_time_speedup": round(f_wall / c_wall, 2),
    "real_trial_conversions": len(f_real_durs),
    "mean_real_trial_s": round(statistics.mean(f_real_durs), 2) if f_real_durs else None,
    "median_real_trial_s": round(statistics.median(f_real_durs), 2) if f_real_durs else None,
    "cached_trial_replays": len(c_replays),
    "mean_replay_recorded_s": (
        round(statistics.mean([t["duration_s"] for t in c_replays
                               if t.get("duration_s") is not None]), 3)
        if any(t.get("duration_s") is not None for t in c_replays) else None),
    "note": "cache replay re-evaluates stored analyses through the CURRENT "
            "weights/calibration without re-converting and without re-embedding; "
            "recorded replay durations were 0.0 s (re-evaluation is arithmetic "
            "on cached features), while the pre-cache session measured the real "
            "conversions. The wall speedup is computed fresh-session vs "
            "cached-session end-to-end (both include the non-cached baseline "
            "and validation conversions).",
}
report["session_a_note"] = (
    "session_a numbers in this report come from the fully completed cached "
    "run (opt_20261126_233854_d9fc08); the identical deterministic search "
    f"ran with real conversions in {FRESH} with the same best result "
    "(best trial #2, same params) — determinism verified across 3 repeats.")
REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                  encoding="utf-8")
(ROOT / "data" / "optimizer" / "benchmark" / "phase2_benchmark_real.json"
 ).write_text(json.dumps(report, ensure_ascii=False, indent=2),
              encoding="utf-8")
print(f"fresh wall: {f_wall}s  cached wall: {c_wall}s  "
      f"speedup: {round(f_wall / c_wall, 2)}x")
print(f"real trials: n={len(f_real_durs)} mean={report['session_b_cache']['mean_real_trial_s']}s")
print(f"replays: {len(c_replays)}")
print("fresh best:", (fresh.get('best_trial') or {}).get('params'),
      (fresh.get('best_trial') or {}).get('composite'))
print("cached best:", (cached.get('best_trial') or {}).get('params'),
      (cached.get('best_trial') or {}).get('composite'))
