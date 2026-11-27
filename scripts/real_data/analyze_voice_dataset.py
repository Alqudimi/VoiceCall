#!/usr/bin/env python3
"""Deep quality analysis + target-speaker reference profile on REAL data.

Runs the project's own Phase-02 analysis stack over the imported dataset:

- AudioAnalyzer (VAD, F0, vibrato, spectral, formants, HNR, loudness, noise)
- grade_sample_quality (per-sample quality scoring)
- ReferenceProfile + ReferenceProfileStore (persisted speaker profile)

Outputs (all under data/ + download/):
- download/voice_analysis/iraqi_voice_per_file.json   (full per-file analyses)
- download/voice_analysis/iraqi_voice_quality.json    (compact dataset quality report)
- speaker profile persisted by the platform store
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make the project importable regardless of the CWD.
PROJECT = Path("/home/z/my-project/VoiceCall")
sys.path.insert(0, str(PROJECT / "src"))

from rvc_platform.optimizer.analysis import AudioAnalyzer  # noqa: E402
from rvc_platform.optimizer.embeddings import SpeakerEmbedderManager  # noqa: E402
from rvc_platform.optimizer.profile import (  # noqa: E402
    ReferenceProfile,
    ReferenceProfileStore,
    grade_sample_quality,
)

DATASET = PROJECT / "data" / "raw" / "iraqi_voice"
OUT_DIR = Path("/home/z/my-project/download/voice_analysis")
MODEL_NAME = "iraqi_voice"          # the model being trained on this data
PROFILE_NAME = "default"


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    embedder = SpeakerEmbedderManager("cpu", prefer="auto")
    print(f"[embedders] active backends: {sorted(embedder._embedders)}")
    analyzer = AudioAnalyzer(embedder, device="cpu")
    print(f"[f0] extractor: {type(analyzer._f0_extractor()).__name__}")

    files = sorted(p for p in DATASET.glob("*.wav"))
    if not files:
        print("no wav files found", file=sys.stderr)
        return 1
    print(f"[dataset] {len(files)} files, analyzing...")

    # ---- per-file checkpoint cache (keyed by active embedder backends) ----
    cache_dir = OUT_DIR / "per_file_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(name: str) -> Path:
        return cache_dir / f"{name}.json"

    def _backends_key() -> str:
        st = embedder.status          # triggers lazy build, no inference
        return ",".join(sorted(
            name for name, ok in (("ecapa", st.ecapa), ("wavlm", st.wavlm),
                                  ("dsp", st.dsp)) if ok))

    per_file: list[dict] = []
    profile = ReferenceProfile(MODEL_NAME, PROFILE_NAME)
    backends_key = _backends_key()
    print(f"[cache] backends key = {backends_key}")
    for i, path in enumerate(files, 1):
        cached_path = _cache_path(path.name)
        entry = None
        if cached_path.is_file():
            try:
                entry = json.loads(cached_path.read_text(encoding="utf-8"))
                if entry.get("backends_key") != backends_key:
                    entry = None            # embedders changed -> re-analyze
            except Exception:
                entry = None
        if entry is not None:
            res = entry["analysis"]
            quality = entry["quality"]
            per_file.append({"file": path.name, "analysis": res,
                             "quality": quality,
                             "analyze_seconds": entry.get("analyze_seconds", 0),
                             "from_cache": True})
        else:
            t1 = time.time()
            res = analyzer.analyze_file(path).to_dict()
            quality = grade_sample_quality(res)
            entry = {"backends_key": backends_key, "analysis": res,
                     "quality": quality,
                     "analyze_seconds": round(time.time() - t1, 2)}
            cached_path.write_text(
                json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            per_file.append({"file": path.name, "analysis": res,
                             "quality": quality,
                             "analyze_seconds": entry["analyze_seconds"]})
        profile.add_sample_analysis(
            entry["analysis"],
            {"id": path.stem, "file": path.name,
             "duration_s": entry["analysis"].get("duration_s")})
        mark = " (cached)" if per_file[-1].get("from_cache") else ""
        print(f"  [{i:2d}/{len(files)}] {path.name} "
              f"dur={entry['analysis'].get('duration_s')}s "
              f"grade={entry['quality'].get('grade')} "
              f"score={entry['quality'].get('score')} "
              f"({entry['analyze_seconds']}s){mark}")

    # ---- dataset-level quality summary (real numbers only) ----
    weights = profile.compute_sample_weights()
    centroids = profile.weighted_centroids()
    outliers = profile.outlier_samples()
    reliability = profile.reliability_report()
    aggregate = profile.aggregate()

    # Persist the profile through the platform store.
    store = ReferenceProfileStore()
    saved_path = store.save(profile)
    print(f"[profile] saved -> {saved_path}")

    durs = [float(f["analysis"]["duration_s"]) for f in per_file]
    snrs = [f["analysis"]["loudness"]["snr_proxy_db"] for f in per_file
            if f["analysis"]["loudness"]["snr_proxy_db"] is not None]
    floors = [f["analysis"]["loudness"]["noise_floor_db"] for f in per_file
              if f["analysis"]["loudness"]["noise_floor_db"] is not None]
    clips = [f["analysis"]["loudness"]["clipping_ratio"] for f in per_file
             if f["analysis"]["loudness"]["clipping_ratio"] is not None]
    voiced = [f["analysis"]["vad"]["voiced_ratio"] for f in per_file
              if f["analysis"]["vad"]["voiced_ratio"] is not None]
    f0med = [f["analysis"]["f0"]["median_hz"] for f in per_file
             if f["analysis"]["f0"]["median_hz"] is not None]
    hnr = [f["analysis"]["hnr_db"] for f in per_file
           if f["analysis"]["hnr_db"] is not None]
    grades = [f["quality"].get("grade") for f in per_file]

    summary = {
        "dataset": str(DATASET),
        "n_files": len(per_file),
        "total_duration_s": round(sum(durs), 2),
        "total_duration_min": round(sum(durs) / 60, 2),
        "duration_min_s": round(min(durs), 2),
        "duration_max_s": round(max(durs), 2),
        "mean_snr_db": round(sum(snrs) / len(snrs), 2) if snrs else None,
        "min_snr_db": round(min(snrs), 2) if snrs else None,
        "mean_noise_floor_db": round(sum(floors) / len(floors), 2) if floors else None,
        "max_clipping_ratio": max(clips) if clips else None,
        "mean_voiced_ratio": round(sum(voiced) / len(voiced), 4) if voiced else None,
        "f0_median_hz_across_files": round(sum(f0med) / len(f0med), 1) if f0med else None,
        "f0_range_across_files_hz": [round(min(f0med), 1), round(max(f0med), 1)] if f0med else None,
        "mean_hnr_db": round(sum(hnr) / len(hnr), 2) if hnr else None,
        "quality_grades": {str(g): grades.count(g) for g in sorted(set(grades), key=str)},
        "sample_weights": {f["file"]: round(w, 3)
                           for f, w in zip(per_file, weights)},
        "outlier_samples": outliers,
        "profile_reliability": reliability,
        "profile_aggregate_f0_median_hz": (aggregate.get("f0", {}) or {}).get("median_hz"),
        "embedder_backends": sorted(profile.embeddings.keys()),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    (OUT_DIR / "iraqi_voice_per_file.json").write_text(
        json.dumps(per_file, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "iraqi_voice_quality.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
