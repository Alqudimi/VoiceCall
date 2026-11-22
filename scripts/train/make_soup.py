#!/usr/bin/env python3
"""Build a checkpoint soup (uniform or metric-weighted average) for one run.

Usage (Colab / local, from the project root):
    uv run python scripts/train/make_soup.py runs/<experiment> --k 5
    uv run python scripts/train/make_soup.py runs/<experiment> --k 5 --weighted
Optional: --export to also write an inference .pth next to the run's exports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rvc_platform.training import soup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="runs/<experiment> directory")
    ap.add_argument("--k", type=int, default=5, help="number of newest G checkpoints (default 5)")
    ap.add_argument("--weighted", action="store_true", help="metric-weighted averaging (MWA) when scores exist")
    ap.add_argument("--export", action="store_true", help="also export an inference checkpoint from the soup")
    args = ap.parse_args()

    summary = soup.build_soup(args.run_dir, k=args.k, weighted=args.weighted)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary.get("ok"):
        return 1

    if args.export:
        from rvc_platform.engine.rvc import checkpoint as engine_ckpt
        from rvc_platform.training.run_config import TrainingRunConfig

        run_dir = Path(args.run_dir)
        cfg = TrainingRunConfig.load(run_dir)
        hps = cfg.to_hparams()
        payload = json.loads(json.dumps({}))  # placeholder to keep linters quiet
        import torch
        full = torch.load(summary["output"], map_location="cpu", weights_only=False)
        epoch = int(full.get("iteration") or full.get("epoch") or 0)
        out = run_dir / "exports" / f"{cfg.experiment_name}_soup_e{epoch}.pth"
        engine_ckpt.export_inference_checkpoint(
            full["model"], hps, epoch, hps.version, out, hps.sample_rate, hps.if_f0)
        print(f"soup inference export: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
