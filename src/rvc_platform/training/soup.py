# ============================================================================ #
# Iraqi Voice Studio — Stage-3 researched quality lever: CHECKPOINT SOUP
#
# Research basis (deep-research dossier, Nov 2026 — docs/RESEARCH_REPORT_AR.md):
#   * "The Unusual Effectiveness of Averaging in GAN Training" (Yazici et al.):
#     averaging generator weights along one trajectory beats the last iterate.
#   * Model Soups (Wortsman et al., ICML 2022) + SWA: averaging fine-tune
#     checkpoints of the same run improves quality/robustness for free.
#   * Metrics-Weighted Averaging (arXiv:2504.18580): weight the average by a
#     validation metric instead of uniform weights.
#   * Applio REMOVED its automated overtraining detector (3.6.4, "obsolete");
#     current best practice = objective checkpoint scoring + averaging, which
#     is exactly what this module automates for the generator (D is never
#     averaged — not standard practice).
#
# Colab-safe contract: pure torch + stdlib, CPU-only, atomic writes, every
# failure degrades to a clear message (never crashes a session).
# ============================================================================ #
"""Checkpoint averaging (uniform / metric-weighted) for the RVC v2 generator."""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

G_STEP_RE = re.compile(r"^G_(\d+|final|latest)\.pth$")


def _step_key(name: str) -> tuple[int, int]:
    m = G_STEP_RE.match(name)
    if not m:
        return (-1, 0)
    tok = m.group(1)
    return (int(tok), 0) if tok.isdigit() else (10**12, 0)


def list_g_checkpoints(checkpoint_dir: str | Path,
                       k: int | None = None) -> list[Path]:
    """Newest-first G_*.pth training checkpoints (excludes D_/D2_/exports)."""
    d = Path(checkpoint_dir)
    names = ([p.name for p in d.iterdir() if G_STEP_RE.match(p.name)]
             if d.is_dir() else [])
    names.sort(key=_step_key, reverse=True)
    paths = [d / n for n in names]
    return paths[:k] if k else paths


def average_state_dicts(states: list[dict],
                        weights: list[float] | None = None) -> dict:
    """Average training state dicts (float tensors averaged, rest from newest).

    Missing float tensors in some states are averaged over the states that do
    have them with renormalized weights; non-float entries (int buffers,
    counters) are copied from the newest state.
    """
    if not states:
        raise ValueError("no state dicts to average")
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    if len(weights) != len(states):
        raise ValueError("weights length must match states length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    weights = [w / total for w in weights]

    newest = states[-1]
    out: dict = {}
    for key, ref in newest.items():
        pairs = [(s[key].float(), w) for s, w in zip(states, weights)
                 if key in s and torch.is_tensor(s[key])
                 and s[key].is_floating_point()]
        if pairs:
            z = sum(w for _, w in pairs)
            if z <= 0:
                out[key] = ref
                continue
            acc = pairs[0][0] * (pairs[0][1] / z)
            for t, w in pairs[1:]:
                acc = acc + t * (w / z)
            out[key] = acc.to(ref.dtype)
        else:
            out[key] = ref
    return out


def _softmax_mwa(scores: list[float], mode: str = "min") -> list[float]:
    """Smooth metric-weighted averaging weights (MWA, arXiv:2504.18580).

    Lower-is-better metrics ('min') map to higher weights; sharpness 2 keeps
    the average close to the best checkpoints without collapsing to argmax.
    """
    mn, mx = min(scores), max(scores)
    span = max(1e-9, mx - mn)
    if mode == "max":
        neg = [-(mx - s) / span for s in scores]
    else:
        neg = [-(s - mn) / span for s in scores]
    exps = [math.exp(2.0 * n) for n in neg]
    z = sum(exps)
    return [e / z for e in exps]


def _score_from_payload(payload: dict,
                        metric_keys: tuple[str, ...]) -> float | None:
    """Best-effort per-checkpoint score from payload metadata."""
    for container in (payload.get("best") or {}, payload):
        for key in metric_keys:
            v = container.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def build_soup(run_dir: str | Path, k: int = 5, weighted: bool = False,
               out_name: str = "G_soup.pth",
               metric_keys: tuple[str, ...] = ("composite", "val_mel", "train_mel"),
               mode: str = "min") -> dict:
    """Average the K newest G checkpoints of ``run_dir`` into checkpoints/G_soup.pth.

    ``weighted=True`` uses MWA weights when a per-checkpoint metric is
    recoverable; uniform otherwise. Returns a summary dict; missing/short data
    returns {"ok": False, "reason": ...} instead of raising.
    """
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    paths = list_g_checkpoints(ckpt_dir, k)
    if len(paths) < 2:
        return {"ok": False,
                "reason": f"need >=2 G checkpoints in {ckpt_dir}, found {len(paths)}",
                "run_dir": str(run_dir)}

    payloads: list[dict] = []
    for p in paths:  # paths are newest-first; keep that order
        try:
            payloads.append(torch.load(p, map_location="cpu", weights_only=False))
        except Exception:
            logger.warning("skipping unreadable checkpoint %s", p, exc_info=True)
    if len(payloads) < 2:
        return {"ok": False, "reason": "fewer than 2 readable G checkpoints",
                "run_dir": str(run_dir)}

    weights = None
    if weighted:
        scores = [_score_from_payload(pl, metric_keys) for pl in payloads]
        if all(s is not None for s in scores) and len(set(scores)) > 1:
            weights = _softmax_mwa(scores, mode)
            logger.info("metric-weighted soup: scores=%s weights=%s",
                        [round(s, 6) for s in scores],
                        [round(w, 4) for w in weights])
        else:
            logger.info("no usable per-checkpoint metric — uniform soup")

    merged = average_state_dicts([pl["model"] for pl in payloads], weights)
    soup_payload = dict(payloads[0])            # newest metadata wins
    soup_payload["model"] = merged
    soup_payload["soup"] = {
        "sources": [p.name for p in paths],
        "weights": weights or [round(1.0 / len(payloads), 6)] * len(payloads),
        "metric_weighted": bool(weighted and weights),
    }

    out_path = ckpt_dir / out_name
    tmp = out_path.with_suffix(".pth.part")
    torch.save(soup_payload, tmp)
    tmp.replace(out_path)

    summary = {
        "ok": True, "run_dir": str(run_dir),
        "checkpoints": [p.name for p in paths],
        "weights": soup_payload["soup"]["weights"],
        "metric_weighted": soup_payload["soup"]["metric_weighted"],
        "output": str(out_path),
        "newest_epoch": payloads[0].get("iteration") or payloads[0].get("epoch"),
    }
    logger.info("checkpoint soup written: %s", summary)
    return summary
