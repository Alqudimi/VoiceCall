"""FAISS retrieval index: build (training) + search (inference).

Upstream behaviour preserved:
* features concatenated from ``3_feature768/*.npy`` (v2) or ``3_feature256`` (v1);
* optional MiniBatchKMeans compression to 10 000 centroids beyond 200 000 rows;
* IVF list size ``16*sqrt(N)`` capped at ``N // 39``, nprobe = 1;
* writes ``added_IVF{n}_Flat_nprobe_1_<name>_<version>.index``.
"""
from __future__ import annotations

import glob
import logging
import time
from pathlib import Path

import faiss
import numpy as np

from rvc_platform.core.errors import CheckpointError

logger = logging.getLogger(__name__)


class FeatureIndex:
    """Read side of the retrieval index."""

    def __init__(self, index_path: Path) -> None:
        index_path = Path(index_path)
        if not index_path.is_file():
            raise CheckpointError(f"index file not found: {index_path}")
        self.path = index_path
        self.index = faiss.read_index(str(index_path))
        # upstream rtrvc/vc warning: only added_*.index (with stored vectors)
        # can blend; trained_*.index files cannot serve retrieval
        self.is_trained_only = "trained_" in index_path.name and "added_" not in index_path.name
        if self.is_trained_only:
            logger.warning(
                "index %s is a trained_* index (no stored vectors) — retrieval "
                "blending is disabled; use the added_*.index instead",
                index_path.name,
            )
        try:
            self.vectors = self.index.reconstruct_n(0, self.index.ntotal)
        except Exception:  # some IVF variants cannot reconstruct
            self.vectors = None
        # Optimize IVF search: search multiple Voronoi cells (nprobe=16) for high-accuracy timbre matching
        try:
            ivf = faiss.extract_index_ivf(self.index)
            if ivf is not None and ivf.nprobe < 8:
                ivf.nprobe = 16
        except Exception:
            pass

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    @property
    def dim(self) -> int:
        return int(self.index.d)

    def blend(self, feats: np.ndarray, index_rate: float, k: int = 8) -> np.ndarray:
        """Upstream top-k weighted replacement:
        ``feats' = r * sum_j w_j * v_j + (1 - r) * feats`` with
        ``w_j ∝ (1/score_j)^2`` (softmax-free weighting used by RVC).
        Returns the blended features (shape == feats.shape).
        """
        if index_rate <= 0 or self.size == 0 or self.vectors is None:
            return feats
        feats32 = np.ascontiguousarray(feats.astype("float32", copy=False))
        try:
            ivf = faiss.extract_index_ivf(self.index)
            if ivf is not None and ivf.nprobe < 8:
                ivf.nprobe = 16
        except Exception:
            pass

        score, ix = self.index.search(feats32, k)
        valid_mask = ix >= 0
        if not valid_mask.any():
            return feats

        safe_ix = np.where(valid_mask, ix, 0)
        weight = np.square(1.0 / np.maximum(score, 1e-8))
        weight = np.where(valid_mask, weight, 0.0)
        weight_sum = weight.sum(axis=1, keepdims=True)
        weight = np.where(weight_sum > 0, weight / np.maximum(weight_sum, 1e-8), 0.0)

        blended = np.sum(self.vectors[safe_ix] * np.expand_dims(weight, axis=2), axis=1)
        has_retrieval = (weight_sum > 0).squeeze(-1)
        mixed = feats.copy()
        mixed[has_retrieval] = (
            blended[has_retrieval].astype(feats.dtype) * index_rate
            + feats[has_retrieval] * (1.0 - index_rate)
        )
        return mixed.astype(feats.dtype)


def build_index(
    feature_dir: Path,
    out_dir: Path,
    name: str,
    version: str,
    n_cpu: int = 1,
    progress=None,
) -> Path:
    """Train + fill the FAISS index from extracted HuBERT features.

    ``feature_dir`` is ``<run>/3_feature768`` (v2) or ``3_feature256`` (v1).
    Returns the written ``added_*.index`` path.
    """
    feature_dir = Path(feature_dir)
    if not feature_dir.is_dir() or not any(feature_dir.glob("*.npy")):
        raise CheckpointError(f"no extracted features in {feature_dir}")
    dims = 256 if version == "v1" else 768

    feats = [np.load(p) for p in sorted(feature_dir.glob("*.npy"))]
    big_npy = np.concatenate(feats, 0)
    big_npy = big_npy[np.random.permutation(big_npy.shape[0])]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if big_npy.shape[0] > 200_000:
        logger.info("clustering %d rows -> 10000 centroids", big_npy.shape[0])
        from sklearn.cluster import MiniBatchKMeans

        big_npy = MiniBatchKMeans(
            n_clusters=10_000,
            verbose=False,
            batch_size=256 * max(n_cpu, 1),
            compute_labels=False,
            init="random",
        ).fit(big_npy).cluster_centers_.astype(np.float32)

    n_ivf = max(1, min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39))
    logger.info("features %s | IVF cells %s", big_npy.shape, n_ivf)
    index = faiss.index_factory(dims, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    t0 = time.time()
    index.train(big_npy)
    logger.info("index trained in %.1fs", time.time() - t0)
    for start in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[start : start + 8192])
        if progress:
            progress(min(1.0, (start + 8192) / big_npy.shape[0]))

    added_name = f"added_IVF{n_ivf}_Flat_nprobe_1_{name}_{version}.index"
    added_path = out_dir / added_name
    faiss.write_index(index, str(added_path))
    logger.info("index written: %s", added_path)
    return added_path


def newest_added_index(run_dir: Path, name: str, version: str) -> Path | None:
    pattern = str(Path(run_dir) / f"added_IVF*_Flat_nprobe_*_{name}_{version}.index")
    hits = [p for p in glob.glob(pattern) if Path(p).is_file()]
    return Path(max(hits, key=lambda p: Path(p).stat().st_mtime)) if hits else None
