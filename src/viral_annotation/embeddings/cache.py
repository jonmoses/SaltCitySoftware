"""Precompute and cache per-protein ESM embeddings.

ESM is a frozen feature extractor, so we embed each sequence once and reuse the
vectors across every training run, threshold sweep, and classifier variant. The
cache is keyed by (model, pooling, layer) so different feature configs never
collide, and it is incremental: only sequences not already cached are computed.

Storage: one .npz per config holding {ids: [P], embeddings: [P x d] float32}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from viral_annotation.config import (
    DEFAULT_ESM_MODEL,
    DEFAULT_POOLING,
    EMBEDDINGS_CACHE,
)


def cache_path(model_key: str, pooling: str, repr_layer: int | None) -> Path:
    """Deterministic cache filename for a feature config."""
    layer = "last" if repr_layer is None else str(repr_layer)
    return EMBEDDINGS_CACHE / f"esm2_{model_key}_{pooling}_layer-{layer}.npz"


def _load_existing(path: Path):
    import numpy as np

    if not path.exists():
        return {}, 0
    data = np.load(path, allow_pickle=False)
    ids = data["ids"].tolist()
    mat = data["embeddings"]
    return {acc: mat[i] for i, acc in enumerate(ids)}, mat.shape[1]


def embed_records(
    records,
    model_key: str = DEFAULT_ESM_MODEL,
    pooling: str = DEFAULT_POOLING,
    repr_layer: int | None = None,
    batch_size: int | None = None,
    embedder=None,
):
    """Return (ids, X) for `records`, computing+caching any missing embeddings.

    Args:
        records: iterable of objects with `.accession` and `.sequence`.
        embedder: optional prebuilt ESMEmbedder (else one is built lazily, and
                  only if there is something to compute).

    Returns:
        ids: list[str] of accessions in input order.
        X:   np.ndarray [len(records) x d], rows aligned to `ids`.
    """
    import numpy as np

    records = list(records)
    path = cache_path(model_key, pooling, repr_layer)
    cached, _ = _load_existing(path)

    missing = [r for r in records if r.accession not in cached]
    if missing:
        if embedder is None:
            from viral_annotation.embeddings.esm import ESMEmbedder

            embedder = ESMEmbedder(
                model_key=model_key, pooling=pooling, repr_layer=repr_layer
            )
        # embed() handles length-sorting, token-budget batching, and preserves
        # input order. Process in chunks and persist after each so a long run
        # (a full proteome is a ~hours-long embed) checkpoints and a crash loses
        # at most one chunk.
        chunk = 1024
        for c in range(0, len(missing), chunk):
            part = missing[c:c + chunk]
            vecs = embedder.embed([r.sequence for r in part], batch_size=batch_size)
            for r, v in zip(part, vecs):
                cached[r.accession] = v
            _save(path, cached)
            print(f"       embedded {min(c + chunk, len(missing))}/{len(missing)} new "
                  f"(cache {len(cached)})", flush=True)

    ids = [r.accession for r in records]
    X = np.stack([cached[a] for a in ids]).astype("float32")
    return ids, X


def _save(path: Path, cached: dict) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(cached.keys())
    mat = np.stack([cached[a] for a in ids]).astype("float32")
    # Write to a temp file then replace, so an interrupted run can't corrupt the
    # cache. Pass an open file object: np.savez would otherwise append ".npz" to
    # any path not already ending in it (turning ".npz.tmp" into ".npz.tmp.npz").
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, ids=np.array(ids, dtype=str), embeddings=mat)
    tmp.replace(path)
