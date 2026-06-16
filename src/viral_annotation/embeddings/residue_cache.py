"""Per-residue embedding cache for learned (attention) pooling.

Unlike the pooled-vector cache, attention pooling needs every residue's vector
(the pooler learns which to upweight), so we store one fp16 array [L x d] per
protein keyed by accession. fp16 halves the footprint (~20 GB for the full set;
a few GB for a subset). Written per-protein so a long run checkpoints and a crash
loses at most the current chunk.
"""

from __future__ import annotations

from pathlib import Path

from viral_annotation.config import (
    DEFAULT_ESM_MODEL,
    DEFAULT_POOLING,
    EMBEDDINGS_CACHE,
)


def residue_cache_dir(model_key: str, repr_layer: int | None, window: bool = True) -> Path:
    layer = "last" if repr_layer is None else str(repr_layer)
    suffix = "_win" if window else ""
    return EMBEDDINGS_CACHE / f"residues_esm2_{model_key}_layer-{layer}{suffix}"


def _path(cache_dir: Path, accession: str) -> Path:
    return cache_dir / f"{accession}.npy"


def cache_residues(
    records,
    model_key: str = DEFAULT_ESM_MODEL,
    repr_layer: int | None = None,
    window: bool = True,
    chunk: int = 256,
    embedder=None,
) -> Path:
    """Ensure per-residue embeddings exist on disk for `records`. Returns the dir."""
    import numpy as np

    records = list(records)
    cache_dir = residue_cache_dir(model_key, repr_layer, window)
    cache_dir.mkdir(parents=True, exist_ok=True)

    missing = [r for r in records if not _path(cache_dir, r.accession).exists()]
    if missing:
        if embedder is None:
            from viral_annotation.embeddings.esm import ESMEmbedder

            embedder = ESMEmbedder(model_key=model_key, pooling="mean",
                                   repr_layer=repr_layer, window=window)
        for c in range(0, len(missing), chunk):
            part = missing[c:c + chunk]
            arrs = embedder.embed_residues([r.sequence for r in part])
            for r, a in zip(part, arrs):
                np.save(_path(cache_dir, r.accession), a)
            print(f"       residues {min(c + chunk, len(missing))}/{len(missing)} new",
                  flush=True)
    return cache_dir


def load_residues(accession: str, cache_dir: Path):
    """Load one protein's per-residue embeddings as float32 [L x d]."""
    import numpy as np

    return np.load(_path(cache_dir, accession)).astype("float32")
