"""Shared data/orchestration helpers for training and benchmarking.

The trainer (`training.train`) and the benchmark (`benchmark.run`) need the same
front half — pick a device, fetch + label proteins, split them — so it lives here
once instead of being copy-pasted. These helpers are quiet (no printing); the
callers narrate their own progress.
"""

from __future__ import annotations

from viral_annotation.config import HOLDOUT_FAMILY, UNIPROT_VIRAL_QUERY
from viral_annotation.data import labels as labels_mod
from viral_annotation.data.cluster import cluster_sequences
from viral_annotation.data.split import cluster_split, split_proteins


def auto_device(torch) -> str:
    """Best available torch device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def annotation_stats(proteins) -> str:
    """One-line summary of a protein set's manual/IEA annotation coverage."""
    n_manual = sum(p.n_manual for p in proteins)
    n_iea = sum(p.n_iea for p in proteins)
    have_manual = sum(1 for p in proteins if p.has_manual)
    pct = 100 * have_manual / max(len(proteins), 1)
    return (f"{len(proteins)} proteins | manual-having {have_manual} ({pct:.1f}%) | "
            f"raw annotations manual={n_manual} iea={n_iea}")


def load_proteins(dag, limit: int | None = None, query: str = UNIPROT_VIRAL_QUERY,
                  records_path=None, leaf_only: bool = False) -> list:
    """Fetch reviewed proteins (or load cached JSONL caches) and build GO labels.

    `query` selects the pathogen domain (default the viral taxon); pass a domain
    profile's `uniprot_query` for bacteria. `records_path`, if given, reads cached
    RawProtein records (from `labels.save_raw`) instead of hitting UniProt — useful
    to skip the slow, rate-limited fetch on re-runs / in cloud notebooks. It may be
    a single path or several (list, or comma-separated string) which are
    concatenated — e.g. the viral + bacterial caches for the unified domain. Cached
    records must match the domain(s) you're training (they aren't re-filtered by
    `query`). `leaf_only` builds most-specific (leaf) labels instead of propagating.
    """
    if records_path:
        from pathlib import Path

        if isinstance(records_path, str):
            paths = [p.strip() for p in records_path.split(",") if p.strip()]
        else:
            paths = list(records_path)
        raw = []
        for path in paths:
            for i, r in enumerate(labels_mod.load_raw(Path(path))):
                if limit and i >= limit:
                    break
                raw.append(r)
    else:
        raw = list(labels_mod.fetch_raw(limit=limit, query=query))
    return [p for p in labels_mod.label_proteins(raw, dag, leaf_only=leaf_only) if p.sequence]


def make_split(proteins, *, use_cluster: bool = True, holdout_families=HOLDOUT_FAMILY,
               family_suffixes: tuple[str, ...] = ("viridae",)):
    """Split proteins into train/val/test (+ optional held-out families).

    `use_cluster` uses the leakage-safe 30%-identity cluster split; otherwise a
    plain random split (not leakage-safe — for quick checks only). `family_suffixes`
    is the lineage-clade suffix(es) that identify the holdout family rank (viral
    'viridae'; bacterial 'aceae'). `holdout_families` may be a single family name or
    a collection (one viral + one bacterial for the unified domain).
    """
    if use_cluster:
        return cluster_split(proteins, cluster_sequences(proteins),
                             holdout_families=holdout_families, family_suffixes=family_suffixes)
    return split_proteins(proteins)
