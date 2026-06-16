"""Term-vocabulary selection and multi-hot label matrices.

The prediction target set (the classifier's output columns) is chosen from the
TRAINING labels: a GO term is kept only if at least `min_count` training proteins
carry it (after propagation), and it isn't an ontology root. Val/test are then
scored against this same fixed vocabulary using their MANUAL label sets.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from viral_annotation.config import GO_ROOTS, MIN_TERM_COUNT


@dataclass
class TermVocab:
    """The fixed set of GO terms the classifier predicts, with a column index."""

    terms: list[str]              # column order
    index: dict[str, int]         # term id -> column
    namespaces: list[str]         # namespace per column (aligned to `terms`)

    def __len__(self) -> int:
        return len(self.terms)

    def columns_by_namespace(self) -> dict[str, list[int]]:
        """Map each GO namespace to the column indices it owns (for per-ns Fmax)."""
        out: dict[str, list[int]] = {}
        for col, ns in enumerate(self.namespaces):
            out.setdefault(ns, []).append(col)
        return out


def select_vocab(
    train_proteins: list,
    dag,
    min_count: int = MIN_TERM_COUNT,
    field: str = "terms_all",
    namespaces: list[str] | None = None,
) -> TermVocab:
    """Pick prediction-target terms from training labels by frequency.

    Args:
        field: which label set to count — "terms_all" (manual+IEA) for the
               asymmetric policy, "terms_manual" for the manual-only policy.
        namespaces: if given, keep only terms in these GO namespaces (used to
               build a per-namespace vocabulary). Default: all namespaces.

    Roots and terms absent from the DAG are excluded. Terms are ordered by
    namespace then term id for stable, readable columns.
    """
    counts: Counter[str] = Counter()
    for p in train_proteins:
        counts.update(getattr(p, field))

    def ns_ok(ns: str | None) -> bool:
        return ns is not None if namespaces is None else ns in namespaces

    kept = [
        t for t, c in counts.items()
        if c >= min_count and t not in GO_ROOTS and ns_ok(dag.namespace_of(t))
    ]
    kept.sort(key=lambda t: (dag.namespace_of(t), t))

    index = {t: i for i, t in enumerate(kept)}
    namespaces = [dag.namespace_of(t) for t in kept]
    return TermVocab(terms=kept, index=index, namespaces=namespaces)


def build_labels(proteins: list, vocab: TermVocab, field: str):
    """Build a multi-hot label matrix Y [P x N] for `proteins`.

    Args:
        field: which label set to use — "terms_all" for train, "terms_manual"
               for val/test (the asymmetric rule).
    """
    import numpy as np

    if field not in ("terms_all", "terms_manual", "terms_iea"):
        raise ValueError(f"field must be terms_all/terms_manual/terms_iea, got {field!r}")

    Y = np.zeros((len(proteins), len(vocab)), dtype="float32")
    for row, p in enumerate(proteins):
        for term in getattr(p, field):
            col = vocab.index.get(term)
            if col is not None:
                Y[row, col] = 1.0
    return Y
