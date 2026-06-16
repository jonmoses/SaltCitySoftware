"""Temporal (CAFA / NetGO) split from dated experimental annotations.

Per ontology: a protein is a TRAIN protein if it has an experimental annotation in
that ontology BEFORE the cutoff (labels = propagate(before-cutoff terms)); it is a
no-knowledge TEST protein if it has NO such annotation before the cutoff but gains
one after (labels = propagate(all its terms in that ontology). This mirrors the
CAFA "no-knowledge" target definition NetGO benchmarks against.

CAVEAT: QuickGO's date is the last-update date, not the original assertion date, so
NK selection is approximate (CAFA uses dated DB snapshots we can't easily obtain).
A cutoff before any bulk re-dating spike reduces contamination.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from viral_annotation.config import GO_NAMESPACES


@dataclass
class BenchProtein:
    """A protein in the benchmark, carrying one ontology's label set.

    `terms_manual` named for drop-in reuse of homology_scores / build_labels /
    select_vocab / embed_records, which expect that attribute.
    """

    accession: str
    sequence: str
    terms_manual: frozenset[str]

    @property
    def has_manual(self) -> bool:
        return bool(self.terms_manual)


@dataclass
class TemporalSplit:
    train: dict        # namespace -> list[BenchProtein] (before-cutoff labels)
    test: dict         # namespace -> list[BenchProtein] (no-knowledge, all labels)
    cutoff: int

    def summary(self) -> str:
        return " | ".join(
            f"{ns.split('_')[0][:2].upper()}: train {len(self.train[ns])} test {len(self.test[ns])}"
            for ns in GO_NAMESPACES
        )


def build_temporal_split(annotations, seq_by_acc: dict, dag, cutoff: int) -> TemporalSplit:
    """Build the temporal split from experimental annotations + sequences.

    annotations: iterable of quickgo.ExpAnnotation. seq_by_acc maps accession->
    sequence (only these proteins, the ones we can embed, are used). cutoff is an
    int YYYYMMDD.
    """
    # accession -> namespace -> list of (go_id, date)
    by_prot: dict = defaultdict(lambda: defaultdict(list))
    for a in annotations:
        if a.accession in seq_by_acc:
            by_prot[a.accession][a.namespace].append((a.go_id, a.date))

    train = {ns: [] for ns in GO_NAMESPACES}
    test = {ns: [] for ns in GO_NAMESPACES}
    for acc, ns_map in by_prot.items():
        seq = seq_by_acc[acc]
        for ns in GO_NAMESPACES:
            terms = ns_map.get(ns, [])
            if not terms:
                continue
            before = [g for g, d in terms if d < cutoff]
            if before:
                train[ns].append(BenchProtein(acc, seq, frozenset(dag.propagate(before))))
            else:  # no ns-annotation before cutoff, but has one after -> no-knowledge
                test[ns].append(BenchProtein(acc, seq, frozenset(dag.propagate([g for g, _ in terms]))))
    return TemporalSplit(train=train, test=test, cutoff=cutoff)
