"""True-path propagation of annotation rows, grouped per protein and per tier.

The `propagate` tool turns raw annotations (annotations.tsv) into propagated
labels (labels.tsv): for each protein, each evidence tier's GO terms are closed
under ancestors independently. Because ancestor closure distributes over union,
propagating tiers separately and unioning downstream is equivalent to propagating
the combined set — so `--evidence manual` and `--evidence manual+iea` both stay
correct off the one labels.tsv.
"""

from __future__ import annotations

from typing import Iterator, Mapping, Sequence

from valib.godag import GoDag


# Pre:  pairs is this protein's (go_id, tier) annotations; dag is loaded.
# Post: yields propagated (go_id, tier) pairs, deduplicated, with each tier's
#       terms closed under ancestors. Order is unspecified (caller sorts).
# Inputs:  pairs (Sequence[tuple[str, str]]); dag (GoDag)
# Outputs: Iterator[tuple[str, str]] — (go_id, tier)
def _propagate_one(pairs: Sequence[tuple[str, str]], dag: GoDag) -> Iterator[tuple[str, str]]:
    by_tier: dict[str, set[str]] = {}
    for go_id, tier in pairs:
        by_tier.setdefault(tier, set()).add(go_id)
    seen: set[tuple[str, str]] = set()
    for tier, terms in by_tier.items():
        for go_id in dag.propagate(terms):
            key = (go_id, tier)
            if key not in seen:
                seen.add(key)
                yield key


# Pre:  grouped maps accession -> list of (go_id, tier); dag is loaded.
# Post: returns propagated label rows (accession, go_id, tier), sorted by
#       (accession, go_id, tier) for deterministic output. Each protein's terms
#       are true-path closed within each tier.
# Inputs:  grouped (Mapping[str, Sequence[tuple[str, str]]]); dag (GoDag)
# Outputs: list[tuple[str, str, str]] — (accession, go_id, tier)
def propagate_annotations(
    grouped: Mapping[str, Sequence[tuple[str, str]]], dag: GoDag
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for accession, pairs in grouped.items():
        for go_id, tier in _propagate_one(pairs, dag):
            rows.append((accession, go_id, tier))
    rows.sort()
    return rows
