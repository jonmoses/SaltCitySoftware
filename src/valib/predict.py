"""Post-hoc hierarchical correction of prediction rows, grouped per protein.

The `correct` tool enforces the true-path rule on predicted scores: within each
protein, a parent term's score is raised to at least the best score among its
descendants, so the output is DAG-consistent (docs rationale: a protein predicted
to bind a specific ligand must also be predicted to "bind").
"""

from __future__ import annotations

from typing import Mapping, Sequence

from valib.godag import GoDag


# Pre:  pairs is one protein's (go_id, score_str) predictions; scores parse as
#       float; dag is loaded.
# Post: returns (go_id, corrected_score) pairs over the same ids, sorted by
#       go_id, each score raised to >= its best descendant present here.
# Inputs:  pairs (Sequence[tuple[str, str]]); dag (GoDag)
# Outputs: list[tuple[str, float]]
def _correct_one(pairs: Sequence[tuple[str, str]], dag: GoDag) -> list[tuple[str, float]]:
    scores = {go_id: float(score) for go_id, score in pairs}
    corrected = dag.correct_scores(scores)
    return sorted(corrected.items())


# Pre:  grouped maps accession -> list of (go_id, score_str); dag is loaded.
# Post: returns corrected rows (accession, go_id, score) as strings, sorted by
#       (accession, go_id). Scores are formatted with %.6g.
# Inputs:  grouped (Mapping[str, Sequence[tuple[str, str]]]); dag (GoDag)
# Outputs: list[tuple[str, str, str]] — (accession, go_id, score)
def correct_predictions(
    grouped: Mapping[str, Sequence[tuple[str, str]]], dag: GoDag
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for accession, pairs in grouped.items():
        for go_id, score in _correct_one(pairs, dag):
            rows.append((accession, go_id, f"{score:.6g}"))
    rows.sort()
    return rows
