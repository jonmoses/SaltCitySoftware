"""Tests for the GO DAG parser and true-path operations.

A tiny synthetic ontology stands in for go-basic.obo so the tests need no
download. Structure (arrows point child -> parent):

    GO:0000003 --is_a--> GO:0000002 --is_a--> GO:0000001 (root, MF)
    GO:0000004 --part_of--> GO:0000002
    GO:0000009 (obsolete)
    GO:0000010 alt_id of GO:0000003
"""

from textwrap import dedent

import pytest

from viral_annotation.ontology import GoDag

TINY_OBO = dedent(
    """
    format-version: 1.2

    [Term]
    id: GO:0000001
    name: root function
    namespace: molecular_function

    [Term]
    id: GO:0000002
    name: mid function
    namespace: molecular_function
    is_a: GO:0000001 ! root function

    [Term]
    id: GO:0000003
    name: leaf function
    namespace: molecular_function
    alt_id: GO:0000010
    is_a: GO:0000002 ! mid function

    [Term]
    id: GO:0000004
    name: part function
    namespace: molecular_function
    relationship: part_of GO:0000002 ! mid function

    [Term]
    id: GO:0000009
    name: obsolete function
    namespace: molecular_function
    is_obsolete: true
    """
).strip()


@pytest.fixture
def dag(tmp_path):
    obo = tmp_path / "tiny.obo"
    obo.write_text(TINY_OBO, encoding="utf-8")
    return GoDag.from_obo(obo)


def test_parse_counts_non_obsolete(dag):
    # 5 terms in file, 1 obsolete dropped -> 4 remain.
    assert len(dag) == 4
    assert "GO:0000009" not in dag


def test_ancestors_via_is_a(dag):
    anc = dag.ancestors("GO:0000003")  # includes self
    assert anc == {"GO:0000003", "GO:0000002", "GO:0000001"}


def test_ancestors_via_part_of(dag):
    anc = dag.ancestors("GO:0000004")
    assert anc == {"GO:0000004", "GO:0000002", "GO:0000001"}


def test_alt_id_resolves(dag):
    assert dag.resolve("GO:0000010") == "GO:0000003"
    assert "GO:0000010" in dag
    assert dag.ancestors("GO:0000010") == {"GO:0000003", "GO:0000002", "GO:0000001"}


def test_propagate_true_path(dag):
    propagated = dag.propagate({"GO:0000003"})
    assert propagated == {"GO:0000003", "GO:0000002", "GO:0000001"}


def test_most_specific_drops_redundant_ancestors(dag):
    # Full lineage in -> only the leaf survives (ancestors are redundant).
    assert dag.most_specific({"GO:0000003", "GO:0000002", "GO:0000001"}) == {"GO:0000003"}


def test_most_specific_keeps_independent_leaves(dag):
    # Two leaves sharing only ancestors -> both kept; the shared parent dropped.
    assert dag.most_specific({"GO:0000003", "GO:0000004", "GO:0000002"}) == {
        "GO:0000003", "GO:0000004"
    }


def test_most_specific_resolves_alt_and_skips_unknown(dag):
    # alt id resolves to its primary; terms absent from the DAG are dropped.
    assert dag.most_specific({"GO:0000010", "GO:0000001", "GO:9999999"}) == {"GO:0000003"}


def test_most_specific_single_term_unchanged(dag):
    assert dag.most_specific({"GO:0000002"}) == {"GO:0000002"}


def test_correct_scores_parent_at_least_child(dag):
    # Child more confident than ancestors -> ancestors lifted to child's score.
    raw = {"GO:0000003": 0.9, "GO:0000002": 0.4, "GO:0000001": 0.2}
    fixed = dag.correct_scores(raw)
    assert fixed["GO:0000002"] == pytest.approx(0.9)
    assert fixed["GO:0000001"] == pytest.approx(0.9)
    assert fixed["GO:0000003"] == pytest.approx(0.9)


def test_correct_scores_leaves_consistent_unchanged(dag):
    raw = {"GO:0000003": 0.3, "GO:0000002": 0.6, "GO:0000001": 0.8}
    fixed = dag.correct_scores(raw)
    assert fixed == pytest.approx(raw)
