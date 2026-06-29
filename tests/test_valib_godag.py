"""Contract tests for valib.godag: OBO parse + true-path ops (pure stdlib)."""

from valib.godag import GoDag


def test_from_obo_parses_terms_and_edges(tiny_obo_path):
    dag = GoDag.from_obo(tiny_obo_path)
    assert len(dag) == 8
    assert "GO:0000003" in dag
    assert dag.namespace_of("GO:0000003") == "molecular_function"
    # is_a edge recorded as a parent.
    assert "GO:0000002" in dag.get("GO:0000003").parents


def test_ancestors_walk_to_root(tiny_obo_path):
    dag = GoDag.from_obo(tiny_obo_path)
    anc = dag.ancestors("GO:0000003", include_self=True)
    assert anc == {"GO:0000003", "GO:0000002", "GO:0003674"}
    strict = dag.ancestors("GO:0000003", include_self=False)
    assert "GO:0000003" not in strict


def test_descendants_invert_ancestors(tiny_obo_path):
    dag = GoDag.from_obo(tiny_obo_path)
    desc = dag.descendants("GO:0000002", include_self=False)
    assert desc == {"GO:0000003", "GO:0000004"}


def test_propagate_closes_under_ancestors(tiny_obo_path):
    dag = GoDag.from_obo(tiny_obo_path)
    out = dag.propagate({"GO:0000003", "GO:0000011"})
    assert out == {
        "GO:0000003", "GO:0000002", "GO:0003674",  # MF chain
        "GO:0000011", "GO:0008150",                # BP chain
    }


def test_correct_scores_lifts_parents_to_best_child(tiny_obo_path):
    dag = GoDag.from_obo(tiny_obo_path)
    scores = {"GO:0003674": 0.1, "GO:0000002": 0.2, "GO:0000003": 0.9}
    out = dag.correct_scores(scores)
    # Parent and grandparent lifted to the leaf's 0.9; leaf unchanged.
    assert out["GO:0000003"] == 0.9
    assert out["GO:0000002"] == 0.9
    assert out["GO:0003674"] == 0.9


def test_unknown_term_is_safe(tiny_obo_path):
    dag = GoDag.from_obo(tiny_obo_path)
    assert dag.ancestors("GO:9999999") == {"GO:9999999"}
    assert dag.namespace_of("GO:9999999") is None
