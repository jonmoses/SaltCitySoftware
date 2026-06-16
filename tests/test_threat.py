"""Tests for the danger DAG descendants, danger-map assertion, and characterization."""

from types import SimpleNamespace

import pytest

from viral_annotation.data.danger_terms import DANGER_CATEGORIES
from viral_annotation.threat import (
    build_danger_map,
    characterize_protein,
    characterize_proteome,
)


def test_descendants(tiny_dag):
    # MF: 0003674(root) -> 0002 -> {0003, 0004}.
    assert tiny_dag.descendants("GO:0000002") == frozenset(
        {"GO:0000002", "GO:0000003", "GO:0000004"}
    )
    assert tiny_dag.descendants("GO:0000002", include_self=False) == frozenset(
        {"GO:0000003", "GO:0000004"}
    )
    # A leaf has no descendants but returns itself when include_self.
    assert tiny_dag.descendants("GO:0000003") == frozenset({"GO:0000003"})
    assert tiny_dag.descendants("GO:0000003", include_self=False) == frozenset()
    # Root reaches every MF term.
    assert "GO:0000004" in tiny_dag.descendants("GO:0003674")


def test_build_danger_map_raises_on_missing_root(tiny_dag):
    # The real danger roots (e.g. GO:0090729) are absent from the tiny DAG, so the
    # non-obsolete assertion must fire rather than silently characterizing nothing.
    with pytest.raises(ValueError):
        build_danger_map(tiny_dag)


def test_characterize_flags_right_category(tiny_dag):
    # Put a known term under one category; everything else empty.
    target_key = DANGER_CATEGORIES[0].key
    danger_map = {cat.key: frozenset() for cat in DANGER_CATEGORIES}
    danger_map[target_key] = frozenset({"GO:0000003"})

    annotated = SimpleNamespace(
        accession="P1", organism="testvirus",
        terms={"GO:0000003": 0.8, "GO:0000011": 0.1},  # 0011 is in no category
    )
    pt = characterize_protein(annotated, danger_map, tiny_dag, display_floor=0.05)

    assert set(pt.categories) == {target_key}
    ch = pt.categories[target_key]
    assert ch.peak == pytest.approx(0.8)
    assert ch.terms[0].go_id == "GO:0000003"
    # With no background passed, lift equals the raw probability.
    assert ch.peak_lift == pytest.approx(0.8)
    assert pt.standout() == pytest.approx(0.8)


def test_display_floor_drops_low_confidence(tiny_dag):
    target_key = DANGER_CATEGORIES[0].key
    danger_map = {cat.key: frozenset() for cat in DANGER_CATEGORIES}
    danger_map[target_key] = frozenset({"GO:0000003"})
    annotated = SimpleNamespace(
        accession="P2", organism="t", terms={"GO:0000003": 0.02},
    )
    pt = characterize_protein(annotated, danger_map, tiny_dag, display_floor=0.05)
    assert pt.categories == {}


def test_lift_over_background_surfaces_standout(tiny_dag):
    # A term that saturates the whole proteome (everyone ~0.9) gives ~0 lift;
    # a term one protein scores high on while others don't gives high lift.
    target_key = DANGER_CATEGORIES[0].key
    second_key = DANGER_CATEGORIES[1].key
    danger_map = {cat.key: frozenset() for cat in DANGER_CATEGORIES}
    danger_map[target_key] = frozenset({"GO:0000003"})   # saturating term
    danger_map[second_key] = frozenset({"GO:0000004"})   # distinctive term

    annotated = [
        SimpleNamespace(accession="A", organism="t",
                        terms={"GO:0000003": 0.9, "GO:0000004": 0.9}),  # standout on 0004
        SimpleNamespace(accession="B", organism="t",
                        terms={"GO:0000003": 0.9, "GO:0000004": 0.0}),
        SimpleNamespace(accession="C", organism="t",
                        terms={"GO:0000003": 0.9, "GO:0000004": 0.0}),
    ]
    pt = characterize_proteome("v", annotated, danger_map, tiny_dag, display_floor=0.0)

    # background: 0003 mean 0.9 (saturated), 0004 mean 0.3.
    assert pt.background["GO:0000003"] == pytest.approx(0.9)
    assert pt.background["GO:0000004"] == pytest.approx(0.3)
    # protein A stands out via the distinctive term, not the saturated one.
    top, lift = pt.ranked(n=1)[0]
    assert top.accession == "A"
    assert lift == pytest.approx(0.6)  # 0.9 - 0.3
    assert top.categories[target_key].peak_lift == pytest.approx(0.0)  # saturated
