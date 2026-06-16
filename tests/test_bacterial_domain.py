"""Tests for the bacterial pathogen-domain extension: the domain profile, the
suffix-configurable family holdout, the domain-keyed danger ontology / target panel,
and a liveness check that every bacterial danger root exists (non-obsolete) in go-basic."""

import pytest

from viral_annotation.config import (
    BACTERIAL_NAMESPACE_POLICY,
    GO_NAMESPACES,
    GO_OBO_PATH,
    get_domain,
)
from viral_annotation.data.danger_terms import (
    BACTERIAL_DANGER_CATEGORIES,
    VIRAL_DANGER_CATEGORIES,
    danger_categories,
)
from viral_annotation.data.proteomes import TARGET_BACTERIA, target_registry
from viral_annotation.data.split import family_of


def test_bacterial_domain_profile():
    dom = get_domain("bacterial")
    assert dom.taxon_id == 2
    assert dom.family_suffixes == ("aceae",)
    assert dom.holdout_family == "Francisellaceae"
    assert dom.models_subdir == "bacterial"
    assert dom.default_pooling == "mean"
    # Viral profile unchanged (back-compat: artifacts stay at MODELS_DIR root).
    assert get_domain("viral").models_subdir == ""


def test_bacterial_policy_all_asymmetric_mean():
    # Starting bacterial policy: train manual+IEA, mean pooling, every namespace.
    for ns in GO_NAMESPACES:
        pol = BACTERIAL_NAMESPACE_POLICY[ns]
        assert pol["train_field"] == "terms_all"
        assert pol["vocab_field"] == "terms_all"
        assert pol["pooling"] == "mean"


def test_family_of_suffix_configurable():
    lineage = ["Bacteria", "Pseudomonadota", "Gammaproteobacteria",
               "Thiotrichales", "Francisellaceae", "Francisella"]
    assert family_of(lineage, ("aceae",)) == "Francisellaceae"
    # The default viral suffix doesn't match a bacterial lineage.
    assert family_of(lineage) is None


def test_danger_categories_domain_keyed():
    bact = danger_categories("bacterial")
    assert bact is BACTERIAL_DANGER_CATEGORIES
    keys = [c.key for c in bact]
    assert len(keys) == len(set(keys))                     # unique keys
    # Bacterial-specific mechanisms the viral set lacks.
    assert {"secretion", "amr", "iron_piracy", "biofilm"} <= set(keys)
    assert danger_categories("viral") is VIRAL_DANGER_CATEGORIES
    with pytest.raises(KeyError):
        danger_categories("fungal")


def test_bacterial_target_panel():
    reg = target_registry("bacterial")
    assert reg is TARGET_BACTERIA
    # The held-out family's agent is in the panel — the genuine zero-shot target.
    assert reg["tularemia"].family == "Francisellaceae"
    assert reg["tularemia"].taxon_id == 263
    assert get_domain("bacterial").holdout_family == reg["tularemia"].family


@pytest.mark.skipif(not GO_OBO_PATH.exists(), reason="go-basic.obo not downloaded")
def test_bacterial_danger_roots_live_in_go_basic():
    from viral_annotation.ontology import GoDag
    from viral_annotation.threat import build_danger_map

    dag = GoDag.from_obo(GO_OBO_PATH)
    # Must NOT raise: every bacterial root present + non-obsolete (the assertion in
    # build_danger_map). Then check the DAG expansion produced a non-trivial map.
    dmap = build_danger_map(dag, BACTERIAL_DANGER_CATEGORIES)
    assert set(dmap) == {c.key for c in BACTERIAL_DANGER_CATEGORIES}
    all_terms = set().union(*dmap.values())
    assert len(all_terms) > 20                              # roots expanded to descendants
    assert dmap["toxin"]                                    # toxin fires for bacteria
