"""Tests for term-vocabulary selection and multi-hot label matrices."""

from viral_annotation.data.labels import LabeledProtein
from viral_annotation.data.dataset import build_labels, select_vocab


def _p(acc, terms_all, terms_manual=None):
    terms_manual = terms_manual if terms_manual is not None else terms_all
    return LabeledProtein(
        accession=acc, sequence="M", organism="v", lineage=[],
        terms_all=frozenset(terms_all), terms_manual=frozenset(terms_manual),
        n_manual=len(terms_manual), n_iea=0,
    )


def test_select_vocab_min_count_and_root_exclusion(tiny_dag):
    # 0000003 in 3 proteins, 0000011 in 1. Roots present via propagation.
    train = [
        _p("a", {"GO:0000003", "GO:0000002", "GO:0003674"}),
        _p("b", {"GO:0000003", "GO:0000002", "GO:0003674"}),
        _p("c", {"GO:0000003", "GO:0000002", "GO:0003674", "GO:0000011", "GO:0008150"}),
    ]
    vocab = select_vocab(train, tiny_dag, min_count=2)
    # 0003 (3x) and 0002 (3x) kept; 0011 (1x) dropped; roots excluded.
    assert set(vocab.terms) == {"GO:0000003", "GO:0000002"}
    assert "GO:0003674" not in vocab.terms
    assert "GO:0008150" not in vocab.terms
    # Namespaces aligned.
    assert all(ns == "molecular_function" for ns in vocab.namespaces)


def test_columns_by_namespace(tiny_dag):
    train = [
        _p("a", {"GO:0000003", "GO:0000011", "GO:0000021"}),
        _p("b", {"GO:0000003", "GO:0000011", "GO:0000021"}),
    ]
    vocab = select_vocab(train, tiny_dag, min_count=2)
    cols = vocab.columns_by_namespace()
    assert set(cols) == {"molecular_function", "biological_process", "cellular_component"}
    assert sum(len(v) for v in cols.values()) == len(vocab)


def test_select_vocab_field_and_namespace_filter(tiny_dag):
    # Propagated labels (ancestors included, roots stripped by select_vocab).
    # terms_all has an IEA-only MF term (0004); terms_manual does not.
    train = [
        _p("a", terms_all={"GO:0000003", "GO:0000004", "GO:0000002", "GO:0000011"},
           terms_manual={"GO:0000003", "GO:0000002", "GO:0000011"}),
        _p("b", terms_all={"GO:0000003", "GO:0000004", "GO:0000002", "GO:0000011"},
           terms_manual={"GO:0000003", "GO:0000002", "GO:0000011"}),
    ]
    # MF only, from manual labels: 0004 (IEA-only) must be excluded; 0011 is BP.
    mf_manual = select_vocab(train, tiny_dag, min_count=2,
                             field="terms_manual", namespaces=["molecular_function"])
    assert set(mf_manual.terms) == {"GO:0000003", "GO:0000002"}
    assert all(ns == "molecular_function" for ns in mf_manual.namespaces)

    # MF only, from terms_all: now the IEA term 0004 is included.
    mf_all = select_vocab(train, tiny_dag, min_count=2,
                          field="terms_all", namespaces=["molecular_function"])
    assert "GO:0000004" in mf_all.terms

    # BP only stays BP.
    bp = select_vocab(train, tiny_dag, min_count=2,
                      field="terms_all", namespaces=["biological_process"])
    assert set(bp.terms) == {"GO:0000011"}


def test_build_labels_uses_correct_field(tiny_dag):
    train = [
        _p("a", terms_all={"GO:0000003", "GO:0000004"}, terms_manual={"GO:0000003"}),
        _p("b", terms_all={"GO:0000003", "GO:0000004"}, terms_manual={"GO:0000003"}),
    ]
    vocab = select_vocab(train, tiny_dag, min_count=2)
    Y_all = build_labels(train, vocab, "terms_all")
    Y_manual = build_labels(train, vocab, "terms_manual")
    assert Y_all.shape == (2, len(vocab))
    # terms_all marks both 0003 and 0004; terms_manual marks only 0003.
    c3, c4 = vocab.index["GO:0000003"], vocab.index["GO:0000004"]
    assert Y_all[0, c3] == 1.0 and Y_all[0, c4] == 1.0
    assert Y_manual[0, c3] == 1.0 and Y_manual[0, c4] == 0.0
