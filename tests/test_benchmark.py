"""Tests for benchmark metrics (M-AUPR, IA, Smin) and the temporal NK split."""

import numpy as np

from viral_annotation.data.quickgo import ExpAnnotation
from viral_annotation.benchmark.temporal import build_temporal_split
from viral_annotation.evaluation.metrics import information_accretion, m_aupr, smin


def test_m_aupr_perfect_ranking():
    # Each term's positives are ranked above negatives -> AUPR 1 per term -> mean 1.
    true = np.array([[1, 0], [0, 1], [1, 0]], dtype="float32")
    prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]], dtype="float32")
    assert m_aupr(prob, true) == 1.0


def test_information_accretion(tiny_dag):
    # 0003 is_a 0002 is_a root(0003674). 2 proteins carry 0003 (so 0002,root too);
    # 2 carry only 0002 (and root). IA(0003) = -log2( count(0003)/count(0002) )
    # = -log2(2/4) = 1.0 ; IA(0002) = -log2(4/4) = 0.
    sets = [
        {"GO:0000003", "GO:0000002", "GO:0003674"},
        {"GO:0000003", "GO:0000002", "GO:0003674"},
        {"GO:0000002", "GO:0003674"},
        {"GO:0000002", "GO:0003674"},
    ]
    ia = information_accretion(sets, tiny_dag)
    assert ia["GO:0000003"] == 1.0
    assert ia["GO:0000002"] == 0.0


def test_smin_zero_for_perfect_prediction(tiny_dag):
    terms = ["GO:0000003", "GO:0000002"]
    ia = {"GO:0000003": 1.0, "GO:0000002": 0.5}
    true = np.array([[1, 1], [0, 1]], dtype="float32")
    prob = np.array([[0.9, 0.9], [0.1, 0.9]], dtype="float32")  # threshold ~0.5 perfect
    assert smin(prob, true, ia, terms) == 0.0


def test_temporal_split_no_knowledge_selection(tiny_dag):
    cutoff = 20240101
    ann = [
        ExpAnnotation("A", "GO:0000003", "molecular_function", "EXP", 20230101),  # before -> train MF
        ExpAnnotation("B", "GO:0000003", "molecular_function", "EXP", 20250101),  # after only -> test MF (NK)
        ExpAnnotation("C", "GO:0000011", "biological_process", "EXP", 20230101),  # before -> train BP
        ExpAnnotation("C", "GO:0000021", "cellular_component", "EXP", 20250101),  # after -> test CC (NK)
        ExpAnnotation("Z", "GO:0000003", "molecular_function", "EXP", 20250101),  # no sequence -> dropped
    ]
    seq_by_acc = {"A": "M", "B": "M", "C": "M"}
    s = build_temporal_split(ann, seq_by_acc, tiny_dag, cutoff)

    assert [p.accession for p in s.train["molecular_function"]] == ["A"]
    assert [p.accession for p in s.test["molecular_function"]] == ["B"]
    assert [p.accession for p in s.train["biological_process"]] == ["C"]
    assert [p.accession for p in s.test["cellular_component"]] == ["C"]
    # propagated train label for A includes ancestors.
    assert s.train["molecular_function"][0].terms_manual == {"GO:0000003", "GO:0000002", "GO:0003674"}
    # protein Z has no sequence -> excluded entirely.
    assert all(p.accession != "Z" for p in s.test["molecular_function"])
