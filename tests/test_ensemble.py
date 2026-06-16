"""Tests for homology label-transfer math and ensemble weight search."""

import numpy as np

from viral_annotation.classifier.ensemble import fuse, search_weights
from viral_annotation.data.homology import _aggregate


def test_homology_bitscore_weighted_transfer():
    # query q1 has two hits: t1 (bits 30, term col 0) and t2 (bits 10, terms 0 & 1).
    # term 0: (30+10)/40 = 1.0 ; term 1: 10/40 = 0.25.
    hits = {"q1": [("t1", 30.0), ("t2", 10.0)], "q2": []}
    db_term_cols = {"t1": [0], "t2": [0, 1]}
    S = _aggregate(["q1", "q2"], hits, db_term_cols, n_vocab=2)
    assert S.shape == (2, 2)
    assert np.allclose(S[0], [1.0, 0.25])
    assert np.allclose(S[1], [0.0, 0.0])   # no hits -> zeros


def test_fuse_weighted_sum():
    comps = {"plm": np.array([[0.5]]), "homology": np.array([[0.2]]),
             "interpro": np.array([[1.0]])}
    out = fuse(comps, {"plm": 1.0, "homology": 0.5, "interpro": 2.0})
    assert np.allclose(out, [[0.5 + 0.1 + 2.0]])


def test_search_weights_prefers_informative_component():
    # plm is pure noise vs the labels; homology equals the labels. The search
    # should upweight homology and beat the plm-only Fmax.
    true = np.array([[1, 0, 1], [0, 1, 0]], dtype="float32")
    plm = np.array([[0.1, 0.9, 0.1], [0.9, 0.1, 0.9]], dtype="float32")  # anti-correlated
    homology = true.astype("float32")
    interpro = np.zeros_like(true)
    comps = {"plm": plm, "homology": homology, "interpro": interpro}
    weights, fbest = search_weights(comps, true)
    assert weights["homology"] > 0          # homology gets weight
    # fused beats plm-only
    plm_f = search_weights({"plm": plm}, true)[1]
    assert fbest >= plm_f
