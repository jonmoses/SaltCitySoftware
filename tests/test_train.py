"""Tests for training-critical math: vectorized Fmax, correction, predict_proba.

These avoid the network and the ESM model — they exercise the numpy/torch pieces
the training loop depends on.
"""

import numpy as np

from viral_annotation.classifier.model import build_classifier, predict_proba
from viral_annotation.data.dataset import build_labels, select_vocab
from viral_annotation.data.labels import LabeledProtein
from viral_annotation.evaluation.metrics import (
    apply_hierarchical_correction,
    fmax,
    fmax_matrix,
)


def test_fmax_matrix_matches_set_based_fmax():
    # Two proteins, three terms. Compare vectorized vs set-based on identical data.
    terms = ["GO:1", "GO:2", "GO:3"]
    prob = np.array([[0.9, 0.2, 0.7], [0.1, 0.8, 0.3]], dtype="float32")
    true = np.array([[1, 0, 1], [0, 1, 0]], dtype="float32")

    mat = fmax_matrix(prob, true)

    preds = [{terms[j]: float(prob[i, j]) for j in range(3)} for i in range(2)]
    truth = [{terms[j] for j in range(3) if true[i, j] > 0} for i in range(2)]
    ref = fmax(preds, truth)

    assert mat.fmax == np_almost(ref.fmax)
    assert mat.precision == np_almost(ref.precision)
    assert mat.recall == np_almost(ref.recall)


def test_fmax_matrix_perfect():
    prob = np.array([[0.99, 0.01], [0.02, 0.97]], dtype="float32")
    true = np.array([[1, 0], [0, 1]], dtype="float32")
    assert fmax_matrix(prob, true).fmax == np_almost(1.0)


def test_predict_proba_shape_and_range():
    model = build_classifier(input_dim=8, num_terms=5, hidden_dims=[])
    X = np.random.RandomState(0).randn(7, 8).astype("float32")
    P = predict_proba(model, X)
    assert P.shape == (7, 5)
    assert P.min() >= 0.0 and P.max() <= 1.0


def test_apply_hierarchical_correction_enforces_parent_ge_child(tiny_dag):
    # Vocab over the MF chain; child more confident than its ancestors.
    train = [
        LabeledProtein("a", "M", "v", [],
                       frozenset({"GO:0000003", "GO:0000002"}),
                       frozenset({"GO:0000003", "GO:0000002"}), 2, 0),
        LabeledProtein("b", "M", "v", [],
                       frozenset({"GO:0000003", "GO:0000002"}),
                       frozenset({"GO:0000003", "GO:0000002"}), 2, 0),
    ]
    vocab = select_vocab(train, tiny_dag, min_count=2)
    c_child = vocab.index["GO:0000003"]
    c_parent = vocab.index["GO:0000002"]
    prob = np.zeros((1, len(vocab)), dtype="float32")
    prob[0, c_child] = 0.9
    prob[0, c_parent] = 0.3
    fixed = apply_hierarchical_correction(prob, vocab, tiny_dag)
    assert fixed[0, c_parent] >= fixed[0, c_child] - 1e-6
    assert fixed[0, c_parent] == np_almost(0.9)


def np_almost(x, tol=1e-5):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) <= tol
    return _Approx()
