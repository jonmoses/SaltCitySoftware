"""Tests for protein-centric Fmax."""

import pytest

from viral_annotation.evaluation.metrics import fmax


def test_perfect_prediction_gives_fmax_one():
    preds = [{"GO:1": 0.99, "GO:2": 0.95}, {"GO:3": 0.9}]
    truth = [{"GO:1", "GO:2"}, {"GO:3"}]
    res = fmax(preds, truth)
    assert res.fmax == pytest.approx(1.0)
    assert res.precision == pytest.approx(1.0)
    assert res.recall == pytest.approx(1.0)


def test_threshold_separates_good_from_bad_calls():
    # A true term scored high, a false term scored low. A threshold between them
    # recovers precision=recall=1 -> Fmax=1.
    preds = [{"GO:1": 0.8, "GO:2": 0.1}]
    truth = [{"GO:1"}]
    res = fmax(preds, truth)
    assert res.fmax == pytest.approx(1.0)
    assert 0.1 < res.threshold <= 0.8


def test_half_recall():
    # Predict one of two true terms, nothing false -> precision 1, recall 0.5.
    preds = [{"GO:1": 0.9}]
    truth = [{"GO:1", "GO:2"}]
    res = fmax(preds, truth)
    # F1 = 2 * 1 * 0.5 / (1 + 0.5) = 0.6667
    assert res.fmax == pytest.approx(2 / 3, abs=1e-6)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        fmax([{"GO:1": 0.9}], [{"GO:1"}, {"GO:2"}])


def test_no_predictions_above_threshold_gives_zero():
    preds = [{"GO:1": 0.0}]
    truth = [{"GO:1"}]
    # With all probs 0 and thresholds starting at 0.01, nothing is ever called.
    res = fmax(preds, truth)
    assert res.fmax == pytest.approx(0.0)
