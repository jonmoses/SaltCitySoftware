"""Tests for the embedding head-to-head comparison: the paired bootstrap CI and
that the harness wires together. The bootstrap tests are numpy-only; the harness
import test is gated on torch (the ml extra)."""

import numpy as np
import pytest

from viral_annotation.evaluation.metrics import fmax_matrix, paired_bootstrap_delta


def _fmax(p, y):
    return fmax_matrix(p, y).fmax


def test_bootstrap_significant_when_b_dominates():
    # 40 proteins, 4 terms. Model B is ~perfect; model A is non-discriminative
    # (constant 0.5). B should beat A in every resample -> delta>0 and CI lower
    # bound strictly above 0 (significant).
    rng = np.random.RandomState(0)
    Y = (rng.rand(40, 4) > 0.5).astype("float32")
    prob_b = Y * 0.9 + 0.05            # perfectly ranked
    prob_a = np.full_like(Y, 0.5)      # no signal
    obs, lo, hi = paired_bootstrap_delta(prob_a, prob_b, Y, _fmax, n_boot=300, seed=1)
    assert obs > 0
    assert lo > 0                      # CI excludes 0 -> significant gain
    assert lo <= obs <= hi


def test_bootstrap_brackets_zero_when_identical():
    rng = np.random.RandomState(1)
    Y = (rng.rand(30, 3) > 0.5).astype("float32")
    prob = rng.rand(30, 3).astype("float32")
    obs, lo, hi = paired_bootstrap_delta(prob, prob, Y, _fmax, n_boot=200, seed=2)
    assert obs == 0.0
    assert lo <= 0.0 <= hi             # no difference -> CI contains 0


def test_bootstrap_disabled_returns_point_estimate():
    Y = np.array([[1, 0], [0, 1]], dtype="float32")
    a = np.array([[0.6, 0.4], [0.4, 0.6]], dtype="float32")
    b = np.array([[0.9, 0.1], [0.1, 0.9]], dtype="float32")
    obs, lo, hi = paired_bootstrap_delta(a, b, Y, _fmax, n_boot=0)
    assert lo == obs == hi


def test_harness_importable_and_callable():
    pytest.importorskip("torch")
    from viral_annotation.benchmark.compare import run_comparison, _resolve_seeds

    assert callable(run_comparison)
    # default seed budgets: more for viral than the heavier bacterial fits.
    assert len(_resolve_seeds("viral", None)) == 5
    assert len(_resolve_seeds("bacterial", None)) == 3
    assert len(_resolve_seeds("viral", 2)) == 2
