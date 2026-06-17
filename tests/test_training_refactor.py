"""Tests for the consolidated trainer's pure helpers — the pieces that don't need
the network or ESM: pooling resolution, the shared report's overall metric, and the
annotation-stats summary."""

from types import SimpleNamespace

import numpy as np

from viral_annotation.config import BACTERIAL_NAMESPACE_POLICY, NAMESPACE_POLICY
from viral_annotation.evaluation.report import overall_fmax
from viral_annotation.training.pipeline import annotation_stats
from viral_annotation.training.train import _pooling_per_namespace


def test_pooling_uniform():
    # A concrete pooling choice applies to every namespace (policy irrelevant here).
    assert _pooling_per_namespace("mean", NAMESPACE_POLICY) == {
        "molecular_function": "mean",
        "biological_process": "mean",
        "cellular_component": "mean",
    }


def test_pooling_per_namespace_reads_policy():
    # "per-namespace" defers to the given policy (viral: attention MF, mean BP/CC).
    resolved = _pooling_per_namespace("per-namespace", NAMESPACE_POLICY)
    assert resolved == {ns: NAMESPACE_POLICY[ns]["pooling"] for ns in resolved}
    assert resolved["molecular_function"] == "attention"
    assert resolved["biological_process"] == "mean"


def test_pooling_per_namespace_bacterial_is_attention_mf():
    # The bacterial profile pools attention for MF (localized catalytic signal,
    # reached via the LoRA fine-tune path) and mean for BP/CC.
    resolved = _pooling_per_namespace("per-namespace", BACTERIAL_NAMESPACE_POLICY)
    assert resolved["molecular_function"] == "attention"
    assert resolved["biological_process"] == "mean"
    assert resolved["cellular_component"] == "mean"


def test_overall_fmax_concatenates_namespace_blocks():
    # Two namespaces, each [2 proteins x 1 term]; concatenated -> a perfect [2 x 2]
    # ranking so Fmax is 1.0.
    prob_parts = [np.array([[0.9], [0.1]], "float32"), np.array([[0.2], [0.8]], "float32")]
    true_parts = [np.array([[1], [0]], "float32"), np.array([[0], [1]], "float32")]
    res = overall_fmax(prob_parts, true_parts)
    assert res.fmax == 1.0
    assert res.n_terms == 2


def test_annotation_stats_counts():
    proteins = [
        SimpleNamespace(n_manual=2, n_iea=1, has_manual=True),
        SimpleNamespace(n_manual=0, n_iea=5, has_manual=False),
    ]
    s = annotation_stats(proteins)
    assert "2 proteins" in s
    assert "manual-having 1" in s
    assert "manual=2" in s and "iea=6" in s
