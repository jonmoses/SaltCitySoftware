"""Tests for non-overlapping windowing + length-weighted regroup.

The regroup is exactly the "how do windows get attributed back to the parent
protein" logic, tested here without loading ESM by stubbing the flat embedder.
"""

import numpy as np

from viral_annotation.embeddings.esm import ESMEmbedder, _finalize_stats, _windows


def test_windows_non_overlapping():
    assert _windows("ABCDEFG", 3) == ["ABC", "DEF", "G"]
    assert _windows("ABC", 5) == ["ABC"]      # short -> single window
    assert _windows("ABCDEF", 3) == ["ABC", "DEF"]
    assert _windows("", 3) == [""]            # empty -> one empty window


def _stub(embedder, monkeypatch):
    """Make embed() runnable without a real model: skip load, fake flat embed."""
    embedder._model = object()
    embedder._tokenizer = object()
    embedder._device = "cpu"

    # Each window -> 2-d vector [len(window), 1.0]; lets us check length-weighting.
    def fake_flat(seqs, batch_size=None):
        return np.array([[float(len(s)), 1.0] for s in seqs], dtype="float32")

    monkeypatch.setattr(embedder, "_embed_flat", fake_flat)


def test_window_regroup_is_length_weighted(monkeypatch):
    e = ESMEmbedder(window=True)
    e.max_length = 3
    _stub(e, monkeypatch)

    out = e.embed(["ABCDEFG", "XY"])
    assert out.shape == (2, 2)
    # "ABCDEFG" -> windows of len 3,3,1. Length-weighted mean of [3,3,1] vals:
    # dim0 = (3*3 + 3*3 + 1*1) / 7 = 19/7 ; dim1 = 1.0
    assert out[0, 0] == np.float32(19 / 7)
    assert out[0, 1] == np.float32(1.0)
    # "XY" -> one window of len 2 -> [2, 1]
    assert out[1, 0] == np.float32(2.0)


def test_finalize_stats_combines_across_windows():
    # One protein, two windows. Stats must combine over ALL residues, with max/min
    # taken ACROSS windows (not averaged), and std exact.
    owner = [0, 0]
    w_n = np.array([2.0, 1.0])
    w_sum = np.array([[2.0, 0.0], [3.0, 0.0]])   # totals: [5, 0]
    w_sq = np.array([[2.0, 0.0], [9.0, 0.0]])    # totals: [11, 0]
    w_max = np.array([[1.0, 0.0], [3.0, 0.0]])
    w_min = np.array([[1.0, 0.0], [3.0, 0.0]])

    out = _finalize_stats(owner, 1, w_n, w_sum, w_sq, w_max, w_min)
    assert out.shape == (1, 8)  # 4 stats x d(=2)
    mean, mx, mn, std = out[0, :2], out[0, 2:4], out[0, 4:6], out[0, 6:8]
    assert np.allclose(mean, [5 / 3, 0])           # exact over all residues
    assert np.allclose(mx, [3.0, 0])               # max ACROSS windows (not 2 = avg)
    assert np.allclose(mn, [1.0, 0])               # min ACROSS windows
    assert np.allclose(std, [np.sqrt(8 / 9), 0], atol=1e-6)  # exact, not per-window avg


def test_one_vector_per_protein_regardless_of_window_count(monkeypatch):
    e = ESMEmbedder(window=True)
    e.max_length = 10
    _stub(e, monkeypatch)
    seqs = ["A" * 35, "A" * 5, "A" * 100]  # 4, 1, 10 windows respectively
    out = e.embed(seqs)
    assert out.shape == (3, 2)              # exactly one row per input protein
    # A short protein's vector equals its single-window embedding.
    assert out[1, 0] == np.float32(5.0)
