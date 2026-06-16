"""Tests for multi-head attention pooling + the variable-length collate."""

import numpy as np


def test_attn_pool_output_shape():
    import torch

    from viral_annotation.classifier.pooling import build_attn_classifier

    m = build_attn_classifier(input_dim=8, num_terms=5, n_heads=4)
    out = m(torch.randn(3, 7, 8), torch.ones(3, 7))
    assert out.shape == (3, 5)


def test_attn_pool_ignores_padding():
    # Padded residues — even with garbage values — must not change the output,
    # because the mask sends their attention scores to -inf.
    import torch

    from viral_annotation.classifier.pooling import build_attn_classifier

    torch.manual_seed(0)
    m = build_attn_classifier(input_dim=6, num_terms=3, n_heads=2).eval()
    X = torch.randn(1, 4, 6)
    X_padded = torch.cat([X, torch.randn(1, 3, 6)], dim=1)          # 3 junk residues
    mask_padded = torch.tensor([[1.0, 1, 1, 1, 0, 0, 0]])
    with torch.no_grad():
        out_exact = m(X, torch.ones(1, 4))
        out_padded = m(X_padded, mask_padded)
    assert torch.allclose(out_exact, out_padded, atol=1e-5)


def test_collate_pads_and_masks():
    from viral_annotation.training.train_attn import _collate

    batch = [
        (np.ones((2, 3), dtype="float32"), np.array([1, 0], dtype="float32")),
        (np.full((4, 3), 2.0, dtype="float32"), np.array([0, 1], dtype="float32")),
    ]
    X, mask, Y = _collate(batch)
    assert tuple(X.shape) == (2, 4, 3)           # padded to the longest (4)
    assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert tuple(Y.shape) == (2, 2)
