"""Multi-label GO-term classifier head (docs/01 Step 2).

Maps a pooled [d] protein embedding to N GO-term logits, one per term, with a
**sigmoid** (not softmax) output — each term is an independent binary question.

Start LINEAR (single layer); NetGO 3.0's LR-ESM shows logistic regression on pLM
embeddings is competitive. Add hidden layers only if they earn measurable Fmax
(docs/01). `hidden_dims=[]` gives the linear baseline; e.g. [512] gives a 1-layer MLP.

The loss (positive-weighted BCE / focal) and training loop live with the training
config, NOT here — those depend on the label set and class balance, which are
empirical. This module is just the architecture. torch is imported lazily.
"""

from __future__ import annotations


def build_classifier(input_dim: int, num_terms: int, hidden_dims=None, dropout: float = 0.0):
    """Construct the multi-label head as a torch.nn.Module.

    Args:
        input_dim: embedding width d (e.g. 1280 for ESM-2 650M).
        num_terms: N, the number of GO terms in the scoped label set.
        hidden_dims: list of hidden widths; [] or None => linear baseline.
        dropout: dropout prob applied before each linear layer in the MLP case.

    Returns:
        torch.nn.Module producing raw logits of shape [B, num_terms]. Apply
        sigmoid at inference (BCEWithLogitsLoss applies it internally at train).
    """
    import torch.nn as nn

    hidden_dims = list(hidden_dims or [])
    layers: list = []
    prev = input_dim
    for h in hidden_dims:
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, num_terms))  # final logits; sigmoid applied later
    return nn.Sequential(*layers)
