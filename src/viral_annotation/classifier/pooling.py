"""Multi-head attention pooling + classifier head.

Instead of a fixed reduction (mean/max/...), the model LEARNS which residues
matter: each of H attention heads has a learned query, produces a softmax over
the protein's residues, and returns its own weighted-sum vector. The H pooled
vectors are concatenated (H*d) and fed to a linear/MLP head. Different heads can
specialize (catalytic site, sorting signal, ...) — more expressive than mean while
still emitting a fixed-size representation. torch is imported lazily.
"""

from __future__ import annotations


def build_attn_classifier(input_dim: int, num_terms: int, n_heads: int = 8,
                          hidden_dims=None, dropout: float = 0.0):
    """AttnPoolClassifier: per-residue X [B,L,d] + mask [B,L] -> logits [B,num_terms]."""
    import torch
    import torch.nn as nn

    from viral_annotation.classifier.model import build_classifier

    class AttnPoolClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            # H learned query vectors; small init keeps early attention near-uniform.
            self.query = nn.Parameter(torch.randn(n_heads, input_dim) * input_dim ** -0.5)
            self.scale = input_dim ** -0.5
            self.head = build_classifier(n_heads * input_dim, num_terms,
                                         hidden_dims=hidden_dims, dropout=dropout)

        def forward(self, X, mask):
            # X: [B, L, d]; mask: [B, L] with 1 for real residues, 0 for padding.
            scores = torch.einsum("bld,hd->blh", X, self.query) * self.scale  # [B,L,H]
            pad = (mask == 0).unsqueeze(-1)                                    # [B,L,1]
            scores = scores.masked_fill(pad, float("-inf"))
            attn = torch.softmax(scores, dim=1)                               # over residues
            pooled = torch.einsum("blh,bld->bhd", attn, X)                    # [B,H,d]
            return self.head(pooled.reshape(X.shape[0], -1))                  # [B,num_terms]

    return AttnPoolClassifier()
