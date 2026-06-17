"""Multi-head attention pooling + classifier head.

Instead of a fixed reduction (mean/max/...), the model LEARNS which residues
matter: each of H attention heads has a learned query, produces a softmax over
the protein's residues, and returns its own weighted-sum vector. The H pooled
vectors are concatenated (H*d) and fed to a linear/MLP head. Different heads can
specialize (catalytic site, sorting signal, ...) — more expressive than mean while
still emitting a fixed-size representation. torch is imported lazily.
"""

from __future__ import annotations


def build_attn_pool(input_dim: int, n_heads: int = 8):
    """Reusable multi-head attention pooler: [B,L,d] + mask [B,L] -> [B, H*d].

    Just the learned reduction (no classifier head), so it can sit on top of
    either cached per-residue embeddings (frozen path) or a live backbone's
    hidden states (the LoRA fine-tune path). `out_dim` is H*d. torch is imported
    lazily, matching the rest of the package.
    """
    import torch
    import torch.nn as nn

    class AttnPool(nn.Module):
        out_dim = n_heads * input_dim

        def __init__(self):
            super().__init__()
            # H learned query vectors; small init keeps early attention near-uniform.
            self.query = nn.Parameter(torch.randn(n_heads, input_dim) * input_dim ** -0.5)
            self.scale = input_dim ** -0.5

        def forward(self, X, mask):
            # X: [B, L, d]; mask: [B, L] with 1 for real residues, 0 for padding.
            scores = torch.einsum("bld,hd->blh", X, self.query) * self.scale  # [B,L,H]
            pad = (mask == 0).unsqueeze(-1)                                    # [B,L,1]
            scores = scores.masked_fill(pad, float("-inf"))
            attn = torch.softmax(scores, dim=1)                               # over residues
            pooled = torch.einsum("blh,bld->bhd", attn, X)                    # [B,H,d]
            return pooled.reshape(X.shape[0], -1)                            # [B, H*d]

    return AttnPool()


def build_attn_classifier(input_dim: int, num_terms: int, n_heads: int = 8,
                          hidden_dims=None, dropout: float = 0.0):
    """AttnPoolClassifier: per-residue X [B,L,d] + mask [B,L] -> logits [B,num_terms]."""
    import torch.nn as nn

    from viral_annotation.classifier.model import build_classifier

    class AttnPoolClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = build_attn_pool(input_dim, n_heads=n_heads)
            self.head = build_classifier(self.pool.out_dim, num_terms,
                                         hidden_dims=hidden_dims, dropout=dropout)

        def forward(self, X, mask):
            return self.head(self.pool(X, mask))                             # [B,num_terms]

    return AttnPoolClassifier()
