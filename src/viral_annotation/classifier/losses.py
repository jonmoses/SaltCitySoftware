"""Multi-label training losses, returned as **per-element** (reduction='none')
callables so the multi-task fine-tuner can mask out namespaces a protein doesn't
contribute to before reducing.

  * "bce"  — positive-weighted BCEWithLogits (the existing default; pos_weight
             counters class imbalance, see heads.compute_pos_weight).
  * "asl"  — Asymmetric Loss (Ben-Baruch et al. 2020): focuses learning on the
             rare positives and down-weights the easy negatives that dominate an
             extreme multi-label problem, with a small probability clip that lets
             very-easy negatives contribute zero gradient. Stronger than a single
             clamped pos_weight when the label matrix is very sparse.

torch is imported lazily so importing the package stays cheap without the ml extra.
"""

from __future__ import annotations


def asymmetric_loss(logits, targets, *, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
    """Element-wise ASL [B, N] (no reduction). `targets` is multi-hot {0,1}."""
    import torch

    x_sig = torch.sigmoid(logits)
    xs_pos = x_sig
    xs_neg = 1.0 - x_sig
    if clip and clip > 0:
        xs_neg = (xs_neg + clip).clamp(max=1.0)   # shift: easy negatives -> ~0 loss

    los_pos = targets * torch.log(xs_pos.clamp(min=eps))
    los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=eps))
    loss = los_pos + los_neg

    # Asymmetric focusing: (1 - p_t) ** gamma, with a larger gamma for negatives.
    pt = xs_pos * targets + xs_neg * (1.0 - targets)
    gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
    loss = loss * torch.pow(1.0 - pt, gamma)
    return -loss


def make_loss(kind: str, *, pos_weight=None, device=None,
              gamma_neg=4.0, gamma_pos=1.0, clip=0.05):
    """Return a `(logits, targets) -> per-element loss [B, N]` callable.

    `kind` is "bce" or "asl". For "bce", `pos_weight` (numpy [N]) is honoured.
    """
    import torch
    from torch import nn

    if kind == "bce":
        pw = (torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
              if pos_weight is not None else None)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        return lambda logits, targets: crit(logits, targets)
    if kind == "asl":
        return lambda logits, targets: asymmetric_loss(
            logits, targets, gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
    raise ValueError(f"unknown loss {kind!r}; choose 'bce' or 'asl'")
