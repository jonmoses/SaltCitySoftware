"""Late-fusion ensemble of per-term score matrices (pLM + homology + InterPro).

Each component produces a [P x N] score matrix over the same namespace vocab. We
fuse by a weighted sum whose weights are grid-searched on the validation set's
Fmax — per namespace, so e.g. InterPro can get ~0 weight for MF (where it doesn't
help) and more for BP/CC. The pLM weight is fixed at 1.0 and the others are scaled
relative to it (Fmax sweeps the threshold, so absolute scale is absorbed).
"""

from __future__ import annotations

from itertools import product

from viral_annotation.evaluation.metrics import fmax_matrix

WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)


def fuse(components: dict, weights: dict):
    """Weighted sum of component matrices -> [P x N]."""
    import numpy as np

    keys = list(components)
    out = np.zeros_like(np.asarray(components[keys[0]], dtype="float32"))
    for k in keys:
        out = out + float(weights.get(k, 0.0)) * np.asarray(components[k], dtype="float32")
    return out


def search_weights(val_components: dict, val_true, base: str = "plm",
                   grid=WEIGHT_GRID) -> tuple[dict, float]:
    """Grid-search component weights to maximize validation Fmax.

    `base` (pLM) is pinned at weight 1.0; the others sweep `grid`. Returns
    (best_weights, best_val_fmax).
    """
    others = [k for k in val_components if k != base]
    best_w = {base: 1.0, **{o: 0.0 for o in others}}
    best_f = fmax_matrix(val_components[base], val_true).fmax
    for combo in product(grid, repeat=len(others)):
        w = {base: 1.0, **dict(zip(others, combo))}
        f = fmax_matrix(fuse(val_components, w), val_true).fmax
        if f > best_f:
            best_f, best_w = f, w
    return best_w, best_f
