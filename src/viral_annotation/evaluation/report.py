"""One renderer for the per-namespace + overall Fmax-vs-naive result tables.

Every trainer and the benchmark used to hand-roll this same table; it now lives
here so "model vs naive, per namespace and overall" prints identically everywhere.
"""

from __future__ import annotations

import numpy as np

from viral_annotation.config import GO_NAMESPACES
from viral_annotation.evaluation.metrics import fmax_matrix

NS_SHORT = {
    "molecular_function": "MF",
    "biological_process": "BP",
    "cellular_component": "CC",
}


def overall_fmax(prob_parts, true_parts):
    """Fmax over all namespaces at once: namespaces own disjoint term columns, so
    concatenate the [P x N_ns] blocks column-wise and score the whole matrix."""
    true = np.concatenate(true_parts, axis=1)
    return fmax_matrix(np.concatenate(prob_parts, axis=1), true)


def print_table(title, rows, overall, overall_naive):
    """Print a titled Fmax table.

    Args:
        rows: list of (namespace, model_FmaxResult, naive_FmaxResult), in display order.
        overall, overall_naive: the across-namespace FmaxResults.
    """
    print(f"\n=== {title} ===")
    print(f"  {'namespace':20s} {'Fmax':>7s}  {'naive':>6s}  {'lift':>6s}")
    for ns, res, naive in rows:
        print(f"  {ns:20s} {res.fmax:7.4f}  {naive.fmax:6.4f}  {res.fmax - naive.fmax:+6.4f}  "
              f"(P={res.precision:.3f} R={res.recall:.3f} N={res.n_terms})")
    print(f"  {'overall':20s} {overall.fmax:7.4f}  {overall_naive.fmax:6.4f}  "
          f"{overall.fmax - overall_naive.fmax:+6.4f}  (N={overall.n_terms})")


def namespace_order(present):
    """Canonical namespace order, filtered to those actually present (a dict/set)."""
    return [ns for ns in GO_NAMESPACES if ns in present]
