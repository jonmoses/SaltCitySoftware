"""Protein-centric Fmax, the CAFA-standard AFP metric (docs/03).

Fmax = max over a threshold sweep of the harmonic mean of the *averaged*
precision and recall, computed per protein then averaged across proteins. Per
the CAFA convention:

  * at threshold t, precision is averaged ONLY over proteins with >= 1 predicted
    term at t (proteins with no prediction are undefined, not zero),
  * recall is averaged over ALL proteins.

Inputs are expected to be already true-path propagated (both predictions and
ground truth) — see GoDag.propagate / GoDag.correct_scores.

Pure stdlib so it runs and is tested without numpy/sklearn. AUPR and Smin (which
needs Information Accretion values) are TODO — see docs/03.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass
class FmaxResult:
    fmax: float
    threshold: float          # tau achieving Fmax
    precision: float          # precision at that tau
    recall: float             # recall at that tau


def fmax(
    predictions: Sequence[Mapping[str, float]],
    ground_truth: Sequence[set[str]],
    thresholds: Sequence[float] | None = None,
) -> FmaxResult:
    """Compute protein-centric Fmax.

    Args:
        predictions: per protein, a mapping {GO term -> probability in [0, 1]}.
        ground_truth: per protein, the set of true (propagated) GO terms.
        thresholds: tau values to sweep; default is 0.01..1.00 in 0.01 steps.

    Returns:
        FmaxResult with the best F1 and the tau / precision / recall there.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truth "
            f"({len(ground_truth)}) must align per protein"
        )
    if thresholds is None:
        thresholds = [i / 100 for i in range(1, 101)]

    n_proteins = len(ground_truth)
    best = FmaxResult(fmax=0.0, threshold=0.0, precision=0.0, recall=0.0)

    for tau in thresholds:
        prec_sum = 0.0
        n_predicted = 0      # proteins with >=1 prediction at tau (precision denom)
        recall_sum = 0.0

        for pred, truth in zip(predictions, ground_truth):
            called = {term for term, p in pred.items() if p >= tau}
            tp = len(called & truth)

            if called:
                prec_sum += tp / len(called)
                n_predicted += 1
            if truth:
                recall_sum += tp / len(truth)
            # proteins with empty truth contribute 0 recall but still count in
            # the denominator (n_proteins), matching CAFA's protein-centric avg.

        if n_predicted == 0:
            continue
        precision = prec_sum / n_predicted
        recall = recall_sum / n_proteins
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best.fmax:
            best = FmaxResult(fmax=f1, threshold=tau, precision=precision, recall=recall)

    return best
