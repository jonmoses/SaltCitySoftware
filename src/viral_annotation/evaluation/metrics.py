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
    n_terms: int = 0          # size of the term set scored (for context)
    n_proteins: int = 0       # proteins scored


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
            best = FmaxResult(
                fmax=f1, threshold=tau, precision=precision, recall=recall,
                n_proteins=n_proteins,
            )

    return best


def fmax_matrix(prob_matrix, true_matrix, thresholds=None) -> FmaxResult:
    """Vectorized protein-centric Fmax over matrices [P x N] (numpy).

    Same CAFA convention as `fmax` (precision averaged only over proteins with
    >=1 prediction at tau; recall averaged over all proteins) but fast enough to
    call every training epoch. Use this for matrix inputs; `fmax` for the
    set-of-terms API.
    """
    import numpy as np

    prob = np.asarray(prob_matrix, dtype="float32")
    true_bool = np.asarray(true_matrix) > 0
    P = prob.shape[0]
    true_count = true_bool.sum(axis=1)
    if thresholds is None:
        thresholds = np.linspace(0.01, 1.0, 100)

    best = FmaxResult(0.0, 0.0, 0.0, 0.0, n_terms=prob.shape[1], n_proteins=P)
    for tau in thresholds:
        pred = prob >= tau
        pred_count = pred.sum(axis=1)
        tp = (pred & true_bool).sum(axis=1)
        has_pred = pred_count > 0
        if not has_pred.any():
            continue
        precision = float((tp[has_pred] / pred_count[has_pred]).mean())
        rec_per = np.zeros(P, dtype="float64")
        nz = true_count > 0
        rec_per[nz] = tp[nz] / true_count[nz]
        recall = float(rec_per.mean())
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best.fmax:
            best = FmaxResult(
                fmax=f1, threshold=float(tau), precision=precision, recall=recall,
                n_terms=prob.shape[1], n_proteins=P,
            )
    return best


def apply_hierarchical_correction(prob_matrix, vocab, dag):
    """Post-hoc true-path correction of a probability matrix [P x N].

    For each protein, a parent term's probability is lifted to at least its most
    likely descendant's (GoDag.correct_scores), so predictions never violate the
    DAG before scoring (docs/01 Step 4). Returns a corrected copy.
    """
    import numpy as np

    corrected = np.array(prob_matrix, dtype="float32", copy=True)
    terms = vocab.terms
    for row in range(corrected.shape[0]):
        scores = {terms[c]: float(corrected[row, c]) for c in range(len(terms))}
        fixed = dag.correct_scores(scores)
        for c, t in enumerate(terms):
            corrected[row, c] = fixed[t]
    return corrected


def fmax_by_namespace(prob_matrix, true_matrix, vocab) -> dict[str, FmaxResult]:
    """Per-namespace + micro-overall Fmax over a prediction matrix.

    Args:
        prob_matrix: [P x N] predicted probabilities (already hierarchically
                     corrected, ideally).
        true_matrix: [P x N] multi-hot ground truth.
        vocab: TermVocab giving the term id and namespace of each column.

    Returns:
        dict mapping each GO namespace present in the vocab (plus "overall") to a
        FmaxResult. Reuses the protein-centric `fmax` above per term subset.
    """
    import numpy as np

    prob = np.asarray(prob_matrix)
    true = np.asarray(true_matrix)
    cols_by_ns = vocab.columns_by_namespace()
    results: dict[str, FmaxResult] = {}
    for ns, cols in cols_by_ns.items():
        results[ns] = fmax_matrix(prob[:, cols], true[:, cols])
    results["overall"] = fmax_matrix(prob, true)
    return results
