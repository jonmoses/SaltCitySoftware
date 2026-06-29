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


def m_aupr(prob_matrix, true_matrix) -> float:
    """Term-centric AUPR (NetGO's "M-AUPR"): mean average-precision over GO terms.

    For each term (column) with >=1 positive, area under its precision-recall curve;
    averaged across terms. Complements protein-centric Fmax.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score

    prob = np.asarray(prob_matrix)
    true = np.asarray(true_matrix) > 0
    scores = []
    for c in range(true.shape[1]):
        if true[:, c].any():
            scores.append(average_precision_score(true[:, c], prob[:, c]))
    return float(np.mean(scores)) if scores else 0.0


def information_accretion(label_sets, dag) -> dict[str, float]:
    """IA(t) = -log2 P(t | parents(t)) from a reference annotation corpus.

    Because the true-path rule makes a term imply its parents, P(t|parents) =
    count(t) / count(proteins carrying ALL of t's direct parents). Rare, specific
    terms get high IA; general terms low. Used to weight Smin.
    """
    import math
    from collections import Counter

    sets = [set(s) for s in label_sets]
    n = len(sets)
    cnt: Counter = Counter()
    for s in sets:
        cnt.update(s)

    ia: dict[str, float] = {}
    for t, c in cnt.items():
        term = dag.get(t)
        parents = {p for p in (term.parents if term else set()) if p in cnt}
        denom = n if not parents else sum(1 for s in sets if parents <= s)
        p_cond = c / denom if denom else 0.0
        ia[t] = -math.log2(p_cond) if 0.0 < p_cond <= 1.0 else 0.0
    return ia


def smin(prob_matrix, true_matrix, ia, terms, thresholds=None) -> float:
    """Smin (CAFA): min over threshold of sqrt(remaining-uncertainty^2 + misinformation^2).

    ru = IA-weighted true terms missed; mi = IA-weighted terms wrongly predicted;
    each averaged over proteins. Lower is better. `terms` aligns columns to IA keys.
    """
    import numpy as np

    prob = np.asarray(prob_matrix)
    true = np.asarray(true_matrix) > 0
    ia_vec = np.array([ia.get(t, 0.0) for t in terms], dtype="float64")
    if thresholds is None:
        thresholds = np.linspace(0.01, 1.0, 100)

    best = float("inf")
    for tau in thresholds:
        pred = prob >= tau
        ru = ((true & ~pred).astype("float64") @ ia_vec).mean()   # missed
        mi = ((pred & ~true).astype("float64") @ ia_vec).mean()   # wrong
        best = min(best, float(np.sqrt(ru * ru + mi * mi)))
    return best


def paired_bootstrap_delta(prob_a, prob_b, true_matrix, metric, *,
                           n_boot=1000, seed=1337, ci=0.95):
    """Paired bootstrap CI for the metric delta metric(b) - metric(a).

    `prob_a`, `prob_b`, and `true_matrix` are aligned [P x N] matrices over the
    SAME proteins (e.g. two models scored on one test set). `metric` is a callable
    `(prob[P x N], true[P x N]) -> float`. Protein rows are resampled with
    replacement, the SAME indices for both models (paired), so the CI reflects the
    per-protein variability of the difference rather than each model's marginal
    spread. Returns (observed_delta, ci_low, ci_high); a CI excluding 0 means the
    difference is significant at the (1 - ci) level.
    """
    import numpy as np

    a = np.asarray(prob_a)
    b = np.asarray(prob_b)
    Y = np.asarray(true_matrix)
    observed = float(metric(b, Y) - metric(a, Y))
    P = Y.shape[0]
    if P == 0 or n_boot <= 0:
        return observed, observed, observed
    rng = np.random.RandomState(seed)
    deltas = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        idx = rng.randint(0, P, size=P)
        deltas[i] = metric(b[idx], Y[idx]) - metric(a[idx], Y[idx])
    half = (1.0 - ci) / 2.0 * 100.0
    lo, hi = np.percentile(deltas, [half, 100.0 - half])
    return observed, float(lo), float(hi)


def _fmax_smin_perprotein(prob, true_bool, ia_vec, thresholds):
    """Precompute per-(threshold, protein) sufficient statistics for Fmax and Smin.

    Resampling proteins (the bootstrap) only changes *which* proteins are averaged,
    not the per-protein/per-threshold quantities — so we compute those once here
    ([T x P], collapsing the N dimension) and then each resample is a cheap row
    average with no N factor. Returns numpy arrays, all shape [T, P]:

      prec_per   tp/pred_count   (0 where the protein predicts nothing at tau)
      predmask   protein predicts >=1 term at tau (precision is averaged only here)
      rec_per    tp/true_count   (0 where the protein has no true terms)
      ru_per     IA-weighted true terms missed   (Smin "remaining uncertainty")
      mi_per     IA-weighted terms wrongly called (Smin "misinformation")
    """
    import numpy as np

    P, _ = prob.shape
    T = len(thresholds)
    true_count = true_bool.sum(axis=1).astype("float64")
    nz_true = true_count > 0
    prec_per = np.zeros((T, P), dtype="float64")
    predmask = np.zeros((T, P), dtype="bool")
    rec_per = np.zeros((T, P), dtype="float64")
    ru_per = np.zeros((T, P), dtype="float64")
    mi_per = np.zeros((T, P), dtype="float64")
    for t, tau in enumerate(thresholds):
        pred = prob >= tau
        pred_count = pred.sum(axis=1).astype("float64")
        tp = (pred & true_bool).sum(axis=1).astype("float64")
        has_pred = pred_count > 0
        predmask[t] = has_pred
        prec_per[t, has_pred] = tp[has_pred] / pred_count[has_pred]
        rec_per[t, nz_true] = tp[nz_true] / true_count[nz_true]
        ru_per[t] = (true_bool & ~pred).astype("float64") @ ia_vec
        mi_per[t] = (pred & ~true_bool).astype("float64") @ ia_vec
    return prec_per, predmask, rec_per, ru_per, mi_per


def _fmax_from_stats(prec_per, predmask, rec_per, idx):
    import numpy as np

    pm = predmask[:, idx]
    denom = pm.sum(axis=1)
    prec = np.zeros(prec_per.shape[0], dtype="float64")
    ok = denom > 0
    prec[ok] = (prec_per[:, idx] * pm)[ok].sum(axis=1) / denom[ok]
    rec = rec_per[:, idx].mean(axis=1)
    s = prec + rec
    f1 = np.zeros_like(prec)
    nz = s > 0
    f1[nz] = 2 * prec[nz] * rec[nz] / s[nz]
    return float(f1.max())


def _smin_from_stats(ru_per, mi_per, idx):
    import numpy as np

    ru = ru_per[:, idx].mean(axis=1)
    mi = mi_per[:, idx].mean(axis=1)
    return float(np.sqrt(ru * ru + mi * mi).min())


def paired_bootstrap_fmax_smin(prob_a, prob_b, true_matrix, ia, terms, *,
                               n_boot=1000, seed=1337, ci=0.95, thresholds=None):
    """Fast paired bootstrap CIs for the Fmax and Smin deltas (b - a) at once.

    Equivalent to calling `paired_bootstrap_delta` with `fmax_matrix` and `smin`,
    but ~3 orders of magnitude faster on wide matrices: the per-protein/per-threshold
    statistics are precomputed once (see `_fmax_smin_perprotein`) so each of the
    `n_boot` resamples is an O(thresholds x P) row average instead of a fresh
    O(thresholds x P x N) sweep. Returns
    {"fmax": {delta, lo, hi}, "smin": {delta, lo, hi}}; a CI excluding 0 (for Fmax)
    or a Smin CI excluding 0 means the difference is significant.
    """
    import numpy as np

    a = np.asarray(prob_a, dtype="float32")
    b = np.asarray(prob_b, dtype="float32")
    Y = np.asarray(true_matrix) > 0
    P = Y.shape[0]
    if thresholds is None:
        thresholds = np.linspace(0.01, 1.0, 100)
    ia_vec = np.array([ia.get(t, 0.0) for t in terms], dtype="float64")

    sa = _fmax_smin_perprotein(a, Y, ia_vec, thresholds)
    sb = _fmax_smin_perprotein(b, Y, ia_vec, thresholds)
    full = np.arange(P)
    obs_f = _fmax_from_stats(sb[0], sb[1], sb[2], full) - _fmax_from_stats(sa[0], sa[1], sa[2], full)
    obs_s = _smin_from_stats(sb[3], sb[4], full) - _smin_from_stats(sa[3], sa[4], full)
    if P == 0 or n_boot <= 0:
        return {"fmax": {"delta": obs_f, "lo": obs_f, "hi": obs_f},
                "smin": {"delta": obs_s, "lo": obs_s, "hi": obs_s}}

    rng = np.random.RandomState(seed)
    df = np.empty(n_boot, dtype="float64")
    ds = np.empty(n_boot, dtype="float64")
    for i in range(n_boot):
        idx = rng.randint(0, P, size=P)
        df[i] = _fmax_from_stats(sb[0], sb[1], sb[2], idx) - _fmax_from_stats(sa[0], sa[1], sa[2], idx)
        ds[i] = _smin_from_stats(sb[3], sb[4], idx) - _smin_from_stats(sa[3], sa[4], idx)
    half = (1.0 - ci) / 2.0 * 100.0
    flo, fhi = np.percentile(df, [half, 100.0 - half])
    slo, shi = np.percentile(ds, [half, 100.0 - half])
    return {"fmax": {"delta": obs_f, "lo": float(flo), "hi": float(fhi)},
            "smin": {"delta": obs_s, "lo": float(slo), "hi": float(shi)}}


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
