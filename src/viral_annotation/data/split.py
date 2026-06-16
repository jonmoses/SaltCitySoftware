"""Train/val/test split with the asymmetric evidence rule.

Because we only trust manual annotations as ground truth (val/test), evaluation
proteins must have >=1 manual annotation. So:

  * val and test are drawn ONLY from manual-having proteins,
  * every IEA-only protein goes to train (its IEA labels are still useful signal).

`SPLIT_RATIOS` apply to the manual-having pool; train then additionally absorbs
all IEA-only proteins. Seeded for determinism. Each protein keeps its lineage, so
swapping to a held-out viral family later is a filter, not a rewrite.

This is a RANDOM split — close homologs can land on both sides. Per the plan,
numbers are not final until the 30%-identity cluster split lands.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from viral_annotation.config import SPLIT_RATIOS, SPLIT_SEED


@dataclass
class Split:
    train: list   # list[LabeledProtein] — manual-having (minus val/test) + all IEA-only
    val: list     # list[LabeledProtein] — manual-having only
    test: list    # list[LabeledProtein] — manual-having only
    holdout: list = field(default_factory=list)  # held-out family, manual-having (zero-shot)

    def summary(self) -> str:
        n_iea_only_train = sum(1 for p in self.train if not p.has_manual)
        s = (
            f"train={len(self.train)} (IEA-only {n_iea_only_train}, "
            f"manual {len(self.train) - n_iea_only_train}) | "
            f"val={len(self.val)} | test={len(self.test)}"
        )
        if self.holdout:
            s += f" | holdout={len(self.holdout)}"
        return s


def family_of(lineage, suffixes: tuple[str, ...] = ("viridae",)) -> str | None:
    """The family-rank clade from a UniProt lineage — first clade ending in one of
    `suffixes`. Default 'viridae' is the viral ICTV family rank; bacteria use
    'aceae' (LPSN/NCBI). `str.endswith` takes the suffix tuple directly."""
    low = tuple(s.lower() for s in suffixes)
    for clade in lineage:
        if clade.lower().endswith(low):
            return clade
    return None


def split_proteins(
    proteins: list,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = SPLIT_SEED,
) -> Split:
    """Split LabeledProtein records per the asymmetric rule above."""
    manual = [p for p in proteins if p.has_manual]
    iea_only = [p for p in proteins if not p.has_manual]

    rng = random.Random(seed)
    idx = list(range(len(manual)))
    rng.shuffle(idx)

    _, val_frac, test_frac = ratios
    n = len(manual)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))

    test = [manual[i] for i in idx[:n_test]]
    val = [manual[i] for i in idx[n_test:n_test + n_val]]
    train_manual = [manual[i] for i in idx[n_test + n_val:]]

    return Split(train=train_manual + iea_only, val=val, test=test)


def cluster_split(
    proteins: list,
    clusters: dict[str, str],
    holdout_family: str | None = None,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = SPLIT_SEED,
    family_suffixes: tuple[str, ...] = ("viridae",),
) -> Split:
    """Identity-cluster split with optional whole-family holdout (docs/03).

    Combines three constraints:
      * Family holdout — every protein of `holdout_family` is removed from
        train/val/test entirely; its manual-having members become the zero-shot
        `holdout` set.
      * Cluster integrity — whole clusters go to one bucket, so no test/val
        protein has a >=30%-identity homolog in train. IEA-only members of
        val/test clusters are dropped (they can't go to train without leaking).
      * Asymmetric evidence — val/test contain only manual-having proteins;
        IEA-only proteins (and clusters) feed train.

    `clusters` maps accession -> cluster representative (from cluster_sequences),
    computed over all `proteins`.
    """
    holdout = []
    pool = []
    for p in proteins:
        if holdout_family and family_of(p.lineage, family_suffixes) == holdout_family:
            if p.has_manual:
                holdout.append(p)
        else:
            pool.append(p)

    by_cluster: dict[str, list] = defaultdict(list)
    for p in pool:
        by_cluster[clusters.get(p.accession, p.accession)].append(p)

    manual_clusters = [c for c, members in by_cluster.items()
                       if any(p.has_manual for p in members)]
    rng = random.Random(seed)
    rng.shuffle(manual_clusters)
    n = len(manual_clusters)
    _, val_frac, test_frac = ratios
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    test_c = set(manual_clusters[:n_test])
    val_c = set(manual_clusters[n_test:n_test + n_val])

    train, val, test = [], [], []
    for c, members in by_cluster.items():
        if c in test_c:
            test += [p for p in members if p.has_manual]
        elif c in val_c:
            val += [p for p in members if p.has_manual]
        else:
            train += members  # all members (incl. IEA-only) train

    return Split(train=train, val=val, test=test, holdout=holdout)
