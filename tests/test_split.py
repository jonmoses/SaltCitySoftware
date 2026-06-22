"""Tests for the asymmetric and cluster-based train/val/test splits."""

from collections import defaultdict

from viral_annotation.data.labels import LabeledProtein
from viral_annotation.data.split import cluster_split, family_of, split_proteins


def _protein(acc: str, manual: bool, family: str = "Testviridae") -> LabeledProtein:
    return LabeledProtein(
        accession=acc,
        sequence="M",
        organism="v",
        lineage=["Viruses", family],
        terms_all=frozenset({"GO:0000003"}),
        terms_manual=frozenset({"GO:0000003"}) if manual else frozenset(),
        n_manual=1 if manual else 0,
        n_iea=0 if manual else 2,
    )


def _make_pop(n_manual: int, n_iea_only: int):
    manual = [_protein(f"M{i}", True) for i in range(n_manual)]
    iea = [_protein(f"I{i}", False) for i in range(n_iea_only)]
    return manual + iea


def test_val_test_are_manual_only_and_iea_in_train():
    pop = _make_pop(n_manual=100, n_iea_only=50)
    s = split_proteins(pop, ratios=(0.70, 0.15, 0.15), seed=0)
    # No IEA-only protein may appear in val/test.
    assert all(p.has_manual for p in s.val)
    assert all(p.has_manual for p in s.test)
    # All IEA-only proteins land in train.
    iea_in_train = sum(1 for p in s.train if not p.has_manual)
    assert iea_in_train == 50


def test_split_sizes():
    pop = _make_pop(n_manual=100, n_iea_only=50)
    s = split_proteins(pop, ratios=(0.70, 0.15, 0.15), seed=0)
    assert len(s.val) == 15 and len(s.test) == 15      # 15% of 100 manual
    assert len(s.train) == (100 - 30) + 50             # rest manual + all iea
    # Partition is exact and non-overlapping.
    accs = [p.accession for p in s.train + s.val + s.test]
    assert len(accs) == len(set(accs)) == 150


def test_split_is_deterministic():
    pop = _make_pop(80, 20)
    a = split_proteins(pop, seed=42)
    b = split_proteins(pop, seed=42)
    assert [p.accession for p in a.test] == [p.accession for p in b.test]
    c = split_proteins(pop, seed=7)
    assert [p.accession for p in a.test] != [p.accession for p in c.test]


# --- cluster split -----------------------------------------------------------
def test_family_of():
    assert family_of(["Viruses", "Riboviria", "Coronaviridae", "Betacoronavirus"]) == "Coronaviridae"
    assert family_of(["Viruses", "Duplodnaviria"]) is None


def test_cluster_split_holds_out_family_and_keeps_val_test_manual():
    prots, clusters = [], {}
    for i in range(20):  # 20 manual-having singletons
        p = _protein(f"M{i}", True); prots.append(p); clusters[p.accession] = p.accession
    for i in range(10):  # 10 IEA-only singletons
        p = _protein(f"I{i}", False); prots.append(p); clusters[p.accession] = p.accession
    for i in range(5):   # 5 Coronaviridae manual -> holdout
        p = _protein(f"C{i}", True, family="Coronaviridae"); prots.append(p); clusters[p.accession] = p.accession

    s = cluster_split(prots, clusters, holdout_families="Coronaviridae",
                      ratios=(0.7, 0.15, 0.15), seed=0)

    assert len(s.holdout) == 5
    placed = s.train + s.val + s.test
    assert not any(p.accession.startswith("C") for p in placed)  # family fully held out
    assert all(p.has_manual for p in s.val + s.test)             # val/test manual-only
    assert sum(1 for p in s.train if not p.has_manual) == 10     # IEA-only singletons -> train


def test_cluster_split_holds_out_multiple_families():
    # Unified domain holds out one viral + one bacterial family at once.
    prots, clusters = [], {}
    for i in range(20):
        p = _protein(f"M{i}", True); prots.append(p); clusters[p.accession] = p.accession
    for i in range(4):
        p = _protein(f"V{i}", True, family="Nairoviridae"); prots.append(p); clusters[p.accession] = p.accession
    for i in range(6):
        p = _protein(f"B{i}", True, family="Francisellaceae"); prots.append(p); clusters[p.accession] = p.accession

    s = cluster_split(prots, clusters, holdout_families=("Nairoviridae", "Francisellaceae"),
                      family_suffixes=("viridae", "aceae"), ratios=(0.7, 0.15, 0.15), seed=0)

    assert len(s.holdout) == 10  # both families fully held out
    placed = s.train + s.val + s.test
    assert not any(p.accession[0] in {"V", "B"} for p in placed)


def test_cluster_split_no_cluster_spans_buckets():
    # Each cluster has a manual + an IEA-only member; whole cluster must stay together.
    prots, clusters = [], {}
    for i in range(24):
        m = _protein(f"c{i}_m", True)
        iea = _protein(f"c{i}_i", False)
        prots += [m, iea]
        clusters[m.accession] = f"c{i}"
        clusters[iea.accession] = f"c{i}"

    s = cluster_split(prots, clusters, holdout_families=None, seed=1)

    bucket = {}
    for name, lst in (("train", s.train), ("val", s.val), ("test", s.test)):
        for p in lst:
            bucket[p.accession] = name
    by_cluster = defaultdict(list)
    for acc, rep in clusters.items():
        by_cluster[rep].append(acc)
    for rep, members in by_cluster.items():
        buckets = {bucket[m] for m in members if m in bucket}
        assert len(buckets) <= 1, f"cluster {rep} leaked across {buckets}"
    assert all(p.has_manual for p in s.val + s.test)
