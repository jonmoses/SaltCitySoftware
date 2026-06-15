"""Tests for the asymmetric train/val/test split."""

from viral_annotation.data.labels import LabeledProtein
from viral_annotation.data.split import split_proteins


def _protein(acc: str, manual: bool) -> LabeledProtein:
    return LabeledProtein(
        accession=acc,
        sequence="M",
        organism="v",
        lineage=["Viruses"],
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
