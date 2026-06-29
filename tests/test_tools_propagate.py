"""Contract test for the propagate tool: annotations.tsv -> labels.tsv."""

from valib.artifacts import read_tsv3, write_tsv3
from tools.propagate import main as propagate_main


def test_propagate_tool_closes_per_tier(tiny_obo_path, tmp_path):
    annotations = tmp_path / "annotations.tsv"
    # P1: a manual leaf and an IEA sibling leaf. P2: a BP leaf, manual.
    write_tsv3(
        annotations,
        [
            ("P1", "GO:0000003", "manual"),
            ("P1", "GO:0000004", "iea"),
            ("P2", "GO:0000011", "manual"),
        ],
    )
    out = tmp_path / "labels.tsv"
    rc = propagate_main(
        ["--obo", str(tiny_obo_path), "--in", str(annotations), "--out", str(out)]
    )
    assert rc == 0
    rows = set(read_tsv3(out))
    # Manual tier of P1 closes 0003 -> 0002 -> root, all tagged manual.
    assert ("P1", "GO:0000002", "manual") in rows
    assert ("P1", "GO:0003674", "manual") in rows
    # IEA tier closes independently and keeps its own tier tag.
    assert ("P1", "GO:0000004", "iea") in rows
    assert ("P1", "GO:0000002", "iea") in rows
    # P2 BP chain.
    assert ("P2", "GO:0008150", "manual") in rows


def test_propagate_tool_output_is_sorted(tiny_obo_path, tmp_path):
    annotations = tmp_path / "annotations.tsv"
    write_tsv3(annotations, [("P2", "GO:0000011", "manual"), ("P1", "GO:0000003", "manual")])
    out = tmp_path / "labels.tsv"
    propagate_main(["--obo", str(tiny_obo_path), "--in", str(annotations), "--out", str(out)])
    rows = list(read_tsv3(out))
    assert rows == sorted(rows)
