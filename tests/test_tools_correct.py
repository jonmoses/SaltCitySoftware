"""Contract test for the correct tool: pred.tsv -> pred.corrected.tsv."""

from valib.artifacts import read_tsv3, write_tsv3
from tools.correct import main as correct_main


def test_correct_tool_lifts_parents(tiny_obo_path, tmp_path):
    pred = tmp_path / "pred.tsv"
    write_tsv3(
        pred,
        [
            ("P1", "GO:0003674", "0.10"),
            ("P1", "GO:0000002", "0.20"),
            ("P1", "GO:0000003", "0.90"),
        ],
    )
    out = tmp_path / "pred.corrected.tsv"
    rc = correct_main(["--obo", str(tiny_obo_path), "--in", str(pred), "--out", str(out)])
    assert rc == 0
    corrected = {(a, g): float(s) for a, g, s in read_tsv3(out)}
    assert corrected[("P1", "GO:0000003")] == 0.9
    assert corrected[("P1", "GO:0000002")] == 0.9
    assert corrected[("P1", "GO:0003674")] == 0.9


def test_correct_tool_leaves_consistent_scores(tiny_obo_path, tmp_path):
    pred = tmp_path / "pred.tsv"
    # Already DAG-consistent (parent >= child): unchanged.
    write_tsv3(pred, [("P1", "GO:0000002", "0.80"), ("P1", "GO:0000003", "0.30")])
    out = tmp_path / "pred.corrected.tsv"
    correct_main(["--obo", str(tiny_obo_path), "--in", str(pred), "--out", str(out)])
    corrected = {(a, g): float(s) for a, g, s in read_tsv3(out)}
    assert corrected[("P1", "GO:0000002")] == 0.8
    assert corrected[("P1", "GO:0000003")] == 0.3
